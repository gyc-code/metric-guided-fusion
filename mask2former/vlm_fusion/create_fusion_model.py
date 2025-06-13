import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# --- CBAM 注意力模块，与之前一致 ---
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2,1,7,padding=3,bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        x = x * torch.sigmoid(avg + mx)
        y = torch.cat([x.mean(1,True), x.max(1,True)[0]], dim=1)
        return x * self.spatial(y)

# --- 静态通道 Fusion 模块（feat1：[128,256,512,1024], feat2:256）---
class SemanticEdgeFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = ['res2','res3','res4','res5']
        # 最新的 feature1 通道数
        f1_ch = {'res2':128,'res3':256,'res4':512,'res5':1024}
        f2_ch = {'res2':256,'res3':256,'res4':256,'res5':256}
        mid_ch = 512

        # 1×1 投影 + BN
        self.proj1   = nn.ModuleDict({k: nn.Conv2d(f1_ch[k], mid_ch, 1, bias=False)
                                      for k in self.keys})
        self.bn1     = nn.ModuleDict({k: nn.BatchNorm2d(mid_ch) for k in self.keys})
        self.proj2   = nn.ModuleDict({k: nn.Conv2d(f2_ch[k], mid_ch, 1, bias=False)
                                      for k in self.keys})
        self.bn2     = nn.ModuleDict({k: nn.BatchNorm2d(mid_ch) for k in self.keys})

        # 拼接后降回 feature1 通道
        self.fuse    = nn.ModuleDict({k: nn.Conv2d(mid_ch*2, f1_ch[k], 1, bias=False)
                                      for k in self.keys})
        self.bn_fuse = nn.ModuleDict({k: nn.BatchNorm2d(f1_ch[k]) for k in self.keys})

        # 可学习残差因子
        self.alpha   = nn.ParameterDict({k: nn.Parameter(torch.tensor(0.2))
                                         for k in self.keys})

        # CBAM + 平滑
        self.cbam   = nn.ModuleDict({k: CBAM(f1_ch[k]) for k in self.keys})
        self.smooth = nn.ModuleDict({k: nn.Sequential(
                                        nn.Conv2d(f1_ch[k], f1_ch[k], 3, padding=1, bias=False),
                                        nn.BatchNorm2d(f1_ch[k]),
                                        nn.ReLU(inplace=True),
                                    ) for k in self.keys})

    def forward(self, feat1: dict, feat2: dict):
        out = {}
        for k in self.keys:
            x1 = feat1[k]  # 可能是 float16 on CUDA
            x2 = feat2[k]

            # 1) 把输入 cast 到 proj1[k].weight.dtype（通常是 float32）
            wt_dtype = self.proj1[k].weight.dtype
            if x1.dtype != wt_dtype:
                x1 = x1.to(dtype=wt_dtype)
            if x2.dtype != wt_dtype:
                x2 = x2.to(dtype=wt_dtype)

            # 2) 原 Fusion 逻辑（全在 float32 下安全运行）
            p1 = F.relu(self.bn1[k](self.proj1[k](x1)), inplace=False)
            up = F.interpolate(x2, size=x1.shape[-2:], 
                               mode='bilinear', align_corners=False)
            p2 = F.relu(self.bn2[k](self.proj2[k](up)), inplace=False)

            cat = torch.cat([p1, p2], dim=1)
            f   = self.bn_fuse[k](self.fuse[k](cat))
            f   = x1 + self.alpha[k] * f
            f   = self.cbam[k](f)
            f   = self.smooth[k](f)

            # 3) 再 cast 回原始 x1.dtype（如 float16）
            orig_dtype = feat1[k].dtype
            if f.dtype != orig_dtype:
                f = f.to(dtype=orig_dtype)

            out[k] = f

        return out


# === 测试前向 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SemanticEdgeFusion().to(device)

    # 随机模拟半精度输入
    feat1 = {
        'res2': torch.randn(1,128,256,256, device=device).half(),
        'res3': torch.randn(1,256,128,128, device=device).half(),
        'res4': torch.randn(1,512,64,64,   device=device).half(),
        'res5': torch.randn(1,1024,32,32,  device=device).half(),
    }
    feat2 = {
        'res2': torch.randn(1,256,64,64, device=device).half(),
        'res3': torch.randn(1,256,64,64, device=device).half(),
        'res4': torch.randn(1,256,64,64, device=device).half(),
        'res5': torch.randn(1,256,64,64, device=device).half(),
    }

    out = model(feat1, feat2)
    for k in model.keys:
        print(f"{k}: shape={out[k].shape}, dtype={out[k].dtype}, device={out[k].device}")
