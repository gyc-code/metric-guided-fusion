import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionBlock(nn.Module):
    """
    Efficient FusionBlock with residual-preserving DINO semantics and lightweight fusion.
    """
    def __init__(self, d_channels, s_channels):
        super(FusionBlock, self).__init__()
        # 1x1 projections
        self.proj_d = nn.Conv2d(d_channels, d_channels, 1)
        self.proj_s = nn.Conv2d(s_channels, d_channels, 1)
        # SE-style channel attention (bottleneck)
        mid = max(d_channels // 32, 4)
        self.chan_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, d_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # spatial attention (light)
        self.spat_att = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # dynamic weighting for attention paths
        self.weight = nn.Parameter(torch.tensor([0.4, 0.4, 0.2], dtype=torch.float))
        # depthwise separable fusion
        self.dw = nn.Conv2d(d_channels * 2, d_channels * 2, 3, padding=1, groups=d_channels * 2, bias=False)
        self.pw = nn.Conv2d(d_channels * 2, d_channels, 1)
        self.act = nn.ReLU(inplace=True)
        # projection for fusion refinement
        self.refine_proj = nn.Conv2d(d_channels, d_channels, 1)

    def forward(self, d_feat, s_feat):
        # project features
        d = self.proj_d(d_feat)
        s = self.proj_s(s_feat)
        # channel attention path
        ca = self.chan_att(s)
        d_ca = d * ca
        # spatial attention path (average-guided)
        avg = torch.mean(s, dim=1, keepdim=True)
        sa = self.spat_att(avg)
        d_sa = d * sa
        # initial weighted fusion (dino identity + ca + sa)
        w = torch.softmax(self.weight, dim=0)
        fused_init = w[0] * d + w[1] * d_ca + w[2] * d_sa
        # depthwise separable fusion conv
        x = torch.cat([d_ca, d_sa], dim=1)
        x = self.dw(x)
        x = self.act(self.pw(x))
        # refine fusion
        x = self.refine_proj(x)
        # final output preserves DINO semantic residual
        out = fused_init + x
        return out

class FeatureFusionHead(nn.Module):
    """
    Efficient Fusion Head preserving DINO semantics with top-down FPN.
    """
    def __init__(self, d_channels_list, s_channels):
        super(FeatureFusionHead, self).__init__()
        levels = len(d_channels_list)
        self.blocks = nn.ModuleList([
            FusionBlock(d_c, s_channels)
            for d_c in d_channels_list
        ])
        # top-down lateral convs for FPN merge
        self.laterals = nn.ModuleList([
            nn.Conv2d(d_channels_list[i+1], d_channels_list[i], 1)
            for i in range(levels - 1)
        ])

    def forward(self, d_feats: dict, s_feats: dict):
        keys = ['res2', 'res3', 'res4', 'res5']
        fused = []
        # per-scale fusion
        for i, k in enumerate(keys):
            d = d_feats[k]
            s = s_feats[k]
            if s.shape[-2:] != d.shape[-2:]:
                s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False)
            fused.append(self.blocks[i](d, s))
        # top-down merge
        outputs = [fused[-1]]
        for i in range(len(fused) - 1, 0, -1):
            up = F.interpolate(outputs[0], size=fused[i-1].shape[-2:], mode='bilinear', align_corners=False)
            td = self.laterals[i-1](up)
            outputs.insert(0, td + fused[i-1])
        # return dict matching input dims
        return {k: outputs[j] for j, k in enumerate(keys)}

# Example test
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_feats = {
        'res2': torch.randn(1, 128, 256, 256, device=device).half(),
        'res3': torch.randn(1, 256, 128, 128, device=device).half(),
        'res4': torch.randn(1, 512,  64,  64, device=device).half(),
        'res5': torch.randn(1,1024,  32,  32, device=device).half(),
    }
    sam_feats = {k: torch.randn(1,256,64,64, device=device).half() for k in dino_feats}
    head = FeatureFusionHead([128,256,512,1024], 256).to(device).half()


    # dino_feats = {
    #     'res2': torch.randn(1, 128, 256, 256, device=device),
    #     'res3': torch.randn(1, 256, 128, 128, device=device),
    #     'res4': torch.randn(1, 512,  64,  64, device=device),
    #     'res5': torch.randn(1,1024,  32,  32, device=device),
    # }
    # sam_feats = {k: torch.randn(1,256,64,64, device=device) for k in dino_feats}
    # head = FeatureFusionHead([128,256,512,1024], 256).to(device)

    out = head(dino_feats, sam_feats)
    for k,v in out.items():
        print(f"{k}: {v.shape}, dtype={v.dtype}")

if __name__ == '__main__':
    test()
