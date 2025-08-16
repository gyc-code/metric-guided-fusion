from hiera import Hiera
import torch
hiera_backbone = Hiera.from_pretrained("facebook/hiera_base_224.mae_in1k_ft_in1k")
print(hiera_backbone)

ckpt = torch.load("pretrain/sam2.1_hiera_large.pt", map_location="cuda")
# 如果 checkpoint 里直接存的是 state_dict:
print(ckpt['model'].keys())
# 3. 把权重载入模型
print("Loading state_dict...")
for k in hiera_backbone.state_dict().keys():
    v = hiera_backbone.state_dict()[k]
    print(k, v.shape)
    if 'image_encoder.'+k in ckpt['model'].keys():
        print(ckpt['model']['image_encoder.'+k].shape)
        print('Loading', k)