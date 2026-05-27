# Metric-Guided Feature Fusion of Visual Foundation Models for Segmentation Tasks


<p align="center">
  <a href="https://cvpr.thecvf.com/Conferences/2026">
    <img src="https://img.shields.io/badge/CVPR%202026-Findings-blue.svg">
  </a>
  <a href="https://arxiv.org/pdf/2605.16864">
    <img src="https://img.shields.io/badge/arXiv-2605.16864-b31b1b.svg">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg">
  </a>
</p>

<p align="center">
  <b>CVPR 2026 Findings Track</b>
</p>

## 📢 News

- **[2026.05]** Code released! 🎉
- **[2026.03]** Paper accepted to CVPR 2026 Findings Track!

## 📖 Abstract

Although large-scale visual foundation models (VFMs) achieve remarkable performance in semantic understanding, they still underperform in instance-aware dense prediction tasks. We observe that different VFMs exhibit complementary representational biases: **SAM2** focuses on fine-grained object boundaries (*edge-strong*), while **DINOv3** emphasizes object-level structure (*structure-strong*).

We propose a **metric-guided fusion framework** that:
1. Introduces label-free metrics (**Structural Coherence** and **Edge Fidelity**) to quantify VFM biases
2. Identifies complementary encoder pairs based on metric profiles
3. Fuses features via a lightweight **master-auxiliary scheme** with single-stage training

Our method achieves consistent improvements on COCO and Cityscapes benchmarks, with notable gains on boundary-sensitive categories.

## 🔑 Key Contributions

- **Novel Metrics Suite**: Label-free SC/EF metrics for interpretable assessment of VFM representational bias
- **Lightweight Fusion**: Simple master-auxiliary scheme with minimal overhead, no complex architectural changes
- **Task-Agnostic Framework**: Feature-level assessment applicable to any encoder backbone
- **State-of-the-Art Results**: Consistent improvements on semantic and instance segmentation

## 📊 Results

### COCO Instance Segmentation

| Method | Backbone | AP | AP<sub>75</sub> | AP<sub>s</sub> |
|--------|----------|:--:|:--:|:--:|
| Mask2Former | Swin-B | 44.1 | 47.1 | 22.8 |
| Mask2Former | DINOv3-B (FT) | 46.0 | 49.3 | 24.2 |
| Mask2Former | SAM2-B (FZ) | 35.8 | 37.6 | 19.2 |
| **Ours** | **ViT-B (Hybrid)** | **47.3** | **51.4** | **27.3** |

### Cityscapes Segmentation

| Method | Backbone | AP | mIoU |
|--------|----------|:--:|:--:|
| Mask2Former | Swin-B | 38.0 | 80.5 |
| Mask2Former | DINOv3-B (FT) | 35.6 | 81.2 |
| Mask2Former | SAM2-B (FZ) | 35.8 | 79.7 |
| **Ours** | **ViT-B (Hybrid)** | **39.5** | **82.8** |

### SC/EF Metric Profiles

| Backbone | Metric | OS 4 | OS 8 | OS 16 | OS 32 |
|----------|--------|:----:|:----:|:-----:|:-----:|
| DINOv3 | SC | **0.73** | 0.71 | 0.65 | 0.53 |
| DINOv3 | EF | 1.27 | 1.64 | 2.33 | 5.88 |
| SAM2 | SC | 0.49 | 0.44 | 0.11 | 0.41 |
| SAM2 | EF | 6.60 | 8.59 | **17.13** | 12.47 |

> DINOv3 shows high SC (structure-strong), SAM2 shows high EF at OS16 (edge-strong)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/xxx/MetricGuidedVFMFusion.git
cd MetricGuidedVFMFusion

# Create conda environment
conda create -n vfm_fusion python=3.10 -y
conda activate vfm_fusion

# Install dependencies
pip install -r requirements.txt

# Install Mask2Former
cd third_party/Mask2Former
pip install -e .
cd ../..
```

## 📁 Data Preparation

```
data/
├── ade/
│   └── ADEChallengeData2016/
│       ├── annotations/
│       ├── annotations_detectron2/
│       ├── images/
│       ├── objectInfo150.txt
│       └── sceneCategories.txt
├── cityscapes/
│   ├── gtFine/
│   └── leftImg8bit/
└── coco/
    ├── annotations/
    ├── train2017/
    ├── val2017/
```

## 🚀 Quick Start


### Training

```bash
# Train with metric-guided fusion on COCO

CUDA_VISIBLE_DEVICES=0 python train_net_custom_vlm_fuse.py --num-gpus 1 --dist-url   tcp://127.0.0.1:50171 --config-file configs/coco/instance-segmentation/vfm/maskformer2_fuse.yaml  MODEL.BACKBONE.NAME "DinoV3Backbone"    MAPPING "D"   OUTPUT_DIR ./output_vlm_link/fuse_dinov3_fire_base_sam2_freeze_base_replace_4_coco

# Train on Cityscapes

CUDA_VISIBLE_DEVICES=0 python train_net_custom_vlm_fuse.py --num-gpus 1 --dist-url   tcp://127.0.0.1:50172 --config-file configs/cityscapes/instance-segmentation/vlm_fusion/maskformer2_dinov3_sam_large.yaml MODEL.MASK_FORMER.FROZE_BACKBONE False DATASETS.TRAIN "('synscapes_instance_seg_train_eval_cityscapes',)"  MAPPING "D" OUTPUT_DIR ./output_vlm_link/dinov3_large_fire_sam2_large_freeze_D_synscapes2cs

```

### VFM Encoders

We use the following pretrained VFM encoders:

| Encoder | Source | Weights |
|---------|--------|:-------:|
| DINOv3-B | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) 
| SAM2-B | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) 

## 📐 Method Overview

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

### Structural Coherence (SC)

Measures whether regions with similar structure are coherently aggregated in feature space:

- **SFC** (Structured Feature Contrast): Between-patch vs within-patch variance ratio
- **SCS** (Structural Clustering Score): Silhouette coefficient from k-means clustering

```
SC = √(SFC · SCS)
```

### Edge Fidelity (EF)

Assesses whether feature activations concentrate along image boundaries:

- **EC** (Edge Concentration): Gradient energy within edge band
- **NC** (Near-edge Concentration): Ring band concentration
- **FC** (Frequency Content): Medium-to-high frequency energy
- **SP** (Spatial Precision): Sensitivity to spatial shifts

```
EF = 100 · EC · NC · FC · SP
```

### Fusion Strategy

1. Select encoder with highest SC as **main** (DINOv3)
2. Select encoder with highest EF as **auxiliary** (SAM2)
3. Replace main features at stage `s* = argmax_s EF_aux(s)` with auxiliary features
4. Train main encoder, freeze auxiliary encoder


## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{xxx2026metricguided,
  title={Metric-Guided Feature Fusion of Visual Foundation Models for Segmentation Tasks},
  author={xxx},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```

## 🔗 Related Projects

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) - Universal Image Segmentation
- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised Vision Transformers
- [SAM2](https://github.com/facebookresearch/sam2) - Segment Anything in Images and Videos
- [ViT-Adapter](https://github.com/czczup/ViT-Adapter) - Vision Transformer Adapter for Dense Predictions

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

## 🙏 Acknowledgements

We thank the authors of [Mask2Former](https://github.com/facebookresearch/Mask2Former), [DINOv3](https://github.com/facebookresearch/dinov3), and [SAM2](https://github.com/facebookresearch/sam2) for their excellent work and open-source code.

---

<p align="center">
  <i>If you have any questions, please open an issue or contact us.</i>
</p>
