# General SISR Framework

A complete, production-grade **Single Image Super-Resolution (SISR)** framework implemented from scratch in pure PyTorch, covering all major model families with a unified training, evaluation, and inference pipeline.

## Supported Models

| Family | Models | Objective |
|--------|--------|-----------|
| Interpolation | Nearest Neighbor, Bilinear, Bicubic | Baseline |
| Shallow CNN | SRCNN | PSNR |
| Deep Residual CNN | VDSR, EDSR, RCAN | PSNR |
| Lightweight | ESPCN, IMDN | PSNR / Edge |
| GAN-based | SRGAN, ESRGAN, Real-ESRGAN | Perceptual |
| Transformer | SwinIR, HAT | PSNR / Perceptual |
| Diffusion | SR3 | Generative |

## Quick Start

```bash
pip install -r requirements.txt

# Train a model
python train.py --config configs/edsr_x4.yaml

# Evaluate a trained model on benchmark datasets
python test.py --config configs/edsr_x4.yaml --checkpoint outputs/edsr_x4/best.pth

# Super-resolve a single image
python infer.py --model edsr --scale 4 --img path/to/lr_image.png --out path/to/output.png

# Run full benchmark comparison
python -m inference.benchmark --checkpoint_dir outputs/
```

## Repository Structure

```
General SISR Framework/
├── configs/            YAML configs per model × scale factor
├── data/               Dataset classes, augmentation, degradation pipelines
├── models/             All model implementations
│   ├── interpolation/  Bicubic, Bilinear, Nearest
│   ├── cnn/            SRCNN, VDSR, EDSR, RCAN, ESPCN
│   ├── gan/            SRGAN, ESRGAN, Real-ESRGAN
│   ├── transformer/    SwinIR, HAT
│   ├── diffusion/      SR3
│   └── lightweight/    IMDN
├── losses/             Pixel, Perceptual, Adversarial, Frequency losses
├── metrics/            PSNR, SSIM, LPIPS, NIQE
├── trainers/           PSNR, GAN, Diffusion trainers
├── inference/          Tiled inference, benchmark runner
├── utils/              Logger, checkpoint, visualization
├── notebooks/          Jupyter notebooks for overview and demos
├── train.py
├── test.py
└── infer.py
```

## Benchmark Datasets

| Dataset | Split | Notes |
|---------|-------|-------|
| DIV2K | Train (800) / Val (100) | High-quality 2K images |
| Set5 | Test (5) | Classic benchmark |
| Set14 | Test (14) | Classic benchmark |
| BSD100 | Test (100) | Berkeley Segmentation |
| Urban100 | Test (100) | Urban scenes |

Download via:
```bash
bash scripts/download_datasets.sh
```

## Training

### PSNR-Oriented Models (SRCNN, VDSR, EDSR, RCAN, SwinIR, HAT, ESPCN, IMDN)

```bash
python train.py --config configs/edsr_x4.yaml
```

### GAN-Based Models (SRGAN, ESRGAN, Real-ESRGAN)

```bash
python train.py --config configs/esrgan_x4.yaml
```

GAN training first pre-trains the generator with pixel loss, then fine-tunes with adversarial + perceptual losses.

### Diffusion Model (SR3)

```bash
python train.py --config configs/sr3_x4.yaml
```

## GPU Constraints (RTX 3050 Ti — 4GB VRAM)

All models support:
- **Mixed precision** (`amp`) — enabled by default
- **Gradient accumulation** — configured via `trainer.grad_accumulation_steps`
- **Tiled inference** — overlap-tile strategy for inference on any image size

Recommended configs for 4GB VRAM:

| Model | patch_size | batch_size | grad_accum |
|-------|-----------|------------|------------|
| EDSR | 48 | 8 | 4 |
| RCAN | 48 | 4 | 8 |
| SwinIR | 64 | 4 | 8 |
| HAT | 64 | 2 | 16 |
| SR3 | 64 | 2 | 8 |

## Model Taxonomy

```
SISR Models
├── Axis 1: Architecture
│   ├── CNN (local receptive field)
│   ├── GAN (adversarial training)
│   ├── Transformer (global attention)
│   └── Diffusion (iterative generative)
└── Axis 2: Objective
    ├── PSNR-driven (MSE / L1)
    ├── Perceptual-driven (GAN + VGG)
    └── Generative prior-driven (Diffusion)
```

## Citation

If you use this framework, please cite the original papers for each model you use.
Key references: SRCNN (Dong et al., 2014), EDSR (Lim et al., 2017), RCAN (Zhang et al., 2018),
SRGAN (Ledig et al., 2017), ESRGAN (Wang et al., 2018), Real-ESRGAN (Wang et al., 2021),
SwinIR (Liang et al., 2021), HAT (Chen et al., 2023), SR3 (Saharia et al., 2022),
ESPCN (Shi et al., 2016), IMDN (Hui et al., 2019).
