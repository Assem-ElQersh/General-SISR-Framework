# General SISR Framework

A complete, production-grade **Single Image Super-Resolution (SISR)** framework implemented from scratch in pure PyTorch, covering all major model families with a unified training, evaluation, and inference pipeline.

---

## Supported Models

### Standard SR

| Family | Models | Objective |
|--------|--------|-----------|
| Interpolation | Nearest Neighbor, Bilinear, Bicubic | Baseline |
| Shallow CNN | SRCNN, FSRCNN | PSNR |
| Deep Residual CNN | VDSR, EDSR, RCAN, DRRN, RDN, DBPN | PSNR |
| Lightweight | ESPCN, IMDN | PSNR / Edge |
| GAN-based | SRGAN, ESRGAN, Real-ESRGAN | Perceptual |
| Transformer | SwinIR, HAT, Restormer | PSNR / Perceptual |
| Normalizing Flow | SRFlow | Generative |
| Implicit Neural | LIIF | Arbitrary-scale |
| SSM / Mamba | MambaIR | PSNR |
| Diffusion | SR3, DDNM (zero-shot) | Generative |

### Physics-Informed SR

| Model | Key Idea |
|-------|----------|
| DASR | Contrastive degradation encoder + SPADE-conditioned SR |
| IKC | Iterative kernel estimation + kernel-conditioned SR (SFTMD) |
| DPSR | Half-Quadratic Splitting with plug-and-play denoiser prior |
| UnrolledSR | Learned unrolling of HQS iterations |

### 3D Volumetric SR

| Model | Key Idea |
|-------|----------|
| SRCNN3D | Direct 3D port of SRCNN with trilinear pre-upsample |
| EDSR3D | 3D EDSR with VoxelShuffle, supports anisotropic scale |
| MedSR | Anisotropic MRI SR (z-axis only); 3D or slice-wise mode |

---

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

# Zero-shot diffusion SR (no training needed — just provide a pretrained DDPM)
python infer.py --config configs/ddnm_x4.yaml --img path/to/lr.png
```

---

## Repository Structure

```
General SISR Framework/
├── configs/              YAML configs per model × scale factor
├── data/
│   ├── datasets.py       PairedSRDataset, DIV2KDataset, BenchmarkDataset
│   ├── transforms.py     SRTransform (random crop, flip, rotate)
│   ├── degradations.py   Real-ESRGAN degradation pipeline
│   ├── kernel_estimation.py   Blind kernel generation (iso/aniso/motion)
│   └── volumetric_dataset.py  NIfTI / DICOM 3D patch dataset
├── models/
│   ├── interpolation/    Bicubic, Bilinear, Nearest
│   ├── cnn/              SRCNN, VDSR, EDSR, RCAN, ESPCN, FSRCNN, DRRN, RDN, DBPN
│   ├── gan/              SRGAN, ESRGAN, Real-ESRGAN
│   ├── transformer/      SwinIR, HAT, Restormer
│   ├── flow/             SRFlow
│   ├── implicit/         LIIF
│   ├── mamba/            MambaIR
│   ├── diffusion/        SR3, DDNM
│   ├── lightweight/      IMDN
│   ├── physics/          DASR, IKC, DPSR, UnrolledSR
│   └── volumetric/       Base3D, VoxelShuffle, SRCNN3D, EDSR3D, MedSR
├── losses/               Pixel, Perceptual, Adversarial, Frequency, Physics
├── metrics/              PSNR, SSIM, LPIPS, NIQE
├── trainers/             PSNR, GAN, Diffusion, Flow trainers
├── inference/            Tiled inference, benchmark runner
├── utils/                Logger, checkpoint, visualization
├── notebooks/            Jupyter notebooks for overview and demos
├── train.py
├── test.py
└── infer.py
```

---

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

---

## Training

### PSNR-Oriented Models

```bash
python train.py --config configs/edsr_x4.yaml
python train.py --config configs/rdn_x4.yaml
python train.py --config configs/mambair_x4.yaml
python train.py --config configs/restormer_x4.yaml
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

### Normalizing Flow (SRFlow)

```bash
python train.py --config configs/srflow_x4.yaml
```

SRFlow uses the `flow` trainer with NLL loss. No pixel loss is needed.

### Physics-Informed SR

```bash
# Blind SR with kernel estimation
python train.py --config configs/dasr_x4.yaml
python train.py --config configs/ikc_x4.yaml

# Plug-and-play (train the denoiser only)
python train.py --config configs/dpsr_x4.yaml
```

The `physics` loss key in configs enables forward-model consistency regularization:

```yaml
losses:
  pixel:
    type: L1
    weight: 1.0
  physics:
    weight: 0.1
    scale: 4
```

### 3D Volumetric SR

```bash
# Isotropic 3D SR
python train.py --config configs/edsr3d_x2.yaml

# Anisotropic MRI z-axis SR
python train.py --config configs/med_sr_x4.yaml
```

Volumetric training requires NIfTI files (`.nii.gz`) in `datasets/volumetric/`.

---

## GPU Constraints (RTX 3050 Ti — 4GB VRAM)

All models support:
- **Mixed precision** (`amp`) — enabled by default
- **Gradient accumulation** — configured via `trainer.grad_accumulation_steps`
- **Tiled inference** — overlap-tile strategy for inference on any image size

Recommended configs for 4GB VRAM:

| Model | patch_size | batch_size | grad_accum |
|-------|-----------|------------|------------|
| EDSR | 48 | 8 | 4 |
| RDN | 48 | 8 | 4 |
| RCAN | 48 | 4 | 8 |
| SwinIR | 64 | 4 | 8 |
| HAT | 64 | 2 | 16 |
| Restormer | 128 | 4 | 4 |
| MambaIR | 64 | 8 | 2 |
| SR3 | 64 | 2 | 8 |
| SRFlow | 160 | 8 | 2 |
| EDSR3D | 32³ | 4 | 2 |

---

## Model Taxonomy

```
SISR Models
├── Axis 1: Architecture
│   ├── CNN (local receptive field)
│   │   ├── Standard: SRCNN, FSRCNN, VDSR, EDSR, RCAN
│   │   ├── Recursive: DRRN
│   │   ├── Dense: RDN
│   │   └── Back-Projection: DBPN
│   ├── GAN (adversarial training)
│   │   └── SRGAN, ESRGAN, Real-ESRGAN
│   ├── Transformer (global attention)
│   │   ├── Window: SwinIR, HAT
│   │   └── Transposed-channel: Restormer
│   ├── SSM (state space model)
│   │   └── MambaIR
│   ├── Normalizing Flow
│   │   └── SRFlow
│   ├── Implicit Neural Representation
│   │   └── LIIF
│   ├── Diffusion (iterative generative)
│   │   └── SR3, DDNM (zero-shot)
│   ├── Physics-Informed
│   │   └── DASR, IKC, DPSR, UnrolledSR
│   └── 3D Volumetric
│       └── SRCNN3D, EDSR3D, MedSR
└── Axis 2: Objective
    ├── PSNR-driven (MSE / L1)
    ├── Perceptual-driven (GAN + VGG)
    ├── Generative prior-driven (Diffusion / Flow)
    └── Physics-constrained (degradation consistency)
```

---

## New Model Notes

### LIIF (Arbitrary-Scale SR)
Unlike other models, LIIF takes `(lr, coords, cell)` as input at training time.
The standard `forward(lr)` generates a fixed-scale output; use `model.query(lr, coords, cell)`
for custom scales:

```python
from models import build_model
model = build_model(cfg)
coords, cell = model.make_coords(lr, scale=6)  # 6× SR
sr = model.query(lr, coords, cell)
```

### DDNM (Zero-Shot)
DDNM requires a pretrained unconditional DDPM checkpoint:

```yaml
model:
  name: "ddnm"
  pretrained_ckpt: "path/to/ddpm.pth"
```

No additional training is needed. Just run inference.

### MambaIR
Uses `mamba-ssm` CUDA kernels if installed (faster), otherwise falls back to a
pure-PyTorch selective scan. Install the CUDA variant separately:

```bash
pip install mamba-ssm  # requires compatible CUDA toolkit
```

### Physics-Informed Models
IKC and DPSR accept an optional `kernel` argument:

```python
sr, k_est = ikc_model(lr, k_init=None)   # predicts kernel internally
sr = dpsr_model(lr, kernel=my_kernel)     # use known kernel
```

Use `data/kernel_estimation.py` to generate synthetic kernels for training:

```python
from data.kernel_estimation import batch_random_kernels
kernels = batch_random_kernels(batch_size=16, kernel_size=21)
```

### 3D Volumetric SR
Set `train_dataset: NIfTI` and point `data_root` to a directory of `.nii.gz` files:

```yaml
data:
  train_dataset: "NIfTI"
  data_root: "datasets/mri"
  scale: [4, 1, 1]      # anisotropic z-axis only
  patch_size_3d: 32
```

---

## Citation

If you use this framework, please cite the original papers for each model you use.

Key references:
- SRCNN (Dong et al., ECCV 2014), FSRCNN (Dong et al., ECCV 2016)
- VDSR (Kim et al., CVPR 2016), DRRN (Tai et al., CVPR 2017)
- EDSR (Lim et al., CVPRW 2017), RCAN (Zhang et al., ECCV 2018)
- RDN (Zhang et al., CVPR 2018), DBPN (Haris et al., CVPR 2018)
- ESPCN (Shi et al., CVPR 2016), IMDN (Hui et al., ACM MM 2019)
- SRGAN (Ledig et al., CVPR 2017), ESRGAN (Wang et al., ECCVW 2018)
- Real-ESRGAN (Wang et al., ICCVW 2021)
- SwinIR (Liang et al., ICCVW 2021), HAT (Chen et al., CVPR 2023)
- Restormer (Zamir et al., CVPR 2022)
- SRFlow (Lugmayr et al., ECCV 2020)
- LIIF (Chen et al., CVPR 2021)
- MambaIR (Guo et al., ECCV 2024)
- SR3 (Saharia et al., IEEE TPAMI 2022), DDNM (Wang et al., ICLR 2023)
- DASR (Liang et al., CVPR 2021), IKC (Gu et al., CVPR 2019)
- DPSR (Zhang et al., CVPR 2019)
