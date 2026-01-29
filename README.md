# EigenPSF-DPS

**Non-blind Spatially Varying Deconvolution via Diffusion Posterior Sampling**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade implementation of EigenPSF-DPS, a novel algorithm for solving the inverse problem of **Spatially Varying (SV) Blur** using pre-trained diffusion models as priors.

## Overview

Traditional deconvolution methods assume spatially invariant blur, which fails for real-world scenarios like camera shake, lens aberrations, or depth-dependent defocus. EigenPSF-DPS addresses this by:

1. **EigenPSF Decomposition**: Efficiently representing spatially varying blur using PCA-based basis kernels
2. **Diffusion Posterior Sampling**: Leveraging pre-trained diffusion models as powerful image priors
3. **Measurement Consistency**: Guiding the diffusion process with physics-based likelihood gradients

## Mathematical Foundation

### Forward Model
```
y = A(x) + n
```
where `A` is a spatially varying blur operator approximated via EigenPSF decomposition:
```
A(x) ≈ Σₖ Cₖ ⊙ (x * Bₖ)
```
- `Bₖ`: Spatially invariant basis kernels (EigenPSFs)
- `Cₖ`: Spatially varying coefficient maps
- `*`: Convolution, `⊙`: Element-wise multiplication

### DPS Update Rule
At each diffusion step `t`:
1. **Tweedie's Estimate**: `x̂₀ = (xₜ - √(1-ᾱₜ)·εθ(xₜ)) / √(ᾱₜ)`
2. **Likelihood Gradient**: `grad = ∇_{xₜ} ||y - A(x̂₀)||²`
3. **Update**: `x_{t-1} ← StandardStep(xₜ, εθ) - ζₜ · grad`

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA 11.8+ (recommended for GPU acceleration)
- Conda package manager

### Step 1: Create Conda Environment

```bash
# Create a new conda environment
conda create -n eigenpsf-dps python=3.10 -y

# Activate the environment
conda activate eigenpsf-dps
```

### Step 2: Install PyTorch with CUDA

For **CUDA 11.8**:
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

For **CUDA 12.1**:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

For **CPU only**:
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### Step 3: Install Remaining Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

Or install individually:
```bash
pip install diffusers transformers accelerate
pip install scipy scikit-image matplotlib
pip install pyyaml tqdm requests
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

## Quick Start

### Basic Usage

First, pre-download the model weights (recommended on HPC clusters):
```bash
python main.py --download_only
```

Then run with default settings (synthetic face image, motion blur):
```bash
python main.py
```

### Custom Configuration

1. **Modify `config.yaml`** for persistent changes
2. **Use command-line overrides** for quick experiments:

```bash
# Use defocus blur instead of motion blur
python main.py --blur_mode defocus

# Use random IID blur (drastically varying PSFs between pixels)
python main.py --blur_mode random_iid --n_eigen_psfs 15

# Use correlated Gaussian blur (realistic optical aberrations)
python main.py --blur_mode correlated --n_eigen_psfs 10

# Adjust DPS step size (higher = stronger guidance)
python main.py --step_size 1.0

# Use more EigenPSF components for better approximation
python main.py --n_eigen_psfs 10

# Reduce inference steps for faster (but lower quality) results
python main.py --num_inference_steps 500

# Use your own image
python main.py --image /path/to/your/image.jpg

# Run on CPU
python main.py --device cpu
```

### Full Example

```bash
# Create conda environment and install
conda create -n eigenpsf-dps python=3.10 -y
conda activate eigenpsf-dps
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Run deconvolution with smooth blur
python main.py --blur_mode defocus --step_size 0.5 --n_eigen_psfs 5

# Run deconvolution with random IID blur (stress test)
python main.py --blur_mode random_iid --step_size 0.5 --n_eigen_psfs 20

# Results will be saved to ./results/run_YYYYMMDD_HHMMSS/
```

## Configuration

### config.yaml Structure

```yaml
# Data settings
data:
  img_size: 256              # Image resolution
  image_source: "sample"     # "sample", URL, or file path

# Diffusion model
model:
  model_id: "google/ddpm-celebahq-256"
  device: "cuda"
  dtype: "float32"

# Physics / Blur simulation
physics:
  blur_mode: "motion"        # "motion", "defocus", "mixed", "random_iid", or "correlated"
  kernel_size: 21            # PSF kernel size
  grid_size: 8               # PSF sampling grid (8 for smooth, 64 for random_iid)
  n_eigen_psfs: 5            # Number of PCA components (5 for smooth, 10+ for correlated/random_iid)
  sigma_noise: 0.01          # Measurement noise level
  random_iid:
    grid_size: 64            # Dense grid for near-pixel PSF variation
  correlated:
    sigma_scale: 1.0         # Gaussian width scale (larger = wider PSFs)
    mu_scale: 0.0            # Center offset scale (0 = centered)
    correlation_length: 7    # Spatial smoothness (larger = smoother)

# DPS parameters
dps:
  step_size: 0.5             # Gradient guidance scale (ζ)
  num_inference_steps: 1000  # Diffusion steps
  gradient_clip: 0.0         # Gradient clipping (0 = disabled)

# Output
output:
  save_dir: "./results"
  save_intermediate: true
  intermediate_freq: 100
```

### Blur Modes

| Mode | Description | PSF Variation | Recommended `n_eigen_psfs` |
|------|-------------|---------------|---------------------------|
| `motion` | Smoothly varying motion blur | Gradual (direction/length change slowly) | 5 |
| `defocus` | Smoothly varying defocus | Gradual (radius increases from center) | 5 |
| `mixed` | Alternating motion and defocus | Moderate | 5-10 |
| `correlated` | Spatially correlated asymmetric Gaussians | Smooth random (realistic aberrations) | 10-15 |
| `random_iid` | Near-IID random PSFs | Drastic (completely different neighbors) | 15-25 |

The `correlated` mode generates realistic optical aberration patterns using spatially correlated random fields:
- Asymmetric Gaussians with varying orientation, scale, and offset
- PSF parameters vary smoothly across the image (controlled by `correlation_length`)
- More realistic than deterministic modes, more tractable than `random_iid`

The `random_iid` mode stress-tests the EigenPSF decomposition by generating independently sampled PSFs at each grid point:
- Random motion blur (any angle, random length)
- Random defocus (random radius)
- Random asymmetric Gaussian (random orientation, aspect ratio, center offset)

### Key Parameters

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `step_size` | DPS guidance strength | 0.1 - 2.0 |
| `n_eigen_psfs` | Number of basis kernels | 5 (smooth), 10-15 (correlated), 15-25 (random_iid) |
| `sigma_noise` | Noise standard deviation | 0.001 - 0.05 |
| `num_inference_steps` | Diffusion sampling steps | 100 - 1000 |

## Output Structure

```
results/run_YYYYMMDD_HHMMSS/
├── config.yaml              # Saved configuration
├── metrics.txt              # PSNR, SSIM values
├── images/
│   ├── clean.png            # Ground truth
│   ├── blurred.png          # Measurement (y)
│   ├── restored.png         # DPS reconstruction
│   ├── comparison.png       # Side-by-side comparison
│   ├── eigenpsfs.png        # Basis kernels visualization
│   └── progress.png         # Sampling progression
└── intermediates/
    ├── step_0000.png
    ├── step_0100.png
    └── ...
```

## Project Structure

```
SVDPS/
├── main.py              # CLI entry point
├── physics.py           # EigenPSF decomposition & blur operators
├── dps.py               # Diffusion Posterior Sampling pipeline
├── utils.py             # Visualization, I/O, metrics
├── config.yaml          # Default configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Module Overview

### physics.py
- `SVBlurSimulator`: Generates spatially varying blur kernel fields
- `compute_eigen_decomposition()`: PCA decomposition of kernel field
- `SVBlurOperator`: Differentiable forward model using EigenPSFs
- `TrueSVBlurOperator`: Exact SV convolution for ground truth

### dps.py
- `EigenPSFDPSPipeline`: Core DPS algorithm with manual loop unrolling
- `tweedie_estimate()`: Clean image estimation from noisy sample
- `compute_likelihood_gradient()`: Autograd-based gradient computation

### utils.py
- Configuration loading/saving
- Image I/O and preprocessing
- Visualization functions
- Quality metrics (PSNR, SSIM)

## Advanced Usage

### Using Custom Images

```python
from utils import load_sample_image
import torch

# Load from file
img = load_sample_image("/path/to/image.jpg", size=256, device=torch.device("cuda"))

# Load from URL
img = load_sample_image("https://example.com/image.jpg", size=256, device=torch.device("cuda"))
```

### Programmatic API

```python
import torch
from physics import create_sv_blur_system
from dps import create_dps_pipeline

# Setup
device = torch.device("cuda")
physics_config = {
    "kernel_size": 21,
    "grid_size": 8,
    "blur_mode": "motion",
    "n_eigen_psfs": 5,
    "motion": {"min_length": 5, "max_length": 25, "angle_variation": 180},
}

# Create blur system
eigenpsf_op, true_op, decomposition = create_sv_blur_system(256, 256, physics_config, device)

# Create DPS pipeline
pipeline = create_dps_pipeline(eigenpsf_op, device=device)

# Run restoration
restored, intermediates = pipeline(
    y=measurement,
    num_inference_steps=1000,
    step_size=100.0,
)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `img_size` in config (e.g., 128 instead of 256)
- Use `--device cpu` for CPU-only execution
- Enable gradient checkpointing: `dps.gradient_checkpointing: true`

### Model Download Issues
The model download (~455 MB) can fail on clusters with unreliable network connections.
The built-in retry logic (exponential backoff, resume support) handles transient failures.

**Pre-download the model** (recommended on clusters):
```bash
# Download model with retries (no GPU needed)
python main.py --download_only

# Or specify a custom cache directory
python main.py --download_only --model_cache ./model_cache

# Increase retries for very unstable connections
python main.py --download_only --max_retries 10
```

Once cached, subsequent runs will load from disk:
```bash
python main.py --model_cache ./model_cache --blur_mode defocus
```

Other options:
- Set HuggingFace cache globally: `export HF_HOME=/path/to/cache`
- Try alternative model: `--model_id "google/ddpm-cifar10-32"`

### Poor Reconstruction Quality
- Adjust `step_size`: try values in 0.1-2.0 range (lower = smoother, higher = sharper but more artifacts)
- Increase `n_eigen_psfs` for better blur approximation (especially for `random_iid` mode)
- Ensure `sigma_noise` matches actual noise level
- Use full 1000 inference steps
- Check `explained_variance_ratio` in logs — if below 90%, increase `n_eigen_psfs`
- For `random_iid` mode, use at least 15-20 EigenPSF components

## Citation

If you use this code in your research, please cite:

```bibtex
@software{eigenpsf_dps,
  title={EigenPSF-DPS: Non-blind Spatially Varying Deconvolution via Diffusion Posterior Sampling},
  year={2024},
  url={https://github.com/yourusername/eigenpsf-dps}
}
```

## References

- [Diffusion Posterior Sampling (DPS)](https://arxiv.org/abs/2209.14687) - Chung et al., 2022
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456) - Song et al., 2020

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace Diffusers team for the excellent diffusion model library
- The DPS paper authors for the foundational algorithm
