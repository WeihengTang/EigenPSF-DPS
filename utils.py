"""
EigenPSF-DPS: Utility Functions
===============================

This module provides utility functions for:
- Configuration loading and management
- Image I/O and preprocessing
- Visualization and plotting
- Metrics computation

Author: EigenPSF-DPS Research Team
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import torch
from torch import Tensor
import yaml

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Management
# =============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from: {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to: {save_path}")


def setup_logging(level: str = "INFO", log_file: Optional[Union[str, Path]] = None) -> None:
    """Setup logging configuration with console and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file. If provided, all output is
                  also written to this file.
    """
    log_level = getattr(logging, level.upper())
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG to file
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_path}")


def get_device(device_str: str = "cuda") -> torch.device:
    """Get torch device, falling back to CPU if CUDA unavailable.

    Args:
        device_str: Requested device ("cuda" or "cpu")

    Returns:
        Torch device
    """
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if device_str == "cuda":
            logger.warning("CUDA requested but not available, falling back to CPU")
        else:
            logger.info("Using CPU device")
    return device


def get_dtype(dtype_str: str = "float32") -> torch.dtype:
    """Convert dtype string to torch dtype.

    Args:
        dtype_str: Data type string ("float32" or "float16")

    Returns:
        Torch dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to: {seed}")


# =============================================================================
# Image I/O and Processing
# =============================================================================

def load_sample_image(
    source: str = "sample",
    size: int = 256,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Load or generate a sample image for testing.

    Args:
        source: Image source - "sample" (uses skimage), URL, or file path
        size: Target image size
        device: Torch device
        dtype: Torch dtype

    Returns:
        Image tensor of shape (1, 3, H, W) normalized to [-1, 1]
    """
    if source == "sample":
        img = _load_skimage_sample(size, device, dtype)
    elif source.startswith(("http://", "https://")):
        img = _load_image_from_url(source, size, device, dtype)
    else:
        img = _load_image_from_file(source, size, device, dtype)

    return img


def _load_skimage_sample(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Load a real sample image from scikit-image.

    Tries astronaut() first (512x512 RGB photo with rich high-frequency detail),
    falls back to a detailed synthetic image if skimage is unavailable.
    """
    try:
        from skimage import data
        from PIL import Image

        # astronaut() is a 512x512 real photo with texture, edges, and fine detail
        img_np = data.astronaut().astype(np.float32) / 255.0

        # Resize to target size using PIL for quality
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil = img_pil.resize((size, size), Image.Resampling.LANCZOS)
        img_np = np.array(img_pil).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]

        logger.info("Loaded skimage.data.astronaut() as test image")
        return img_tensor.unsqueeze(0).to(device=device, dtype=dtype)

    except ImportError:
        logger.warning("scikit-image not available, using synthetic test image")
        return _generate_synthetic_image(size, device, dtype)


def _generate_synthetic_image(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Generate a synthetic image with rich high-frequency content.

    Creates a test pattern with sharp edges, fine textures, and varying
    spatial frequencies that blur will visibly degrade.
    """
    H, W = size, size

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )

    # Zone plate pattern (chirp) - has all spatial frequencies
    r = torch.sqrt(x**2 + y**2)
    zone_plate = 0.5 * torch.cos(40.0 * r**2)

    # Sharp-edged geometric shapes
    # Checkerboard pattern (high frequency)
    checker = torch.sign(torch.sin(16 * np.pi * x) * torch.sin(16 * np.pi * y))

    # Concentric sharp rings
    rings = torch.sign(torch.sin(20 * np.pi * r))

    # Star pattern with sharp edges
    theta = torch.atan2(y, x)
    star = torch.sign(torch.sin(8 * theta)) * (r < 0.7).float()

    # Text-like fine horizontal lines
    fine_lines = torch.sign(torch.sin(50 * np.pi * y)) * (torch.abs(x) < 0.3).float()

    # Compose: different patterns in different quadrants
    # Top-left: zone plate, Top-right: checkerboard
    # Bottom-left: star, Bottom-right: rings + lines
    tl = (x < 0).float() * (y < 0).float()
    tr = (x >= 0).float() * (y < 0).float()
    bl = (x < 0).float() * (y >= 0).float()
    br = (x >= 0).float() * (y >= 0).float()

    pattern = (
        tl * zone_plate +
        tr * checker * 0.5 +
        bl * star * 0.5 +
        br * (0.3 * rings + 0.3 * fine_lines)
    )

    # Add some smooth gradient variation for color
    red = 0.5 + pattern * 0.4 + 0.1 * x
    green = 0.5 + pattern * 0.35 - 0.05 * y
    blue = 0.5 + pattern * 0.3 + 0.05 * (x + y)

    img = torch.stack([red, green, blue], dim=0)
    img = torch.clamp(img, 0, 1)
    img = img * 2 - 1  # Normalize to [-1, 1]

    return img.unsqueeze(0)


def _load_image_from_url(
    url: str,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Load image from URL."""
    try:
        from PIL import Image
        import requests
        from io import BytesIO

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        # Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]

        return img_tensor.unsqueeze(0).to(device=device, dtype=dtype)

    except Exception as e:
        logger.warning(f"Failed to load image from URL: {e}. Using synthetic image.")
        return _generate_synthetic_image(size, device, dtype)


def _load_image_from_file(
    path: str,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Load image from file."""
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        img_tensor = img_tensor * 2 - 1

        return img_tensor.unsqueeze(0).to(device=device, dtype=dtype)

    except Exception as e:
        logger.warning(f"Failed to load image from file: {e}. Using synthetic image.")
        return _generate_synthetic_image(size, device, dtype)


def tensor_to_image(tensor: Tensor) -> np.ndarray:
    """Convert tensor to numpy image for visualization.

    Args:
        tensor: Image tensor of shape (B, C, H, W) or (C, H, W), range [-1, 1]

    Returns:
        NumPy array of shape (H, W, 3), range [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch element

    # Move to CPU and convert
    img = tensor.detach().cpu().float()

    # Denormalize from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)

    # Convert to HWC format
    img = img.permute(1, 2, 0).numpy()

    return img


def save_image(tensor: Tensor, path: Union[str, Path], format: str = "png") -> None:
    """Save tensor as image file.

    Args:
        tensor: Image tensor
        path: Output path
        format: Image format (png, jpg)
    """
    try:
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        img_np = tensor_to_image(tensor)
        img_np = (img_np * 255).astype(np.uint8)

        img = Image.fromarray(img_np)
        img.save(path)

        logger.debug(f"Image saved to: {path}")

    except Exception as e:
        logger.error(f"Failed to save image: {e}")


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(
    clean: Tensor,
    blurred: Tensor,
    restored: Tensor,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "EigenPSF-DPS Deconvolution Results",
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """Create visualization of deconvolution results.

    Args:
        clean: Clean image tensor
        blurred: Blurred (measured) image tensor
        restored: Restored image tensor
        save_path: Optional path to save figure
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    images = [
        (clean, "Ground Truth"),
        (blurred, "Blurred + Noise (Measurement)"),
        (restored, "Restored (EigenPSF-DPS)"),
    ]

    for ax, (img, subtitle) in zip(axes, images):
        ax.imshow(tensor_to_image(img))
        ax.set_title(subtitle, fontsize=12)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Visualization saved to: {save_path}")

    return fig


def visualize_eigenpsfs(
    basis_kernels: Tensor,
    coefficient_maps: Tensor,
    save_path: Optional[Union[str, Path]] = None,
    n_show: int = 5,
) -> plt.Figure:
    """Visualize EigenPSF basis kernels and coefficient maps.

    Args:
        basis_kernels: Shape (K, 1, kH, kW)
        coefficient_maps: Shape (K, H, W)
        save_path: Optional path to save figure
        n_show: Number of components to show

    Returns:
        Matplotlib figure
    """
    K = min(n_show, basis_kernels.shape[0])

    fig = plt.figure(figsize=(4 * K, 8))
    gs = gridspec.GridSpec(2, K, figure=fig)

    for k in range(K):
        # Basis kernel
        ax_kernel = fig.add_subplot(gs[0, k])
        kernel = basis_kernels[k, 0].detach().cpu().numpy()
        im = ax_kernel.imshow(kernel, cmap="RdBu_r")
        ax_kernel.set_title(f"EigenPSF {k+1}", fontsize=10)
        ax_kernel.axis("off")

        # Add colorbar
        divider = make_axes_locatable(ax_kernel)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Coefficient map
        ax_coeff = fig.add_subplot(gs[1, k])
        coeff = coefficient_maps[k].detach().cpu().numpy()
        im = ax_coeff.imshow(coeff, cmap="viridis")
        ax_coeff.set_title(f"Coefficient Map {k+1}", fontsize=10)
        ax_coeff.axis("off")

        divider = make_axes_locatable(ax_coeff)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    fig.suptitle("EigenPSF Decomposition", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"EigenPSF visualization saved to: {save_path}")

    return fig


def visualize_intermediate_results(
    intermediates: List[Tensor],
    steps: List[int],
    save_path: Optional[Union[str, Path]] = None,
    n_show: int = 8,
) -> plt.Figure:
    """Visualize intermediate reconstruction results during DPS sampling.

    Args:
        intermediates: List of intermediate image tensors
        steps: Corresponding step numbers
        save_path: Optional path to save figure
        n_show: Number of intermediates to show

    Returns:
        Matplotlib figure
    """
    # Subsample if too many intermediates
    n_total = len(intermediates)
    if n_total > n_show:
        indices = np.linspace(0, n_total - 1, n_show, dtype=int)
        intermediates = [intermediates[i] for i in indices]
        steps = [steps[i] for i in indices]

    n_cols = min(4, len(intermediates))
    n_rows = (len(intermediates) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, (img, step) in enumerate(zip(intermediates, steps)):
        if i < len(axes):
            axes[i].imshow(tensor_to_image(img))
            axes[i].set_title(f"Step {step}", fontsize=10)
            axes[i].axis("off")

    # Hide unused axes
    for i in range(len(intermediates), len(axes)):
        axes[i].axis("off")

    fig.suptitle("DPS Sampling Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Intermediate results saved to: {save_path}")

    return fig


def visualize_kernel_field(
    kernel_grid: Tensor,
    save_path: Optional[Union[str, Path]] = None,
    n_show: int = 4,
) -> plt.Figure:
    """Visualize spatially varying kernel field.

    Args:
        kernel_grid: Shape (gH, gW, kH, kW)
        save_path: Optional path to save figure
        n_show: Number of kernels per axis to show

    Returns:
        Matplotlib figure
    """
    gH, gW = kernel_grid.shape[:2]

    # Sample kernels at regular intervals
    h_indices = np.linspace(0, gH - 1, n_show, dtype=int)
    w_indices = np.linspace(0, gW - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, n_show, figsize=(3 * n_show, 3 * n_show))

    for i, hi in enumerate(h_indices):
        for j, wi in enumerate(w_indices):
            kernel = kernel_grid[hi, wi].detach().cpu().numpy()
            axes[i, j].imshow(kernel, cmap="hot")
            axes[i, j].set_title(f"({hi}, {wi})", fontsize=8)
            axes[i, j].axis("off")

    fig.suptitle("Spatially Varying Kernel Field", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Kernel field visualization saved to: {save_path}")

    return fig


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(img1: Tensor, img2: Tensor, max_val: float = 2.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum value range (2.0 for [-1, 1] normalized images)

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr


def compute_ssim(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)
        window_size: Size of sliding window
        size_average: Whether to average over spatial dimensions

    Returns:
        SSIM value
    """
    # Convert to [0, 1] range for SSIM computation
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()

    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device, img1.dtype)

    # Expand window for all channels
    C = img1.shape[1]
    window = window.expand(C, 1, window_size, window_size)

    padding = window_size // 2

    # Compute means
    mu1 = torch.nn.functional.conv2d(img1, window, padding=padding, groups=C)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=padding, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = torch.nn.functional.conv2d(img1 ** 2, window, padding=padding, groups=C) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 ** 2, window, padding=padding, groups=C) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=padding, groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3]).item()


def compute_metrics(
    clean: Tensor,
    restored: Tensor,
    blurred: Optional[Tensor] = None,
) -> Dict[str, float]:
    """Compute all image quality metrics.

    Args:
        clean: Ground truth clean image
        restored: Restored image
        blurred: Optional blurred image for comparison

    Returns:
        Dictionary of metric values
    """
    metrics = {
        "psnr_restored": compute_psnr(clean, restored),
        "ssim_restored": compute_ssim(clean, restored),
    }

    if blurred is not None:
        metrics["psnr_blurred"] = compute_psnr(clean, blurred)
        metrics["ssim_blurred"] = compute_ssim(clean, blurred)
        metrics["psnr_improvement"] = metrics["psnr_restored"] - metrics["psnr_blurred"]
        metrics["ssim_improvement"] = metrics["ssim_restored"] - metrics["ssim_blurred"]

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in formatted table.

    Args:
        metrics: Dictionary of metric values
    """
    print("\n" + "=" * 50)
    print("IMAGE QUALITY METRICS")
    print("=" * 50)

    for name, value in metrics.items():
        if "psnr" in name.lower():
            print(f"  {name:25s}: {value:8.2f} dB")
        else:
            print(f"  {name:25s}: {value:8.4f}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Quick test
    setup_logging("INFO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test image generation
    img = load_sample_image("sample", size=256, device=device)
    print(f"Generated sample image: {img.shape}")

    # Test metrics
    noisy_img = img + 0.1 * torch.randn_like(img)
    psnr = compute_psnr(img, noisy_img)
    ssim = compute_ssim(img, noisy_img)
    print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    # Test visualization
    fig = visualize_results(img, noisy_img, img, title="Test Visualization")
    plt.close(fig)
    print("Visualization test passed")
