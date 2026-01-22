"""
EigenPSF-DPS: Physics Module
============================

This module implements the core physics of spatially varying (SV) blur using
the EigenPSF decomposition framework.

Mathematical Foundation:
    Forward Model: y = A(x) + n

    Where A is approximated via EigenPSF decomposition:
        A(x) ≈ Σ_{k=1}^K C_k ⊙ (x * B_k)

    - B_k: Spatially invariant basis kernels (EigenPSFs)
    - C_k: Spatially varying coefficient maps
    - *: Convolution
    - ⊙: Element-wise multiplication

Author: EigenPSF-DPS Research Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EigenPSFDecomposition:
    """Container for EigenPSF decomposition results.

    Attributes:
        basis_kernels: Shape (K, 1, kH, kW) - The K principal basis kernels
        coefficient_maps: Shape (K, H, W) - Spatially varying weight maps
        singular_values: Shape (K,) - Singular values from PCA
        explained_variance_ratio: float - Fraction of variance explained by K components
    """
    basis_kernels: Tensor
    coefficient_maps: Tensor
    singular_values: Tensor
    explained_variance_ratio: float


class SVBlurSimulator:
    """Simulator for Spatially Varying (SV) blur kernels.

    Generates grids of spatially varying point spread functions (PSFs) for
    motion blur, defocus blur, or mixed blur patterns.

    Args:
        img_height: Image height in pixels
        img_width: Image width in pixels
        kernel_size: Size of each local PSF kernel (must be odd)
        grid_size: Number of PSF samples along each spatial dimension
        device: Torch device for computation
        dtype: Torch dtype for computation
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        kernel_size: int = 21,
        grid_size: int = 8,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.img_height = img_height
        self.img_width = img_width
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.device = device
        self.dtype = dtype

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    def simulate_sv_kernel_grid(
        self,
        mode: Literal["motion", "defocus", "mixed"] = "motion",
        motion_length_range: Tuple[float, float] = (5.0, 25.0),
        motion_angle_variation: float = 180.0,
        defocus_radius_range: Tuple[float, float] = (2.0, 12.0),
        seed: Optional[int] = None,
    ) -> Tensor:
        """Generate a grid of spatially varying blur kernels.

        Args:
            mode: Type of blur - "motion", "defocus", or "mixed"
            motion_length_range: (min, max) motion blur length in pixels
            motion_angle_variation: Angular variation in degrees
            defocus_radius_range: (min, max) defocus blur radius in pixels
            seed: Random seed for reproducibility

        Returns:
            Tensor of shape (grid_size, grid_size, kernel_size, kernel_size)
            containing the spatially varying kernels
        """
        if seed is not None:
            torch.manual_seed(seed)

        kernels = torch.zeros(
            self.grid_size, self.grid_size, self.kernel_size, self.kernel_size,
            device=self.device, dtype=self.dtype
        )

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Normalized position in [0, 1]
                y_pos = i / (self.grid_size - 1) if self.grid_size > 1 else 0.5
                x_pos = j / (self.grid_size - 1) if self.grid_size > 1 else 0.5

                if mode == "motion":
                    kernel = self._create_motion_kernel(
                        x_pos, y_pos,
                        motion_length_range,
                        motion_angle_variation
                    )
                elif mode == "defocus":
                    kernel = self._create_defocus_kernel(
                        x_pos, y_pos,
                        defocus_radius_range
                    )
                else:  # mixed
                    if (i + j) % 2 == 0:
                        kernel = self._create_motion_kernel(
                            x_pos, y_pos,
                            motion_length_range,
                            motion_angle_variation
                        )
                    else:
                        kernel = self._create_defocus_kernel(
                            x_pos, y_pos,
                            defocus_radius_range
                        )

                kernels[i, j] = kernel

        return kernels

    def _create_motion_kernel(
        self,
        x_pos: float,
        y_pos: float,
        length_range: Tuple[float, float],
        angle_variation: float,
    ) -> Tensor:
        """Create a motion blur kernel based on spatial position.

        The motion direction and intensity vary smoothly across the image.
        """
        # Vary motion length with position (stronger blur at edges)
        distance_from_center = math.sqrt((x_pos - 0.5)**2 + (y_pos - 0.5)**2)
        t = min(distance_from_center * 2, 1.0)  # Normalized distance [0, 1]
        length = length_range[0] + t * (length_range[1] - length_range[0])

        # Vary angle based on position (radial pattern from center)
        base_angle = math.atan2(y_pos - 0.5, x_pos - 0.5)
        angle_offset = (angle_variation / 180.0) * math.pi * (t - 0.5)
        angle = base_angle + angle_offset

        return self._motion_kernel(length, angle)

    def _motion_kernel(self, length: float, angle: float) -> Tensor:
        """Generate a single motion blur kernel.

        Args:
            length: Length of motion blur in pixels
            angle: Angle of motion in radians

        Returns:
            Normalized motion blur kernel
        """
        kernel = torch.zeros(
            self.kernel_size, self.kernel_size,
            device=self.device, dtype=self.dtype
        )

        center = self.kernel_size // 2

        # Calculate line endpoints
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2

        # Use anti-aliased line drawing
        num_samples = max(int(length * 2), 10)
        for t in torch.linspace(-0.5, 0.5, num_samples, device=self.device):
            x = center + t * 2 * dx
            y = center + t * 2 * dy

            # Bilinear interpolation for anti-aliasing
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1

            if 0 <= x0 < self.kernel_size and 0 <= y0 < self.kernel_size:
                wx = x - x0
                wy = y - y0

                kernel[y0, x0] += (1 - wx) * (1 - wy)
                if x1 < self.kernel_size:
                    kernel[y0, x1] += wx * (1 - wy)
                if y1 < self.kernel_size:
                    kernel[y1, x0] += (1 - wx) * wy
                if x1 < self.kernel_size and y1 < self.kernel_size:
                    kernel[y1, x1] += wx * wy

        # Normalize
        kernel = kernel / (kernel.sum() + 1e-8)

        return kernel

    def _create_defocus_kernel(
        self,
        x_pos: float,
        y_pos: float,
        radius_range: Tuple[float, float],
    ) -> Tensor:
        """Create a defocus blur kernel based on spatial position.

        Simulates depth-dependent defocus (larger blur at edges).
        """
        # Vary radius with position
        distance_from_center = math.sqrt((x_pos - 0.5)**2 + (y_pos - 0.5)**2)
        t = min(distance_from_center * 2, 1.0)
        radius = radius_range[0] + t * (radius_range[1] - radius_range[0])

        return self._disk_kernel(radius)

    def _disk_kernel(self, radius: float) -> Tensor:
        """Generate a disk (defocus) blur kernel.

        Args:
            radius: Radius of the disk in pixels

        Returns:
            Normalized disk kernel
        """
        center = self.kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(self.kernel_size, device=self.device, dtype=self.dtype),
            torch.arange(self.kernel_size, device=self.device, dtype=self.dtype),
            indexing="ij"
        )

        # Distance from center
        dist = torch.sqrt((x - center)**2 + (y - center)**2)

        # Soft disk with anti-aliasing
        kernel = torch.clamp(1.0 - (dist - radius) / 1.5, 0.0, 1.0)

        # Normalize
        kernel = kernel / (kernel.sum() + 1e-8)

        return kernel

    def interpolate_kernels_to_image(
        self,
        kernel_grid: Tensor,
    ) -> Tensor:
        """Interpolate kernel grid to full image resolution.

        Args:
            kernel_grid: Shape (grid_H, grid_W, kH, kW)

        Returns:
            Interpolated kernels of shape (H, W, kH, kW)
        """
        gH, gW, kH, kW = kernel_grid.shape

        # Reshape for interpolation: (kH*kW, 1, gH, gW)
        grid_flat = kernel_grid.permute(2, 3, 0, 1).reshape(kH * kW, 1, gH, gW)

        # Bilinear interpolation to image size
        interpolated = F.interpolate(
            grid_flat,
            size=(self.img_height, self.img_width),
            mode="bilinear",
            align_corners=True,
        )

        # Reshape back: (H, W, kH, kW)
        interpolated = interpolated.reshape(kH, kW, self.img_height, self.img_width)
        interpolated = interpolated.permute(2, 3, 0, 1)

        # Re-normalize each kernel
        interpolated = interpolated / (interpolated.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        return interpolated


def compute_eigen_decomposition(
    kernel_grid: Tensor,
    img_height: int,
    img_width: int,
    n_components: int = 5,
) -> EigenPSFDecomposition:
    """Compute EigenPSF decomposition of spatially varying kernels.

    Uses PCA to decompose a grid of spatially varying kernels into:
    - K basis kernels (EigenPSFs)
    - K spatially varying coefficient maps

    Args:
        kernel_grid: Shape (grid_H, grid_W, kH, kW) - Grid of SV kernels
        img_height: Target image height
        img_width: Target image width
        n_components: Number of PCA components (K)

    Returns:
        EigenPSFDecomposition containing basis kernels and coefficient maps
    """
    device = kernel_grid.device
    dtype = kernel_grid.dtype
    gH, gW, kH, kW = kernel_grid.shape

    # Flatten kernels: (gH * gW, kH * kW)
    kernels_flat = kernel_grid.reshape(gH * gW, kH * kW)

    # Center the data (subtract mean)
    mean_kernel = kernels_flat.mean(dim=0, keepdim=True)
    kernels_centered = kernels_flat - mean_kernel

    # PCA via SVD
    # kernels_centered = U @ S @ V^T
    # Columns of V are principal components (basis kernels)
    # U @ S gives the coefficients
    U, S, Vh = torch.linalg.svd(kernels_centered, full_matrices=False)

    # Extract top K components
    K = min(n_components, len(S))

    # Basis kernels: (K, kH, kW)
    # Note: Vh rows are the principal components
    basis_kernels = Vh[:K].reshape(K, kH, kW)

    # Add mean kernel as the first basis (for reconstruction accuracy)
    mean_basis = mean_kernel.reshape(1, kH, kW)

    # Coefficient maps at grid resolution: (K, gH, gW)
    # Coefficients = U @ diag(S)
    coefficients_grid = (U[:, :K] * S[:K].unsqueeze(0)).T.reshape(K, gH, gW)

    # Interpolate coefficient maps to full image resolution
    coefficients_interp = F.interpolate(
        coefficients_grid.unsqueeze(0),  # (1, K, gH, gW)
        size=(img_height, img_width),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)  # (K, H, W)

    # Create coefficient map for mean kernel (constant = 1)
    mean_coeff = torch.ones(1, img_height, img_width, device=device, dtype=dtype)

    # Combine: mean kernel + PCA components
    all_basis = torch.cat([mean_basis, basis_kernels], dim=0)  # (K+1, kH, kW)
    all_coeffs = torch.cat([mean_coeff, coefficients_interp], dim=0)  # (K+1, H, W)

    # Compute explained variance ratio
    total_variance = (S ** 2).sum()
    explained_variance = (S[:K] ** 2).sum()
    explained_ratio = (explained_variance / (total_variance + 1e-8)).item()

    # Reshape basis kernels for conv2d: (K+1, 1, kH, kW)
    all_basis = all_basis.unsqueeze(1)

    return EigenPSFDecomposition(
        basis_kernels=all_basis,
        coefficient_maps=all_coeffs,
        singular_values=S[:K],
        explained_variance_ratio=explained_ratio,
    )


class SVBlurOperator(nn.Module):
    """Differentiable Spatially Varying Blur Operator using EigenPSF decomposition.

    Implements the forward model:
        A(x) = Σ_{k=1}^K C_k ⊙ (x * B_k)

    where:
        - B_k: Basis kernels (EigenPSFs) of shape (K, 1, kH, kW)
        - C_k: Coefficient maps of shape (K, H, W)
        - *: Convolution
        - ⊙: Element-wise multiplication

    Args:
        decomposition: EigenPSFDecomposition containing basis kernels and coefficients
        padding_mode: Padding mode for convolution ("zeros", "reflect", "replicate")
    """

    def __init__(
        self,
        decomposition: EigenPSFDecomposition,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        # Register as buffers (not trainable parameters)
        self.register_buffer("basis_kernels", decomposition.basis_kernels)
        self.register_buffer("coefficient_maps", decomposition.coefficient_maps)

        self.n_components = decomposition.basis_kernels.shape[0]
        self.kernel_size = decomposition.basis_kernels.shape[-1]
        self.padding = self.kernel_size // 2
        self.padding_mode = padding_mode

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatially varying blur to input image.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Blurred tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Apply reflection padding
        if self.padding > 0:
            x_padded = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
        else:
            x_padded = x

        # Initialize output
        output = torch.zeros_like(x)

        # Sum over all EigenPSF components
        for k in range(self.n_components):
            # Get k-th basis kernel: (1, 1, kH, kW)
            kernel_k = self.basis_kernels[k:k+1]

            # Get k-th coefficient map: (1, H, W) -> (1, 1, H, W)
            coeff_k = self.coefficient_maps[k:k+1].unsqueeze(0)

            # Convolve each channel with the basis kernel
            # Process all channels together using groups
            kernel_expanded = kernel_k.expand(C, 1, -1, -1)  # (C, 1, kH, kW)

            conv_result = F.conv2d(
                x_padded,
                kernel_expanded,
                padding=0,
                groups=C,
            )  # (B, C, H, W)

            # Apply spatially varying coefficients
            # coeff_k broadcasts over batch and channel dimensions
            output = output + coeff_k * conv_result

        return output

    def adjoint(self, y: Tensor) -> Tensor:
        """Apply adjoint (transpose) of the blur operator.

        The adjoint of the SV blur is needed for some optimization algorithms.
        For our decomposition: A^T(y) = Σ_k (C_k ⊙ y) * B_k^T
        where B_k^T is the flipped kernel.

        Args:
            y: Input tensor of shape (B, C, H, W)

        Returns:
            Result of adjoint operation, shape (B, C, H, W)
        """
        B, C, H, W = y.shape

        # Initialize output
        output = torch.zeros_like(y)

        for k in range(self.n_components):
            # Get coefficient map and apply to input
            coeff_k = self.coefficient_maps[k:k+1].unsqueeze(0)  # (1, 1, H, W)
            weighted = coeff_k * y  # (B, C, H, W)

            # Apply padding
            if self.padding > 0:
                weighted_padded = F.pad(weighted, [self.padding] * 4, mode=self.padding_mode)
            else:
                weighted_padded = weighted

            # Flip kernel for adjoint (transpose convolution)
            kernel_k = self.basis_kernels[k:k+1]
            kernel_flipped = torch.flip(kernel_k, dims=[-2, -1])
            kernel_expanded = kernel_flipped.expand(C, 1, -1, -1)

            # Convolve with flipped kernel
            conv_result = F.conv2d(
                weighted_padded,
                kernel_expanded,
                padding=0,
                groups=C,
            )

            output = output + conv_result

        return output


class TrueSVBlurOperator(nn.Module):
    """True Spatially Varying Blur Operator (for generating ground truth).

    This implements the exact SV convolution without EigenPSF approximation.
    Used for generating measurements y = A_true(x) + noise.

    Note: This is computationally expensive and not differentiable for
    large images. Use SVBlurOperator for the inverse problem.

    Args:
        kernel_field: Full resolution kernel field of shape (H, W, kH, kW)
    """

    def __init__(self, kernel_field: Tensor) -> None:
        super().__init__()
        self.register_buffer("kernel_field", kernel_field)
        self.kernel_size = kernel_field.shape[-1]
        self.padding = self.kernel_size // 2

    def forward(self, x: Tensor) -> Tensor:
        """Apply true spatially varying blur.

        This uses unfold for efficient SV convolution.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Blurred tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        kH, kW = self.kernel_size, self.kernel_size

        # Pad input
        x_padded = F.pad(x, [self.padding] * 4, mode="reflect")

        # Unfold to get all local patches: (B, C, kH*kW, H, W)
        patches = F.unfold(x_padded, kernel_size=(kH, kW))  # (B, C*kH*kW, H*W)
        patches = patches.reshape(B, C, kH * kW, H, W)

        # Flatten kernel field: (H, W, kH*kW)
        kernels_flat = self.kernel_field.reshape(H, W, kH * kW)
        kernels_flat = kernels_flat.permute(2, 0, 1)  # (kH*kW, H, W)

        # Apply spatially varying convolution via einsum
        # patches: (B, C, kH*kW, H, W)
        # kernels: (kH*kW, H, W)
        # output: (B, C, H, W)
        output = torch.einsum("bckhw,khw->bchw", patches, kernels_flat)

        return output


def create_sv_blur_system(
    img_height: int,
    img_width: int,
    config: dict,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[SVBlurOperator, TrueSVBlurOperator, EigenPSFDecomposition]:
    """Create complete SV blur system with EigenPSF decomposition.

    Args:
        img_height: Image height
        img_width: Image width
        config: Physics configuration dictionary
        device: Torch device
        dtype: Torch dtype

    Returns:
        Tuple of (eigenpsf_operator, true_operator, decomposition)
    """
    # Create simulator
    simulator = SVBlurSimulator(
        img_height=img_height,
        img_width=img_width,
        kernel_size=config.get("kernel_size", 21),
        grid_size=config.get("grid_size", 8),
        device=device,
        dtype=dtype,
    )

    # Get blur mode parameters
    blur_mode = config.get("blur_mode", "motion")
    motion_cfg = config.get("motion", {})
    defocus_cfg = config.get("defocus", {})

    # Generate kernel grid
    kernel_grid = simulator.simulate_sv_kernel_grid(
        mode=blur_mode,
        motion_length_range=(
            motion_cfg.get("min_length", 5),
            motion_cfg.get("max_length", 25),
        ),
        motion_angle_variation=motion_cfg.get("angle_variation", 180),
        defocus_radius_range=(
            defocus_cfg.get("min_radius", 2),
            defocus_cfg.get("max_radius", 12),
        ),
    )

    # Interpolate to full resolution for true operator
    kernel_field = simulator.interpolate_kernels_to_image(kernel_grid)

    # Compute EigenPSF decomposition
    n_eigen = config.get("n_eigen_psfs", 5)
    decomposition = compute_eigen_decomposition(
        kernel_grid,
        img_height,
        img_width,
        n_components=n_eigen,
    )

    # Create operators
    eigenpsf_operator = SVBlurOperator(decomposition).to(device)
    true_operator = TrueSVBlurOperator(kernel_field).to(device)

    return eigenpsf_operator, true_operator, decomposition


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    H, W = 256, 256
    config = {
        "kernel_size": 21,
        "grid_size": 8,
        "blur_mode": "motion",
        "n_eigen_psfs": 5,
        "motion": {"min_length": 5, "max_length": 25, "angle_variation": 180},
    }

    eigenpsf_op, true_op, decomp = create_sv_blur_system(H, W, config, device)

    print(f"EigenPSF decomposition:")
    print(f"  - Number of components: {decomp.basis_kernels.shape[0]}")
    print(f"  - Kernel size: {decomp.basis_kernels.shape[-1]}")
    print(f"  - Explained variance: {decomp.explained_variance_ratio:.2%}")

    # Test forward pass
    x = torch.randn(1, 3, H, W, device=device)

    y_eigen = eigenpsf_op(x)
    y_true = true_op(x)

    error = (y_eigen - y_true).abs().mean()
    print(f"  - Approximation error (MAE): {error:.6f}")

    # Test gradient flow
    x.requires_grad_(True)
    y = eigenpsf_op(x)
    loss = y.sum()
    loss.backward()
    print(f"  - Gradient flow: OK (grad norm: {x.grad.norm():.4f})")
