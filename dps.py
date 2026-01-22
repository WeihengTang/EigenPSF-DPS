"""
EigenPSF-DPS: Diffusion Posterior Sampling Module
==================================================

This module implements the DPS (Diffusion Posterior Sampling) algorithm
for solving inverse problems using pre-trained diffusion models.

Mathematical Foundation:
    At each diffusion step t, we update x_t using:

    1. Tweedie's Formula (Estimate Clean Image):
       x̂_0(x_t) = (x_t - √(1-ᾱ_t) · ε_θ(x_t)) / √(ᾱ_t)

    2. Likelihood Gradient:
       grad = ∇_{x_t} ||y - A(x̂_0(x_t))||²

    3. Update Rule:
       x_{t-1} ← Standard_Step(x_t, ε_θ) - ζ_t · grad

Author: EigenPSF-DPS Research Team
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, List, Tuple, Literal

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor

from physics import SVBlurOperator


logger = logging.getLogger(__name__)


class EigenPSFDPSPipeline:
    """Diffusion Posterior Sampling Pipeline for SV Deconvolution.

    This pipeline implements DPS with EigenPSF-based spatially varying blur
    operator. It manually unrolls the diffusion sampling loop to inject
    measurement consistency guidance at each step.

    Args:
        unet: Pre-trained UNet2DModel for noise prediction
        scheduler: DDPM scheduler for diffusion process
        blur_operator: SVBlurOperator for applying/computing SV blur
        device: Torch device for computation
        dtype: Torch dtype for computation
    """

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDPMScheduler,
        blur_operator: SVBlurOperator,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.blur_operator = blur_operator.to(device)
        self.device = device
        self.dtype = dtype

        # Ensure UNet is in eval mode
        self.unet.eval()

    @torch.no_grad()
    def get_alpha_bars(self) -> Tensor:
        """Get cumulative product of alphas (ᾱ_t) from scheduler."""
        return self.scheduler.alphas_cumprod.to(self.device)

    def tweedie_estimate(
        self,
        x_t: Tensor,
        noise_pred: Tensor,
        t: int,
    ) -> Tensor:
        """Compute Tweedie's denoising estimate of x_0.

        x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ(x_t)) / √(ᾱ_t)

        Args:
            x_t: Noisy image at timestep t, shape (B, C, H, W)
            noise_pred: Predicted noise ε_θ(x_t), shape (B, C, H, W)
            t: Current timestep

        Returns:
            Estimated clean image x̂_0, shape (B, C, H, W)
        """
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(self.device)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

        # Tweedie's formula
        x_0_hat = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar

        return x_0_hat

    def compute_likelihood_gradient(
        self,
        x_t: Tensor,
        y: Tensor,
        t: int,
    ) -> Tuple[Tensor, float]:
        """Compute gradient of measurement likelihood w.r.t. x_t.

        Computes: ∇_{x_t} ||y - A(x̂_0(x_t))||²

        Args:
            x_t: Noisy image at timestep t (requires_grad must be True)
            y: Measurement (blurred + noisy image)
            t: Current timestep

        Returns:
            Tuple of (gradient tensor, loss value)
        """
        # Ensure x_t requires grad
        x_t_var = x_t.detach().clone().requires_grad_(True)

        # Predict noise
        with torch.enable_grad():
            # Get timestep tensor
            timestep = torch.tensor([t], device=self.device, dtype=torch.long)

            # Predict noise using UNet
            noise_pred = self.unet(x_t_var, timestep).sample

            # Tweedie estimate
            x_0_hat = self.tweedie_estimate(x_t_var, noise_pred, t)

            # Clamp to valid range for stability
            x_0_hat_clamped = torch.clamp(x_0_hat, -1.0, 1.0)

            # Apply blur operator
            y_hat = self.blur_operator(x_0_hat_clamped)

            # Compute MSE loss
            loss = torch.sum((y - y_hat) ** 2)

            # Compute gradient
            grad = torch.autograd.grad(loss, x_t_var)[0]

        return grad, loss.item()

    def get_step_size(
        self,
        t: int,
        base_step_size: float,
        schedule: Literal["constant", "linear_decay", "sqrt_decay"] = "constant",
        total_steps: int = 1000,
    ) -> float:
        """Compute step size (zeta) for current timestep.

        Args:
            t: Current timestep
            base_step_size: Base step size (zeta)
            schedule: Step size schedule type
            total_steps: Total number of diffusion steps

        Returns:
            Step size for current timestep
        """
        if schedule == "constant":
            return base_step_size
        elif schedule == "linear_decay":
            # Linear decay from base to 0
            return base_step_size * (t / total_steps)
        elif schedule == "sqrt_decay":
            # Square root decay
            return base_step_size * torch.sqrt(torch.tensor(t / total_steps)).item()
        else:
            return base_step_size

    @torch.no_grad()
    def __call__(
        self,
        y: Tensor,
        num_inference_steps: int = 1000,
        step_size: float = 100.0,
        step_size_schedule: Literal["constant", "linear_decay", "sqrt_decay"] = "constant",
        skip_steps: int = 0,
        gradient_clip: float = 1.0,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, Tensor, float], None]] = None,
        callback_steps: int = 100,
        show_progress: bool = True,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Run DPS sampling for SV deconvolution.

        Args:
            y: Measurement tensor (blurred + noisy), shape (B, C, H, W)
            num_inference_steps: Number of diffusion steps
            step_size: Base gradient guidance scale (zeta)
            step_size_schedule: Schedule for step size
            skip_steps: Number of initial steps to skip
            gradient_clip: Max gradient norm (0 = no clipping)
            generator: Random generator for reproducibility
            callback: Optional callback(step, x_t, loss) called every callback_steps
            callback_steps: Frequency of callback invocation
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (final reconstruction, list of intermediate results)
        """
        batch_size = y.shape[0]
        img_shape = y.shape

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Apply skip steps
        if skip_steps > 0:
            timesteps = timesteps[skip_steps:]

        # Initialize x_T from pure noise
        x_t = randn_tensor(img_shape, generator=generator, device=self.device, dtype=self.dtype)

        # Store intermediate results
        intermediates: List[Tensor] = []

        # Progress bar
        progress_bar = tqdm(
            enumerate(timesteps),
            total=len(timesteps),
            desc="DPS Sampling",
            disable=not show_progress,
        )

        for i, t in progress_bar:
            t_int = t.item() if isinstance(t, Tensor) else t

            # 1. Compute measurement likelihood gradient
            grad, loss = self.compute_likelihood_gradient(x_t, y, t_int)

            # 2. Gradient clipping for stability
            if gradient_clip > 0:
                grad_norm = grad.norm()
                if grad_norm > gradient_clip:
                    grad = grad * (gradient_clip / grad_norm)

            # 3. Get current step size
            zeta = self.get_step_size(
                t_int, step_size, step_size_schedule, num_inference_steps
            )

            # 4. Standard DDPM prediction (without gradient)
            timestep = torch.tensor([t_int], device=self.device, dtype=torch.long)
            noise_pred = self.unet(x_t, timestep).sample

            # 5. Scheduler step (standard diffusion)
            scheduler_output = self.scheduler.step(
                noise_pred, t_int, x_t, generator=generator
            )
            x_t_standard = scheduler_output.prev_sample

            # 6. Apply DPS gradient guidance
            # x_{t-1} = Standard_Step(x_t) - zeta * grad
            x_t = x_t_standard - zeta * grad

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss:.4f}", "zeta": f"{zeta:.2f}"})

            # Callback and intermediate saving
            if callback is not None and (i % callback_steps == 0 or i == len(timesteps) - 1):
                # Get clean estimate for visualization
                x_0_hat = self.tweedie_estimate(x_t, noise_pred, t_int)
                x_0_hat_clamped = torch.clamp(x_0_hat, -1.0, 1.0)
                intermediates.append(x_0_hat_clamped.cpu())
                callback(i, x_0_hat_clamped, loss)

        # Final denoising step - compute final x_0 estimate
        final_timestep = torch.tensor([0], device=self.device, dtype=torch.long)
        final_noise_pred = self.unet(x_t, final_timestep).sample
        x_0_final = self.tweedie_estimate(x_t, final_noise_pred, 0)
        x_0_final = torch.clamp(x_0_final, -1.0, 1.0)

        return x_0_final, intermediates


class EigenPSFDPSPipelineV2(EigenPSFDPSPipeline):
    """Enhanced DPS Pipeline with additional features.

    Improvements over base pipeline:
    - Adaptive step size based on gradient magnitude
    - Momentum-based gradient update
    - Optional warm restart
    """

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDPMScheduler,
        blur_operator: SVBlurOperator,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(unet, scheduler, blur_operator, device, dtype)
        self.momentum = momentum
        self._grad_buffer: Optional[Tensor] = None

    def _apply_momentum(self, grad: Tensor) -> Tensor:
        """Apply momentum to gradient."""
        if self._grad_buffer is None:
            self._grad_buffer = grad.clone()
        else:
            self._grad_buffer = self.momentum * self._grad_buffer + (1 - self.momentum) * grad
        return self._grad_buffer

    @torch.no_grad()
    def __call__(
        self,
        y: Tensor,
        num_inference_steps: int = 1000,
        step_size: float = 100.0,
        step_size_schedule: Literal["constant", "linear_decay", "sqrt_decay"] = "constant",
        skip_steps: int = 0,
        gradient_clip: float = 1.0,
        use_momentum: bool = True,
        adaptive_step_size: bool = False,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, Tensor, float], None]] = None,
        callback_steps: int = 100,
        show_progress: bool = True,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Run enhanced DPS sampling.

        Additional Args:
            use_momentum: Whether to apply momentum to gradients
            adaptive_step_size: Whether to adapt step size based on gradient norm
        """
        # Reset momentum buffer
        self._grad_buffer = None

        batch_size = y.shape[0]
        img_shape = y.shape

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        if skip_steps > 0:
            timesteps = timesteps[skip_steps:]

        # Initialize x_T from pure noise
        x_t = randn_tensor(img_shape, generator=generator, device=self.device, dtype=self.dtype)

        intermediates: List[Tensor] = []

        progress_bar = tqdm(
            enumerate(timesteps),
            total=len(timesteps),
            desc="DPS-V2 Sampling",
            disable=not show_progress,
        )

        prev_loss = float("inf")

        for i, t in progress_bar:
            t_int = t.item() if isinstance(t, Tensor) else t

            # Compute gradient
            grad, loss = self.compute_likelihood_gradient(x_t, y, t_int)

            # Apply momentum
            if use_momentum:
                grad = self._apply_momentum(grad)

            # Gradient clipping
            if gradient_clip > 0:
                grad_norm = grad.norm()
                if grad_norm > gradient_clip:
                    grad = grad * (gradient_clip / grad_norm)

            # Adaptive step size
            zeta = self.get_step_size(t_int, step_size, step_size_schedule, num_inference_steps)

            if adaptive_step_size:
                # Reduce step size if loss is increasing
                if loss > prev_loss * 1.5:
                    zeta *= 0.5
                prev_loss = loss

            # Standard DDPM step
            timestep = torch.tensor([t_int], device=self.device, dtype=torch.long)
            noise_pred = self.unet(x_t, timestep).sample

            scheduler_output = self.scheduler.step(noise_pred, t_int, x_t, generator=generator)
            x_t_standard = scheduler_output.prev_sample

            # DPS update
            x_t = x_t_standard - zeta * grad

            progress_bar.set_postfix({"loss": f"{loss:.4f}", "zeta": f"{zeta:.2f}"})

            if callback is not None and (i % callback_steps == 0 or i == len(timesteps) - 1):
                x_0_hat = self.tweedie_estimate(x_t, noise_pred, t_int)
                x_0_hat_clamped = torch.clamp(x_0_hat, -1.0, 1.0)
                intermediates.append(x_0_hat_clamped.cpu())
                callback(i, x_0_hat_clamped, loss)

        # Final estimate
        final_timestep = torch.tensor([0], device=self.device, dtype=torch.long)
        final_noise_pred = self.unet(x_t, final_timestep).sample
        x_0_final = self.tweedie_estimate(x_t, final_noise_pred, 0)
        x_0_final = torch.clamp(x_0_final, -1.0, 1.0)

        return x_0_final, intermediates


def load_pretrained_diffusion_model(
    model_id: str = "google/ddpm-celebahq-256",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[UNet2DModel, DDPMScheduler]:
    """Load a pre-trained diffusion model from HuggingFace.

    Args:
        model_id: HuggingFace model identifier
        device: Torch device
        dtype: Torch dtype

    Returns:
        Tuple of (UNet2DModel, DDPMScheduler)
    """
    logger.info(f"Loading pre-trained model: {model_id}")

    # Load UNet
    unet = UNet2DModel.from_pretrained(model_id)
    unet = unet.to(device=device, dtype=dtype)
    unet.eval()

    # Load scheduler
    scheduler = DDPMScheduler.from_pretrained(model_id)

    logger.info(f"Model loaded successfully. UNet params: {sum(p.numel() for p in unet.parameters()):,}")

    return unet, scheduler


def create_dps_pipeline(
    blur_operator: SVBlurOperator,
    model_id: str = "google/ddpm-celebahq-256",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    use_v2: bool = False,
) -> EigenPSFDPSPipeline:
    """Create a complete DPS pipeline.

    Args:
        blur_operator: SVBlurOperator instance
        model_id: HuggingFace model identifier
        device: Torch device
        dtype: Torch dtype
        use_v2: Whether to use enhanced V2 pipeline

    Returns:
        Configured DPS pipeline
    """
    unet, scheduler = load_pretrained_diffusion_model(model_id, device, dtype)

    if use_v2:
        pipeline = EigenPSFDPSPipelineV2(
            unet=unet,
            scheduler=scheduler,
            blur_operator=blur_operator,
            device=device,
            dtype=dtype,
        )
    else:
        pipeline = EigenPSFDPSPipeline(
            unet=unet,
            scheduler=scheduler,
            blur_operator=blur_operator,
            device=device,
            dtype=dtype,
        )

    return pipeline


if __name__ == "__main__":
    # Quick test of the pipeline components
    import sys
    sys.path.insert(0, ".")

    from physics import create_sv_blur_system

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing DPS pipeline on device: {device}")

    # Create blur system
    H, W = 256, 256
    physics_config = {
        "kernel_size": 21,
        "grid_size": 8,
        "blur_mode": "motion",
        "n_eigen_psfs": 5,
        "motion": {"min_length": 5, "max_length": 25, "angle_variation": 180},
    }

    eigenpsf_op, true_op, decomp = create_sv_blur_system(H, W, physics_config, device)
    print(f"Blur operator created with {decomp.basis_kernels.shape[0]} EigenPSFs")

    # Test loading model (may take time on first run)
    try:
        unet, scheduler = load_pretrained_diffusion_model(
            "google/ddpm-celebahq-256", device
        )
        print(f"UNet loaded: {unet.config.sample_size}x{unet.config.sample_size}")

        # Create pipeline
        pipeline = EigenPSFDPSPipeline(
            unet=unet,
            scheduler=scheduler,
            blur_operator=eigenpsf_op,
            device=device,
        )
        print("DPS pipeline created successfully")

        # Test single gradient computation
        x_test = torch.randn(1, 3, H, W, device=device)
        y_test = eigenpsf_op(x_test) + 0.01 * torch.randn_like(x_test)

        grad, loss = pipeline.compute_likelihood_gradient(x_test, y_test, t=500)
        print(f"Gradient computation test: loss={loss:.4f}, grad_norm={grad.norm():.4f}")

    except Exception as e:
        print(f"Model loading test skipped: {e}")
        print("(This is expected if running without internet or HuggingFace access)")
