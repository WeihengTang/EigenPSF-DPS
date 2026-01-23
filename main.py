#!/usr/bin/env python3
"""
EigenPSF-DPS: Non-blind Spatially Varying Deconvolution via Diffusion Posterior Sampling
========================================================================================

Main entry point for running EigenPSF-DPS deconvolution.

This script:
1. Loads configuration from YAML file
2. Sets up the physics model (spatially varying blur with EigenPSF decomposition)
3. Generates or loads a test image
4. Creates the measurement (blurred + noisy image)
5. Runs DPS restoration
6. Saves and visualizes results

Usage:
    python main.py                          # Use default config.yaml
    python main.py --config custom.yaml     # Use custom config
    python main.py --step_size 50.0         # Override specific parameters

Author: EigenPSF-DPS Research Team
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch import Tensor
import matplotlib.pyplot as plt

# Local imports
from physics import create_sv_blur_system, EigenPSFDecomposition, SVBlurOperator, TrueSVBlurOperator
from dps import create_dps_pipeline, download_model, DPSResult, EigenPSFDPSPipeline
from utils import (
    load_config,
    save_config,
    setup_logging,
    get_device,
    get_dtype,
    set_seed,
    load_sample_image,
    save_image,
    tensor_to_image,
    visualize_results,
    visualize_eigenpsfs,
    visualize_intermediate_results,
    compute_metrics,
    print_metrics,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EigenPSF-DPS: Spatially Varying Deconvolution via Diffusion Posterior Sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )

    # Override options (these take precedence over config file)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use for computation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # Physics overrides
    parser.add_argument(
        "--blur_mode",
        type=str,
        choices=["motion", "defocus", "mixed"],
        help="Type of spatially varying blur",
    )
    parser.add_argument(
        "--n_eigen_psfs",
        type=int,
        help="Number of EigenPSF components",
    )
    parser.add_argument(
        "--sigma_noise",
        type=float,
        help="Measurement noise standard deviation",
    )

    # DPS overrides
    parser.add_argument(
        "--step_size",
        type=float,
        help="DPS gradient step size (zeta)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        help="Number of diffusion sampling steps",
    )
    parser.add_argument(
        "--skip_steps",
        type=int,
        help="Number of initial steps to skip",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable visualization (useful for headless servers)",
    )

    # Image source
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (or 'sample' for synthetic)",
    )

    # Model download options
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download and cache the model, then exit (useful on clusters)",
    )
    parser.add_argument(
        "--model_cache",
        type=str,
        help="Directory for caching downloaded model weights",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of download retry attempts",
    )

    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command-line overrides to configuration.

    Args:
        config: Base configuration dictionary
        args: Parsed command-line arguments

    Returns:
        Updated configuration dictionary
    """
    # Device override
    if args.device is not None:
        config["model"]["device"] = args.device

    # Seed override
    if args.seed is not None:
        config["seed"] = args.seed

    # Physics overrides
    if args.blur_mode is not None:
        config["physics"]["blur_mode"] = args.blur_mode
    if args.n_eigen_psfs is not None:
        config["physics"]["n_eigen_psfs"] = args.n_eigen_psfs
    if args.sigma_noise is not None:
        config["physics"]["sigma_noise"] = args.sigma_noise

    # DPS overrides
    if args.step_size is not None:
        config["dps"]["step_size"] = args.step_size
    if args.num_inference_steps is not None:
        config["dps"]["num_inference_steps"] = args.num_inference_steps
    if args.skip_steps is not None:
        config["dps"]["skip_steps"] = args.skip_steps

    # Output overrides
    if args.output_dir is not None:
        config["output"]["save_dir"] = args.output_dir

    # Image source override
    if args.image is not None:
        config["data"]["image_source"] = args.image

    return config


def setup_output_directory(config: Dict[str, Any]) -> Path:
    """Create and setup output directory structure.

    Args:
        config: Configuration dictionary

    Returns:
        Path to output directory
    """
    base_dir = Path(config["output"]["save_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create unique run directory
    output_dir = base_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "intermediates").mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    return output_dir


def generate_measurement(
    clean_image: Tensor,
    true_operator: TrueSVBlurOperator,
    sigma_noise: float,
    device: torch.device,
) -> Tensor:
    """Generate noisy measurement from clean image.

    Args:
        clean_image: Clean image tensor (B, C, H, W)
        true_operator: True SV blur operator
        sigma_noise: Noise standard deviation
        device: Torch device

    Returns:
        Measurement tensor y = A(x) + n
    """
    # Apply true spatially varying blur
    blurred = true_operator(clean_image)

    # Add Gaussian noise
    noise = sigma_noise * torch.randn_like(blurred)
    measurement = blurred + noise

    return measurement


def run_dps_restoration(
    pipeline: EigenPSFDPSPipeline,
    measurement: Tensor,
    config: Dict[str, Any],
    output_dir: Path,
) -> tuple[Tensor, List[Tensor], List[int]]:
    """Run DPS restoration process.

    Args:
        pipeline: Configured DPS pipeline
        measurement: Measurement tensor (blurred + noisy)
        config: Configuration dictionary
        output_dir: Output directory for intermediates

    Returns:
        DPSResult containing restored image and full diagnostics
    """
    dps_config = config["dps"]
    output_config = config["output"]

    def callback(step: int, x_t: Tensor, loss: float) -> None:
        """Callback for saving intermediate results."""
        if output_config.get("save_intermediate", True):
            save_image(
                x_t,
                output_dir / "intermediates" / f"step_{step:04d}.png"
            )

    # Create random generator for reproducibility
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(config.get("seed", 42))

    # Run DPS
    logger.info("Starting DPS restoration...")
    result = pipeline(
        y=measurement,
        num_inference_steps=dps_config.get("num_inference_steps", 1000),
        step_size=dps_config.get("step_size", 100.0),
        step_size_schedule=dps_config.get("step_size_schedule", "constant"),
        skip_steps=dps_config.get("skip_steps", 0),
        gradient_clip=dps_config.get("gradient_clip", 1.0),
        generator=generator,
        callback=callback,
        callback_steps=output_config.get("intermediate_freq", 100),
        show_progress=config["logging"].get("progress_bar", True),
    )

    # Save diagnostics
    result.save_diagnostics(output_dir / "diagnostics.json")

    return result


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a config.yaml file or specify an existing one with --config")
        return 1

    # Apply command-line overrides
    config = apply_overrides(config, args)

    # Setup logging
    setup_logging(config["logging"].get("level", "INFO"))
    logger.info("EigenPSF-DPS: Non-blind Spatially Varying Deconvolution")
    logger.info("=" * 60)

    # Handle download-only mode
    if args.download_only:
        model_id = config["model"].get("model_id", "google/ddpm-celebahq-256")
        logger.info("Download-only mode: pre-downloading model weights...")
        try:
            download_model(
                model_id=model_id,
                cache_dir=args.model_cache,
                max_retries=args.max_retries,
            )
            logger.info("Model downloaded successfully. You can now run without --download_only.")
            return 0
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return 1

    # Setup reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup device and dtype
    device = get_device(config["model"].get("device", "cuda"))
    dtype = get_dtype(config["model"].get("dtype", "float32"))

    # Setup output directory
    output_dir = setup_output_directory(config)

    # Re-initialize logging with file output now that output_dir exists
    log_file = output_dir / "run.log"
    setup_logging(config["logging"].get("level", "INFO"), log_file=log_file)
    logger.info(f"All output logged to: {log_file}")

    # Save configuration to output directory
    save_config(config, output_dir / "config.yaml")

    # Get data parameters
    img_size = config["data"].get("img_size", 256)
    physics_config = config["physics"]

    # =========================================================================
    # Step 1: Load or generate test image
    # =========================================================================
    logger.info("Loading/generating test image...")
    clean_image = load_sample_image(
        source=config["data"].get("image_source", "sample"),
        size=img_size,
        device=device,
        dtype=dtype,
    )
    logger.info(f"Image shape: {clean_image.shape}")

    # Save clean image
    save_image(clean_image, output_dir / "images" / "clean.png")

    # =========================================================================
    # Step 2: Setup physics (SV blur with EigenPSF decomposition)
    # =========================================================================
    logger.info("Setting up physics model...")
    logger.info(f"  Blur mode: {physics_config.get('blur_mode', 'motion')}")
    logger.info(f"  EigenPSF components: {physics_config.get('n_eigen_psfs', 5)}")

    eigenpsf_operator, true_operator, decomposition = create_sv_blur_system(
        img_height=img_size,
        img_width=img_size,
        config=physics_config,
        device=device,
        dtype=dtype,
    )

    logger.info(f"  Explained variance ratio: {decomposition.explained_variance_ratio:.2%}")

    # Visualize EigenPSFs
    if not args.no_viz:
        fig = visualize_eigenpsfs(
            decomposition.basis_kernels,
            decomposition.coefficient_maps,
            save_path=output_dir / "images" / "eigenpsfs.png",
        )
        plt.close(fig)

    # =========================================================================
    # Step 3: Generate measurement (blur + noise)
    # =========================================================================
    logger.info("Generating measurement...")
    sigma_noise = physics_config.get("sigma_noise", 0.01)
    logger.info(f"  Noise level (sigma): {sigma_noise}")

    measurement = generate_measurement(
        clean_image, true_operator, sigma_noise, device
    )

    # Save blurred image
    save_image(measurement, output_dir / "images" / "blurred.png")

    # Compute initial metrics
    blurred_metrics = compute_metrics(clean_image, measurement)
    logger.info(f"  Blurred PSNR: {blurred_metrics['psnr_restored']:.2f} dB")
    logger.info(f"  Blurred SSIM: {blurred_metrics['ssim_restored']:.4f}")

    # =========================================================================
    # Step 4: Create DPS pipeline
    # =========================================================================
    logger.info("Loading pre-trained diffusion model...")
    model_id = config["model"].get("model_id", "google/ddpm-celebahq-256")

    try:
        pipeline = create_dps_pipeline(
            blur_operator=eigenpsf_operator,
            model_id=model_id,
            device=device,
            dtype=dtype,
            use_v2=False,
            cache_dir=args.model_cache,
            max_retries=args.max_retries,
        )
    except Exception as e:
        logger.error(f"Failed to load diffusion model: {e}")
        logger.error("Try pre-downloading the model with: python main.py --download_only")
        logger.error("Or specify a cache directory: python main.py --model_cache ./model_cache")
        return 1

    # =========================================================================
    # Step 5: Run DPS restoration
    # =========================================================================
    result = run_dps_restoration(pipeline, measurement, config, output_dir)

    # Save restored image
    save_image(result.restored, output_dir / "images" / "restored.png")

    # =========================================================================
    # Step 6: Compute and display metrics
    # =========================================================================
    logger.info("Computing quality metrics...")
    metrics = compute_metrics(
        clean=clean_image,
        restored=result.restored,
        blurred=measurement,
    )
    print_metrics(metrics)

    # Save metrics
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("EigenPSF-DPS Results\n")
        f.write("=" * 40 + "\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    # =========================================================================
    # Step 7: Visualizations
    # =========================================================================
    if not args.no_viz:
        logger.info("Creating visualizations...")

        # Main results comparison
        fig = visualize_results(
            clean=clean_image,
            blurred=measurement,
            restored=result.restored,
            save_path=output_dir / "images" / "comparison.png",
            title=f"EigenPSF-DPS (PSNR: {metrics['psnr_restored']:.2f} dB, SSIM: {metrics['ssim_restored']:.4f})",
        )
        plt.close(fig)

        # Intermediate results
        if result.intermediates:
            fig = visualize_intermediate_results(
                intermediates=result.intermediates,
                steps=[result.timesteps[i] for i in range(0, len(result.timesteps),
                       max(1, len(result.timesteps) // len(result.intermediates)))][:len(result.intermediates)],
                save_path=output_dir / "images" / "progress.png",
            )
            plt.close(fig)

        # Loss curve
        if result.loss_history:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].semilogy(result.timesteps, result.loss_history)
            axes[0].set_xlabel("Timestep t")
            axes[0].set_ylabel("MSE Loss")
            axes[0].set_title("Measurement Consistency Loss")
            axes[0].invert_xaxis()
            axes[0].grid(True, alpha=0.3)

            axes[1].semilogy(result.timesteps, result.grad_norm_history)
            axes[1].set_xlabel("Timestep t")
            axes[1].set_ylabel("Gradient Norm")
            axes[1].set_title("Likelihood Gradient Norm")
            axes[1].invert_xaxis()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(output_dir / "images" / "loss_curve.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Loss curve saved to: {output_dir / 'images' / 'loss_curve.png'}")

    # =========================================================================
    # Step 8: Comet ML logging (optional)
    # =========================================================================
    _log_to_comet(config, metrics, result, output_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Restored PSNR: {metrics['psnr_restored']:.2f} dB (improvement: {metrics.get('psnr_improvement', 0):.2f} dB)")
    logger.info(f"  Restored SSIM: {metrics['ssim_restored']:.4f} (improvement: {metrics.get('ssim_improvement', 0):.4f})")
    logger.info(f"  Final loss: {result.loss_history[-1]:.6f}")
    logger.info(f"  Min loss: {min(result.loss_history):.6f}")
    logger.info("=" * 60)
    logger.info("Done!")

    return 0


def _log_to_comet(
    config: Dict[str, Any],
    metrics: Dict[str, float],
    result: DPSResult,
    output_dir: Path,
) -> None:
    """Log experiment results to Comet ML if available.

    Comet ML integration is optional. Install with: pip install comet_ml
    Set your API key via environment variable: COMET_API_KEY

    Args:
        config: Configuration dictionary
        metrics: Quality metrics
        result: DPS result with loss history
        output_dir: Output directory containing images
    """
    try:
        import comet_ml
    except ImportError:
        logger.debug("Comet ML not installed. Skipping experiment tracking.")
        return

    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        logger.debug("COMET_API_KEY not set. Skipping Comet ML logging.")
        return

    try:
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=os.environ.get("COMET_PROJECT_NAME", "eigenpsf-dps"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            auto_metric_logging=False,
            auto_param_logging=False,
        )

        # Log hyperparameters
        experiment.log_parameters({
            "blur_mode": config["physics"].get("blur_mode"),
            "n_eigen_psfs": config["physics"].get("n_eigen_psfs"),
            "sigma_noise": config["physics"].get("sigma_noise"),
            "step_size": config["dps"].get("step_size"),
            "num_inference_steps": config["dps"].get("num_inference_steps"),
            "gradient_clip": config["dps"].get("gradient_clip"),
            "step_size_schedule": config["dps"].get("step_size_schedule"),
            "img_size": config["data"].get("img_size"),
            "seed": config.get("seed"),
        })

        # Log final metrics
        experiment.log_metrics(metrics)

        # Log loss curve as time series
        for i, (t, loss, grad_norm) in enumerate(zip(
            result.timesteps, result.loss_history, result.grad_norm_history
        )):
            experiment.log_metric("loss", loss, step=i, epoch=t)
            experiment.log_metric("grad_norm", grad_norm, step=i, epoch=t)
            experiment.log_metric("step_size", result.step_size_history[i], step=i, epoch=t)

        # Log images
        for img_name in ["clean.png", "blurred.png", "restored.png", "comparison.png", "loss_curve.png"]:
            img_path = output_dir / "images" / img_name
            if img_path.exists():
                experiment.log_image(str(img_path), name=img_name.replace(".png", ""))

        experiment.end()
        logger.info(f"Comet ML experiment logged: {experiment.get_key()}")

    except Exception as e:
        logger.warning(f"Comet ML logging failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
