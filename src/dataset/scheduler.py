import numpy as np
import torch


class Scheduler:
    """Scheduler class for managing the noise addition process during training of a diffusion model.

    This class is responsible for generating the noise schedule and adding noise to the original samples
    based on the specified timesteps. The noise schedule is defined by the beta values, which are computed
    between the square roots of the start and end values, then squared. This ensures that the betas are
    distributed more evenly across the training steps, especially for small values of beta.

    Args:
        generator (torch.Generator): A PyTorch random number generator for reproducibility.
        training_steps (int): The total number of training steps for the diffusion model.
        beta_start (float): The starting value of beta for the noise schedule. Default is 1e-4.
        beta_end (float): The ending value of beta for the noise schedule. Default is 0.02.
    """

    def __init__(
        self,
        generator: torch.Generator,
        training_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        """Initialize the Scheduler with the specified parameters and compute the noise schedule."""
        self.generator = generator
        self.training_steps = training_steps

        # Betas are computed between the square roots of the start and end values, then squared.
        # This is done to ensure that the betas are distributed more evenly
        # across the training steps, especially for small values of beta.
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # Timesteps sequence tensor for denoising (training_steps -> 0)
        self.timesteps = torch.from_numpy(np.arange(0, training_steps)[::-1].copy())

    def add_noise(
        self,
        original_samples: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to the original samples based on the specified timesteps.

        Args:
            original_samples (torch.Tensor): The original samples to which noise will be added.
            timesteps (torch.Tensor): The timesteps at which to add noise.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the noisy samples and the generated noise.
        """

        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )

        timesteps = timesteps.to(original_samples.device)

        alphas_cumprod_sqrt = alphas_cumprod[timesteps] ** 0.5
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.flatten()

        # Ensure that the shape of alphas_cumprod_sqrt matches the shape of original_samples for broadcasting
        while len(alphas_cumprod_sqrt.shape) < len(original_samples.shape):
            alphas_cumprod_sqrt = alphas_cumprod_sqrt.unsqueeze(-1)

        alphas_cumprod_sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5
        alphas_cumprod_sqrt_one_minus = alphas_cumprod_sqrt_one_minus.flatten()

        # Ensure that the shape of alphas_cumprod_sqrt_one_minus matches the shape of original_samples for broadcasting
        while len(alphas_cumprod_sqrt_one_minus.shape) < len(original_samples.shape):
            alphas_cumprod_sqrt_one_minus = alphas_cumprod_sqrt_one_minus.unsqueeze(-1)

        # Generate noise and add it to the original samples based on the computed alphas
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        # Compute the noisy samples by combining the original samples and the generated noise
        # using the computed alphas for the specified timesteps.
        # The formula used is: x_noisy = sqrt(alpha_cumprod) * x_original + sqrt(1 - alpha_cumprod) * noise
        noisy_samples = (
            alphas_cumprod_sqrt * original_samples
            + alphas_cumprod_sqrt_one_minus * noise
        )

        return noisy_samples, noise

    def remove_noise(
        self,
        timestep: int,
        noisy_sample: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Removes noise from the noisy sample based on the specified timestep and predicted noise.

        In order to remove noise from the noisy sample, we first compute the predicted original sample (x_0)
        from the noisy sample and the predicted noise. Then, we compute the predicted previous sample (x_t-1)
        by sampling from a normal distribution with mean µ_t and variance σ_t^2,
        where µ_t is a weighted average of the predicted original sample and the current sample,
        and σ_t^2 is a weighted average of the variances for the current and previous timesteps.

        Args:
            timestep (int): The timestep at which to remove noise.
            noisy_sample (torch.Tensor): The noisy sample from which noise will be removed.
            predicted_noise (torch.Tensor): The predicted noise to be subtracted from the noisy samples.
        Returns:
            torch.Tensor: The denoised sample.
        """

        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. Compute the cumulative products of alphas and betas for the current and previous timesteps.

        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_cumprod_t = 1 - alpha_cumprod_t
        beta_cumprod_t_prev = 1 - alpha_cumprod_t_prev
        current_alpha_t = alpha_cumprod_t / alpha_cumprod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. Compute the predicted original sample (x_0) from the noisy sample and the predicted noise.
        # Knowing that the formula used for adding noise is:
        # x_noisy = sqrt(alpha_cumprod) * x_original + sqrt(1 - alpha_cumprod) * noise
        # We can rearrange this formula to solve for x_original (predicted original sample):
        # x_original = (x_noisy - sqrt(1 - alpha_cumprod) * noise) / sqrt(alpha_cumprod)
        pred_original_sample = (
            noisy_sample - (beta_cumprod_t**0.5) * predicted_noise
        ) / (alpha_cumprod_t**0.5)

        # 3. Compute the mean (µ_t) for the predicted previous sample (x_t-1).
        # Knowing that the formula for the mean (µ_t) is:
        # µ_t = pred_original_sample_coeff * x_0 + current_sample_coeff * x_t

        # The coefficients for the predicted original sample is:
        # pred_original_sample_coeff = (sqrt(alpha_cumprod_t_prev) * beta_t) / beta_cumprod_t
        pred_original_sample_coeff = (
            alpha_cumprod_t_prev ** (0.5) * current_beta_t / beta_cumprod_t
        )

        # The coefficient for the current sample is:
        # current_sample_coeff = (sqrt(alpha_t) * beta_cumprod_t_prev)
        current_sample_coeff = (
            current_alpha_t ** (0.5) * beta_cumprod_t_prev / beta_cumprod_t
        )

        # Now the mean (µ_t) for the predicted previous sample (x_t-1) can be computed as:
        # µ_t = pred_original_sample_coeff * x_0 + current_sample_coeff * x_t
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * noisy_sample
        )

        # 4. Compute the variance (σ_t^2) for the predicted previous sample (x_t-1).
        # Knowing that the formula for the variance (σ_t^2) is:
        # σ_t^2 = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * beta_t
        # Note: We only add noise from variance if we are not at the last timestep (t = 0).
        variance = 0
        if t > 0:
            variance = (
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * current_beta_t
            )

            # Clamp to avoid numerical instability.
            variance = torch.clamp(variance, min=1e-20)
            # Compute the standard deviation (σ_t) from the variance (σ_t^2).
            variance **= 0.5

            # Generate noise and scale it by the computed standard deviation (σ_t)
            # to add to the mean (µ_t) for the predicted previous sample (x_t-1).
            # The noise is generated only if we are not at the last timestep (t = 0)
            # to avoid adding noise to the final output.
            noise = torch.randn(
                predicted_noise.shape,
                generator=self.generator,
                device=predicted_noise.device,
                dtype=predicted_noise.dtype,
            )

            # Scale the generated noise by the computed standard deviation (σ_t) to obtain the variance term.
            variance = variance * noise

        # Compute the predicted previous sample (x_t-1) by adding the standard deviation term (σ_t) to the mean (µ_t).
        # Knowing that the formula for the predicted previous sample (x_t-1) is:
        # x_t-1 = µ_t + σ_t * noise
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def set_inference_timesteps(self, inference_steps: int = 50) -> None:
        """Set the timesteps for inference based on the specified number of inference steps.

        During inference, the model will use a reduced number of timesteps compared to training.
        This method computes the appropriate timesteps for inference by determining the step ratio
        between training and inference steps, and then generating a sequence of timesteps ù
        that are evenly spaced across the training steps.

        Args:
            inference_steps (int): The number of inference steps to use for denoising. Default is 50.
        """

        self.num_inference_steps = inference_steps
        step_ratio = self.training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """Get the previous timestep for the given timestep during inference.

        This method computes the previous timestep based on the current timestep and the ratio of
        training steps to inference steps. It ensures that the previous timestep is correctly calculated
        for the denoising process during inference.

        Args:
            timestep (int): The current timestep.

        Returns:
            int: The previous timestep.
        """

        prev_t = timestep - self.training_steps // self.num_inference_steps
        return prev_t
