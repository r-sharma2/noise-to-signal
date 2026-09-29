"""
Diffusion Model Sampler

Generates images from pure noise using the trained UNet and DDPM reverse process.

Usage:
    python -m src.sample
"""

from src.unet import UNet
from src.diffusion import T, beta_start, beta_end, get_schedule
import torch


# no training here
@torch.no_grad()
def sample(model, shape, betas, alphas, alpha_bars, device):
    model.eval()
    images = torch.randn(shape, device=device)
    for step in reversed(range(T)):
        timesteps = torch.full((shape[0],), step, device=device)
        predicted_noise = model(images, timesteps)

        # schedule values
        beta = betas[step]
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]

        # denoising formula
        mean = (1 / torch.sqrt(alpha)) * (
            images - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
        )

        if step != 0:  # add noise
            noise = torch.randn_like(images)
            images = mean + torch.sqrt(beta) * noise
        else:  # step 0 = clean image
            images = mean
    return images


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    betas, alphas, alpha_bars = get_schedule(T, beta_start, beta_end)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)
    model = UNet().to(device)
    model.load_state_dict(torch.load("checkpoints/epoch_38.pt", map_location=device))
    samples = sample(model, (16, 1, 28, 28), betas, alphas, alpha_bars, device)
    torch.save(samples, "outputs/samples.pt")
    print("Done! Saved --> outputs/samples.pt")