"""
Diffusion Model Trainer

Trains a UNet to predict noise at random diffusion timesteps on MNIST
using the DDPM forward process

Usage:
    python -m src.train
"""

from src.diffusion import T, beta_start, beta_end, get_schedule, q_sample
from src.unet import UNet
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json


# use M2 gpu otherwise cpu
def train(epochs=15, batch_size=128, learning_rate=2e-4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, _, alpha_bars = get_schedule(T, beta_start, beta_end)  # ignore betas & alphas
    alpha_bars = alpha_bars.to(device)
    # scale tensor from [0,1] to [-1,1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST("data", train=True, transform=transform, download=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )  # batching dataset
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for step, (images, _) in enumerate(
            tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        ):
            images = images.to(device)
            timesteps = torch.randint(
                low=0, high=T, size=[images.shape[0]], device=device
            )
            noise = torch.randn_like(images)
            noisy_images = q_sample(images, timesteps, alpha_bars, noise)
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(input=predicted_noise, target=noise)

            # core functions for optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if step % 100 == 0:
                print(f"\nepoch {epoch} step {step} loss {loss.item():.4f}")
            avg_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(avg_loss)
            print(f"\nepoch {epoch} avg {avg_loss:.4f}")
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch-1}.pt")

    with open("checkpoints/loss_history.json", "w") as file:
        json.dump(epoch_losses, file)


if __name__ == "__main__":
    train(epochs=15)
