"""
Loads samples, checkpoints, training loss history, and produces plots.
"""

import json
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from src.diffusion import T, beta_start, beta_end, get_schedule, q_sample


def plot_samples():
    samples = torch.load("outputs/samples_40epoch.pt", map_location="cpu")
    images = (samples + 1) / 2
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Generated MNIST Digits")
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.savefig("outputs/samples_40epoch.png")
    plt.show()


def plot_loss_curve():
    with open("checkpoints/loss_history.json") as f:
        losses = json.load(f)

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.savefig("outputs/loss_curve.png")
    plt.show()


def plot_forward_diffusion():
    # check MNIST image
    _, _, alpha_bars = get_schedule(T, beta_start, beta_end)
    dataset = datasets.MNIST(
        "../data", train=True, transform=transforms.ToTensor(), download=True
    )
    img, _ = dataset[0]
    img = torch.unsqueeze(img, 0)
    timesteps = [0, 100, 200, 500, 700, 999]
    fig, axs = plt.subplots(nrows=1, ncols=len(timesteps), figsize=(18, 3))
    fig.suptitle("MNIST Forward Diffusion")
    for i, t_value in enumerate(timesteps):
        t = torch.tensor(t_value)
        x_t = q_sample(img, t, alpha_bars)  # noisy
        axs[i].imshow(x_t.squeeze(), cmap="gray")
        axs[i].set_title(f"t={t_value}")
    plt.savefig("outputs/forward_diffusion.png")
    plt.show()


if __name__ == "__main__":
    plot_forward_diffusion()
    plot_loss_curve()
    plot_samples("outputs/samples_40epoch.pt", "outputs/samples_40epoch.png")
