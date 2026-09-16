import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

T = 1000
beta_start = 1e-4
beta_end = 0.02


def linear_beta_schedule(T, beta_start, beta_end):
    """Implements linear beta schedule from beta_start to beta_end in T timesteps"""
    return torch.linspace(beta_start, beta_end, T)


def q_sample(x_0, t, alpha_bars, noise=None):
    """Calculate noisy x_t at timestep t from x_0 using  closed-form sampling equation"""
    if noise is None:
        noise = torch.randn_like(x_0)
    alpha_bars_t = alpha_bars[t]
    alpha_bars_t = alpha_bars_t.view(
        -1, 1, 1, 1
    )  # reshape to [B,1,1,1] for broadcasting
    sqrt_alpha_bar_t = torch.sqrt(alpha_bars_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars_t)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

    return x_t


if __name__ == "__main__":
    betas = linear_beta_schedule(T, beta_start, beta_end)

    # check beta, alpha_bars vals
    print(betas.shape)
    print(betas[0], betas[-1])
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    print(alphas.shape)
    print(alpha_bars.shape)
    print(alpha_bars[0], alpha_bars[-1])

    # check shape of q_sample
    x_0 = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, T, (4,))
    x_t = q_sample(x_0, t, alpha_bars)
    print(x_t.shape)

    # check MNIST image
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
    plt.show()