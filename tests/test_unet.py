import torch
from src.unet import get_sinusoidal_time_embedding
from src.unet import UNet

def test_embedding_shape():
    t = torch.randint(0, 1000, (4,))
    emb = get_sinusoidal_time_embedding(128, t)
    assert emb.shape == (4, 128)

def test_embedding_range():
    t = torch.randint(0, 1000, (4,))
    emb = get_sinusoidal_time_embedding(128, t)
    assert emb.min() >= -1 and emb.max() <= 1

def test_unet_forward_pass():
    model = UNet(in_channels=1, base_channels=64, time_dim=128)
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)
    assert out.shape == (4, 1, 28, 28)