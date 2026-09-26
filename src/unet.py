import torch
import torch.nn as nn

def get_sinusoidal_time_embedding(d, t):
    i = torch.arange(d//2)
    denominator = torch.pow(10000, 2*i/d)

    #reshaped
    denominator = denominator[None, :]
    t = t[:, None]

    t = t/denominator #scaled
    sines = torch.sin(t)
    cosines = torch.cos(t)
    t_embedding = torch.cat([sines, cosines], dim=-1)
    return t_embedding

class TimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))

    def forward(self, t):
        embedding = get_sinusoidal_time_embedding(self.dim, t)
        return self.mlp(embedding)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_layer = nn.Linear(dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, t_embedding):
        features = self.act(self.norm1(self.conv1(x)))
        t_projection = self.time_layer(t_embedding)[:,:,None,None]
        features = features + t_projection
        features = self.act(self.norm2(self.conv2(features)))
        return features

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.down1 = ConvBlock(in_channels, base_channels, time_dim)
        self.down2 = ConvBlock(base_channels, base_channels*2, time_dim)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels*2, base_channels*2, time_dim)
        self.up1 = ConvBlock(base_channels*2 + base_channels*2, base_channels, time_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = ConvBlock(base_channels * 2, base_channels, time_dim)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_embedding = self.time_embedding(t)
        h1 = self.down1(x, t_embedding)
        h2 = self.down2(self.pool(h1), t_embedding)
        b = self.bottleneck(self.pool(h2), t_embedding)
        u = self.upsample(b)
        u = torch.cat([u, h2], dim=1)
        u = self.up1(u, t_embedding)
        u = self.upsample(u)
        u = torch.cat([u, h1], dim=1)
        u = self.up2(u, t_embedding)
        return self.out_conv(u)