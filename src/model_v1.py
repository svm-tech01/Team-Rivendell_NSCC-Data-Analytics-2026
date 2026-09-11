"""
NSSC 2026 - Model V1
Baseline convolutional autoencoder with a 256-D latent space.

This reproduces the V1 architecture used in the final notebook:
- 227x227 grayscale input
- 4 strided convolution stages
- AdaptiveAvgPool2d(4,4)
- 256-D latent vector
- transposed-convolution decoder
- final bilinear resize to 227x227

The V1 baseline used MSE-only training in the final comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_SIZE = 227
LATENT_DIM = 256


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, base_channels=32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, base_channels=32,
                 output_size=IMG_SIZE):
        super().__init__()

        self.base_channels = base_channels
        self.output_size = output_size

        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, base_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, 4, 4)
        x = self.deconv(x)
        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(x)


class ConvAutoencoderV1(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ConvAutoencoderV1()
    print(f"Trainable parameters: {count_parameters(model):,}")

    x = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
    reconstruction, latent = model(x)
    print("Input shape:         ", x.shape)
    print("Latent shape:        ", latent.shape)
    print("Reconstruction shape:", reconstruction.shape)
