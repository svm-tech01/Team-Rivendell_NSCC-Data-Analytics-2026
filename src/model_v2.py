"""
NSSC 2026 - Model V2 (Final Autoencoder)
Improved convolutional autoencoder used for the final latent representation.

Changes from V1:
- Removes AdaptiveAvgPool2d bottleneck.
- Uses an additional learned strided-convolution stage to reach 8x8.
- Projects 8x8x256 features into a 256-D latent vector.
- Decoder reconstructs to 128x128 and then performs 128->227 bilinear resize.
- Adds a learned convolutional refinement block after interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_SIZE = 227
LATENT_DIM = 256


class EncoderV2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, base_channels=32):
        super().__init__()

        self.features = nn.Sequential(
            # 227 -> 114
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # 114 -> 57
            nn.Conv2d(base_channels, base_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 57 -> 29
            nn.Conv2d(base_channels * 2, base_channels * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 29 -> 15
            nn.Conv2d(base_channels * 4, base_channels * 8,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 15 -> 8
            nn.Conv2d(base_channels * 8, base_channels * 8,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(
            base_channels * 8 * 8 * 8,
            latent_dim,
        )

        self.base_channels = base_channels

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class DecoderV2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, base_channels=32,
                 output_size=IMG_SIZE):
        super().__init__()

        self.base_channels = base_channels
        self.output_size = output_size

        self.fc = nn.Linear(
            latent_dim,
            base_channels * 8 * 8 * 8,
        )

        self.deconv = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # 16 -> 32
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # 32 -> 64
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(base_channels, base_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, 8, 8)
        x = self.deconv(x)

        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

        x = self.refine(x)
        return torch.sigmoid(x)


class ConvAutoencoderV2(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = EncoderV2(latent_dim)
        self.decoder = DecoderV2(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ConvAutoencoderV2()
    print(f"Trainable parameters: {count_parameters(model):,}")

    x = torch.rand(2, 1, IMG_SIZE, IMG_SIZE)
    reconstruction, latent = model(x)
    print("Input shape:         ", x.shape)
    print("Latent shape:        ", latent.shape)
    print("Reconstruction shape:", reconstruction.shape)
