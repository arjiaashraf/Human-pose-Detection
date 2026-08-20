import torch
import torch.nn as nn


# ============================================================
# Double Convolution Block
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# UNetPose
# ============================================================

class UNetPose(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_keypoints=17
    ):
        super().__init__()

        # ====================================================
        # Encoder
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # Bottleneck
        # ====================================================

        self.bottleneck = DoubleConv(
            256,
            256
        )

        # ====================================================
        # Decoder
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        # 128 from up4 + 256 from e4 = 384
        self.dec4 = DoubleConv(
            384,
            128
        )

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        # 64 from up3 + 128 from e3 = 192
        self.dec3 = DoubleConv(
            192,
            64
        )

        # ====================================================
        # Heatmap Head
        # ====================================================

        self.heatmap_head = nn.Conv2d(
            64,
            num_keypoints,
            kernel_size=1
        )

        # ====================================================
        # Visibility Head
        # ====================================================

        self.visibility_head = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(
                64,
                num_keypoints
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # ====================================================
        # Encoder
        # ====================================================

        e1 = self.enc1(x)
        # [B, 32, 256, 256]

        e2 = self.enc2(
            self.pool(e1)
        )
        # [B, 64, 128, 128]

        e3 = self.enc3(
            self.pool(e2)
        )
        # [B, 128, 64, 64]

        e4 = self.enc4(
            self.pool(e3)
        )
        # [B, 256, 32, 32]

        # ====================================================
        # Bottleneck
        # ====================================================

        b = self.bottleneck(
            self.pool(e4)
        )
        # [B, 256, 16, 16]

        # ====================================================
        # Decoder
        # ====================================================

        d4 = self.up4(b)
        # [B, 128, 32, 32]

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )
        # [B, 384, 32, 32]

        d4 = self.dec4(d4)
        # [B, 128, 32, 32]

        d3 = self.up3(d4)
        # [B, 64, 64, 64]

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )
        # [B, 192, 64, 64]

        d3 = self.dec3(d3)
        # [B, 64, 64, 64]

        # ====================================================
        # Output Heads
        # ====================================================

        heatmaps = self.heatmap_head(d3)
        # [B, 17, 64, 64]

        visibility = self.visibility_head(d3)
        # [B, 17]

        return heatmaps, visibility