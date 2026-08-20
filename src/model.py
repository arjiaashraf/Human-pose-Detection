import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
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


class UNetPose(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_keypoints=17
    ):
        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

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
            2,
            2
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            256,
            256
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

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

        self.dec3 = DoubleConv(
            192,
            64
        )

        # ----------------------------------------------------
        # Heads
        # ----------------------------------------------------

        self.heatmap_head = nn.Conv2d(
            64,
            num_keypoints,
            kernel_size=1
        )

        self.visibility_head = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                32,
                num_keypoints
            )
        )

        # ----------------------------------------------------
        # Initialize heatmap head
        # ----------------------------------------------------

        nn.init.normal_(
            self.heatmap_head.weight,
            mean=0.0,
            std=0.001
        )

        nn.init.constant_(
            self.heatmap_head.bias,
            -2.0
        )

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        # Outputs

        heatmaps = self.heatmap_head(d3)

        visibility = self.visibility_head(d3)

        return (
            heatmaps,
            visibility
        )