import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import HEATMAP_SIZE


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.conv = ConvBNAct(
            in_ch + skip_ch,
            out_ch
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNetPose(nn.Module):

    def __init__(self, num_keypoints=17):
        super().__init__()

        # ============================================================
        # ENCODER
        # ============================================================

        self.enc1 = ConvBNAct(
            3,
            32
        )

        self.enc2 = ConvBNAct(
            32,
            64,
            stride=2
        )

        self.enc3 = ConvBNAct(
            64,
            128,
            stride=2
        )

        self.enc4 = ConvBNAct(
            128,
            256,
            stride=2
        )

        # 48 -> 24
        self.bottleneck = ConvBNAct(
            256,
            256,
            stride=2
        )

        # ============================================================
        # DECODER
        # ============================================================

        # 24 -> 48
        self.up4 = UpBlock(
            256,
            256,
            256
        )

        # 48 -> 96
        self.up3 = UpBlock(
            256,
            128,
            128
        )

        # 96 -> 192
        self.up2 = UpBlock(
            128,
            64,
            64
        )

        # 192 -> 384
        self.up1 = UpBlock(
            64,
            32,
            32
        )

        # ============================================================
        # HEATMAP FEATURES
        #
        # d1 is 384x384.
        # We need 96x96 output.
        #
        # 384 -> 192 -> 96
        # ============================================================

        self.final_down = nn.Sequential(

            # 384 -> 192
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 192 -> 96
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ============================================================
        # HEATMAP HEAD
        # ============================================================

        self.heatmap_head = nn.Conv2d(
            64,
            num_keypoints,
            kernel_size=1
        )

        # ============================================================
        # VISIBILITY HEAD
        #
        # Uses bottleneck features: 256 channels
        # ============================================================

        self.visibility_head = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.15),

            nn.Linear(
                128,
                num_keypoints
            )
        )

    def forward(self, x):

        # ============================================================
        # ENCODER
        # ============================================================

        e1 = self.enc1(x)
        # [B, 32, 384, 384]

        e2 = self.enc2(e1)
        # [B, 64, 192, 192]

        e3 = self.enc3(e2)
        # [B, 128, 96, 96]

        e4 = self.enc4(e3)
        # [B, 256, 48, 48]

        b = self.bottleneck(e4)
        # [B, 256, 24, 24]

        # ============================================================
        # DECODER
        # ============================================================

        d4 = self.up4(b, e4)
        # [B, 256, 48, 48]

        d3 = self.up3(d4, e3)
        # [B, 128, 96, 96]

        d2 = self.up2(d3, e2)
        # [B, 64, 192, 192]

        d1 = self.up1(d2, e1)
        # [B, 32, 384, 384]

        # ============================================================
        # HEATMAP FEATURES
        # ============================================================

        features = self.final_down(d1)
        # [B, 64, 96, 96]

        heatmaps = self.heatmap_head(features)
        # [B, 17, 96, 96]

        # ============================================================
        # SAFETY CHECK
        #
        # Guarantees the output exactly matches config.
        # ============================================================

        if heatmaps.shape[-2:] != (
            HEATMAP_SIZE,
            HEATMAP_SIZE
        ):

            heatmaps = F.interpolate(
                heatmaps,
                size=(
                    HEATMAP_SIZE,
                    HEATMAP_SIZE
                ),
                mode="bilinear",
                align_corners=False
            )

        # ============================================================
        # VISIBILITY
        # ============================================================

        visibility = self.visibility_head(b)
        # [B, 17]

        return heatmaps, visibility


class SoftArgmax2D(nn.Module):

    def __init__(self, beta=50.0):
        super().__init__()

        self.beta = beta

    def forward(self, heatmaps):

        b, k, h, w = heatmaps.shape

        # Flatten spatial dimensions
        flat = heatmaps.reshape(
            b,
            k,
            -1
        )

        # Convert heatmaps into probability distributions
        probs = F.softmax(
            flat * self.beta,
            dim=-1
        )

        # Normalized coordinates [0, 1]
        ys = torch.linspace(
            0.0,
            1.0,
            h,
            device=heatmaps.device,
            dtype=heatmaps.dtype
        )

        xs = torch.linspace(
            0.0,
            1.0,
            w,
            device=heatmaps.device,
            dtype=heatmaps.dtype
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij"
        )

        xx = xx.reshape(
            1,
            1,
            -1
        )

        yy = yy.reshape(
            1,
            1,
            -1
        )

        x = (probs * xx).sum(-1)

        y = (probs * yy).sum(-1)

        coords = torch.stack(
            [x, y],
            dim=-1
        )

        return coords