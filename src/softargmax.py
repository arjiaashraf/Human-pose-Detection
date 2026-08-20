import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax2D(nn.Module):

    def __init__(self, beta=100.0):
        super().__init__()

        self.beta = beta

    def forward(self, heatmaps):

        # heatmaps:
        # [B, K, H, W]

        B, K, H, W = heatmaps.shape

        # Flatten spatial dimensions
        heatmaps = heatmaps.view(
            B,
            K,
            -1
        )

        # Convert heatmap values into probabilities
        probabilities = F.softmax(
            heatmaps * self.beta,
            dim=-1
        )

        # ----------------------------------------------------
        # X coordinates
        # ----------------------------------------------------

        x_coords = torch.linspace(
            0,
            W - 1,
            W,
            device=heatmaps.device
        )

        y_coords = torch.linspace(
            0,
            H - 1,
            H,
            device=heatmaps.device
        )

        # Create coordinate grid
        yy, xx = torch.meshgrid(
            y_coords,
            x_coords,
            indexing="ij"
        )

        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # ----------------------------------------------------
        # Expected coordinates
        # ----------------------------------------------------

        x = torch.sum(
            probabilities * xx,
            dim=-1
        )

        y = torch.sum(
            probabilities * yy,
            dim=-1
        )

        # [B, K, 2]
        coordinates = torch.stack(
            [x, y],
            dim=-1
        )

        return coordinates