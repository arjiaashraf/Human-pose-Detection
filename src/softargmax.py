import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax2D(nn.Module):

    def __init__(self, beta=100.0):
        super().__init__()

        self.beta = beta

    def forward(self, heatmaps):

        B, K, H, W = heatmaps.shape

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        flat = heatmaps.reshape(
            B,
            K,
            H * W
        )

        # ----------------------------------------------------
        # Spatial probability
        # ----------------------------------------------------

        probabilities = F.softmax(
            flat * self.beta,
            dim=-1
        )

        # ----------------------------------------------------
        # Coordinate grid
        # ----------------------------------------------------

        y_grid, x_grid = torch.meshgrid(
            torch.arange(
                H,
                device=heatmaps.device,
                dtype=heatmaps.dtype
            ),
            torch.arange(
                W,
                device=heatmaps.device,
                dtype=heatmaps.dtype
            ),
            indexing="ij"
        )

        x_grid = x_grid.reshape(-1)
        y_grid = y_grid.reshape(-1)

        # ----------------------------------------------------
        # Expected coordinate
        # ----------------------------------------------------

        x = torch.sum(
            probabilities * x_grid,
            dim=-1
        )

        y = torch.sum(
            probabilities * y_grid,
            dim=-1
        )

        return torch.stack(
            [x, y],
            dim=-1
        )