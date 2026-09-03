import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax2D(nn.Module):

    def __init__(
        self,
        beta=50.0
    ):

        super().__init__()

        self.beta = beta

    def forward(
        self,
        heatmaps
    ):

        B, K, H, W = heatmaps.shape

        # ====================================================
        # FLATTEN
        # ====================================================

        flat = heatmaps.reshape(
            B,
            K,
            -1
        )

        # ====================================================
        # SOFTMAX
        #
        # Higher beta = sharper distribution.
        # ====================================================

        probabilities = F.softmax(
            flat * self.beta,
            dim=-1
        )

        # ====================================================
        # COORDINATE GRID
        # ====================================================

        y = torch.arange(
            H,
            device=heatmaps.device,
            dtype=heatmaps.dtype
        )

        x = torch.arange(
            W,
            device=heatmaps.device,
            dtype=heatmaps.dtype
        )

        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij"
        )

        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # ====================================================
        # EXPECTED COORDINATES
        # ====================================================

        pred_x = torch.sum(
            probabilities * xx,
            dim=-1
        )

        pred_y = torch.sum(
            probabilities * yy,
            dim=-1
        )

        return torch.stack(
            [
                pred_x,
                pred_y
            ],
            dim=-1
        )