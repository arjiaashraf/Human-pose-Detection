import torch

import torch.nn as nn

import torch.nn.functional as F

from src.model import SoftArgmax2D


class PoseLoss(nn.Module):

    def __init__(
        self,
        heatmap_weight=1.0,
        coord_weight=2.0,
        vis_weight=0.25,
        coord_huber_beta=0.03,
        softargmax_beta=50.0,
    ):

        super().__init__()

        self.heatmap_weight = (
            heatmap_weight
        )

        self.coord_weight = (
            coord_weight
        )

        self.vis_weight = (
            vis_weight
        )

        self.coord_huber_beta = (
            coord_huber_beta
        )

        self.softargmax = SoftArgmax2D(
            beta=softargmax_beta
        )

        self.visibility_loss = (
            nn.BCEWithLogitsLoss()
        )

    def forward(
        self,
        pred_heatmaps,
        pred_visibility,
        target_heatmaps,
        target_coords,
        target_visibility,
        visible,
    ):

        # ====================================================
        # HEATMAP LOSS
        # ====================================================

        heatmap_loss = F.mse_loss(
            pred_heatmaps,
            target_heatmaps
        )

        # ====================================================
        # COORDINATES
        # ====================================================

        pred_coords = self.softargmax(
            pred_heatmaps
        )

        # Use labeled keypoints.
        coord_mask = (
            target_visibility > 0
        ).float()

        coord_mask = coord_mask.unsqueeze(
            -1
        )

        coordinate_difference = (
            pred_coords
            -
            target_coords
        )

        abs_diff = (
            coordinate_difference.abs()
        )

        beta = self.coord_huber_beta

        huber = torch.where(
            abs_diff < beta,

            0.5
            *
            abs_diff.pow(2)
            / beta,

            abs_diff
            -
            0.5 * beta
        )

        huber = huber * coord_mask

        valid_values = coord_mask.sum() * 2.0

        coordinate_loss = (
            huber.sum()
            /
            valid_values.clamp(
                min=1.0
            )
        )

        # ====================================================
        # VISIBILITY
        # ====================================================

        visibility_loss = (
            self.visibility_loss(
                pred_visibility,
                target_visibility
            )
        )

        # ====================================================
        # TOTAL
        # ====================================================

        total_loss = (

            self.heatmap_weight
            * heatmap_loss

            +

            self.coord_weight
            * coordinate_loss

            +

            self.vis_weight
            * visibility_loss
        )

        return (
            total_loss,
            heatmap_loss,
            coordinate_loss,
            visibility_loss,
            pred_coords
        )