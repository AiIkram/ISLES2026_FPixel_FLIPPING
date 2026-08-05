"""
Custom nnU-Net trainer: DC_and_Focal_loss.

NOTE: as of nnunetv2 (checked against 2.8.1), there is NO 'DC_and_Focal_loss'
or standalone focal-loss class anywhere in nnunetv2.training.loss — only
DC_and_CE_loss, DC_and_BCE_loss, and DC_and_topk_loss exist. So this file
defines the Focal loss and the combined Dice+Focal loss itself, following
the exact same conventions as nnunetv2's own DC_and_CE_loss
(nnunetv2/training/loss/compound_losses.py) and RobustCrossEntropyLoss
(nnunetv2/training/loss/robust_ce_loss.py), rather than importing something
that doesn't exist in the installed package.
"""

import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalLoss(nn.Module):
    """
    Multiclass focal loss, softmax-based, matching nnU-Net's target
    convention: target is (b, 1, x, y(, z)) integer class labels, input
    is (b, c, x, y(, z)) LOGITS (not probabilities).
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        log_probs = torch.log_softmax(net_output, dim=1)
        # gather the log-prob of the true class at every voxel
        target_for_gather = target.unsqueeze(1)
        log_pt = log_probs.gather(1, target_for_gather).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1 - pt).pow(self.gamma)
        loss = -self.alpha * focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DC_and_Focal_loss(nn.Module):
    """Combined Dice + Focal loss, same call signature/behavior style as
    nnunetv2's own DC_and_CE_loss."""

    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1,
                 dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

        self.focal = FocalLoss(**focal_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        focal_loss = self.focal(net_output, target) if self.weight_focal != 0 else 0
        return self.weight_dice * dc_loss + self.weight_focal * focal_loss


class nnUNetTrainerDiceFocal(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_and_Focal_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
            },
            focal_kwargs={"alpha": 0.5, "gamma": 2.0},
            weight_focal=1,
            weight_dice=1,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
