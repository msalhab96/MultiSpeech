from typing import List, Union
from torch import Tensor
import torch.nn as nn
import torch


class Loss(nn.Module):
    def __init__(
            self,
            h: int,
            dc_strength: float,
            dc_bandwidth: int,
            stop_weight: Union[float, int]
            ) -> None:
        self.h = h
        self.dc_strength = dc_strength
        self.l1_loss = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.stop_weight = stop_weight
        self.dc_bandwidth = dc_bandwidth

    def calc_bce_loss(self, mask: Tensor, stop_pred: Tensor):
        # stop_pred of shape [B, M]
        # mask of shape [B, M]
        total = 0
        for m, p in zip(mask, stop_pred):
            p_preds = torch.masked_select(p, m)
            target = torch.zeros(*p_preds.shape)
            target[-1] = 1
            wieght = target * self.stop_weight
            target = target.to(stop_pred.device)
            wieght = wieght.to(stop_pred.device)
            total = total + nn.BCELoss(
                weight=wieght,
                reduction='sum'
                )(p_preds, target)
        return total / mask.sum()

    def _compute_dc(self, alignment: Tensor, lengths):
        bh, s, t = alignment.shape
        batch_size = bh // self.h
        alignment = alignment.view(self.h, batch_size, s, t)
        alignment = alignment.permute(1, 0, 2, 3)
        k = t // s
        time_range = torch.arange(0, t)
        result = 0
        for i in range(t):
            min_idx = max(0, k * time_range - self.dc_bandwidth)
            max_idx = min(s, k * time_range + self.dc_bandwidth)
            mask = t >= lengths
            mask = mask.view(-1, 1, 1, 1)
            mask = mask.to(alignment.device)
            total = alignment[..., min_idx:max_idx, i:i+1] * mask
            result = result + total.sum()
        return result

    def calc_diagonal_constraint(
            self,
            alignments: List[Tensor],
            lengths: Tensor
            ) -> Tensor:
        total = 0
        for alignment in alignments:
            total = total + self._compute_dc(alignment, lengths)
        total = total / (self.h * lengths.sum())
        return total

    def forward(
            self,
            length: Tensor,
            mask: Tensor,
            stop_pred: Tensor,
            mels_pred: Tensor,
            mels_target: Tensor,
            alignments: Tensor
            ) -> Tensor:
        mel_loss = self.l1_loss(
            mels_pred * mask.unsqueeze(dim=-1), mels_target
            )
        stop_loss = self.calc_bce_loss(mask, stop_pred)
        diagonal_constraint = self.calc_diagonal_constraint(alignments, length)
        total_loss = mel_loss + stop_loss
        total_loss = total_loss - self.dc_strength * diagonal_constraint
        return total_loss
