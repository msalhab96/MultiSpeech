from typing import List, Union
from torch import Tensor
import torch


class Loss(torch.nn.Module):
    def __init__(
            self,
            h: int,
            dc_strength: float,
            dc_bandwidth: int,
            stop_weight: Union[float, int]
            ) -> None:
        super().__init__()
        self.h = h
        self.dc_strength = dc_strength
        self.l1_loss = torch.nn.L1Loss()
        self.bce = torch.nn.BCELoss()
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
            total = total + torch.nn.BCELoss(
                weight=wieght,
                reduction='sum'
                )(p_preds, target)
        return total / mask.sum()

    def calc_diagonal_constraint(
            self, alignments: List[Tensor], lengths: Tensor
            ) -> Tensor:
        alignments = torch.stack(alignments, dim=0)
        n, bh, s, t = alignments.shape
        batch_size = bh // self.h
        alignments = alignments.view(n, self.h, batch_size, s, t)
        alignments = alignments.permute(0, 2, 1, 3, 4)
        # As the data will be sorted by length, we assume all the batch items
        # approximately has the same length so k constant across all the batch
        # items
        k = t // s
        time_range = torch.arange(0, t)
        min_indices = torch.clip(
            k * time_range - self.dc_bandwidth, min=0, max=s
            ).long()
        max_indices = torch.clip(
            k * time_range + self.dc_bandwidth, min=0, max=s
            ).long()
        result = 0
        for i in range(t):
            min_idx = min_indices[i].item()
            max_idx = max_indices[i].item()
            mask = t >= lengths
            mask = mask.view(1, -1, 1, 1, 1)
            mask = mask.to(alignments.device)
            total = alignments[..., min_idx:max_idx, i:i+1] * mask
            result = result + total.sum()
        result = result / (self.h * lengths.sum() * n)
        return result

    def forward(
            self,
            lengths: Tensor,
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
        diagonal_constraint = self.calc_diagonal_constraint(
            alignments, lengths
            )
        total_loss = mel_loss + stop_loss
        total_loss = total_loss - self.dc_strength * diagonal_constraint
        return total_loss
