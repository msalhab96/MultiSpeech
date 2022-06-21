from interfaces import IPadder
from torch import Tensor
import torch


class TextPadder(IPadder):
    def __init__(
            self,
            pad_id: int
            ) -> None:
        super().__init__()
        self.pad_id = pad_id

    def pad(self, x: Tensor, max_len: int) -> Tensor:
        length = x.shape[0]
        pad = torch.ones(max_len - length, dtype=torch.int) * self.pad_id
        return torch.cat([x, pad], dim=0)


class AudPadder(IPadder):
    def __init__(
            self,
            pad_val: int,
            ) -> None:
        super().__init__()
        self.pad_val = pad_val

    def pad(self, x: Tensor, max_len: int) -> Tensor:
        length, dim = x.shape
        pad = torch.ones(max_len - length, dim, dtype=torch.int) * self.pad_val
        return torch.cat([x, pad], dim=0)
