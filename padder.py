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
        length = x.shape[1]
        pad = torch.ones(1, max_len - length) * self.pad_id
        return torch.cat([x, pad], dim=1)


class AudPadder(IPadder):
    def __init__(
            self,
            pad_val: int,
            n_mels: int
            ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.pad_val = pad_val

    def pad(self, x: Tensor, max_len: int) -> Tensor:
        length = x.shape[1]
        pad = torch.ones(1, 1, max_len - length) * self.pad_val
        return torch.cat([x, pad], dim=1)
