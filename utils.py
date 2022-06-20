import math
import torch
from torch import Tensor
from functools import lru_cache
import json
import torchaudio


@lru_cache(maxsize=2)
def get_positionals(max_length: int, d_model: int) -> Tensor:
    """Create Positionals tensor to be added to the input
    Args:
        max_length (int): The maximum length of the positionals sequence.
        d_model (int): The dimensionality of the positionals sequence.
    Returns:
        Tensor: Positional tensor
    """
    result = torch.zeros(max_length, d_model, dtype=torch.float)
    for pos in range(max_length):
        for i in range(0, d_model, 2):
            denominator = pow(10000, 2 * i / d_model)
            result[pos, i] = math.sin(pos / denominator)
            result[pos, i + 1] = math.cos(pos / denominator)
    return result


def cat_speaker_emb(speaker_emb: Tensor, x: Tensor) -> Tensor:
    """Concat the speaker embedding to the prenet/encoder results.

    Args:
        speaker_emb (Tensor): The speaker embedding of dimension [B, 1, E]
        x (Tensor): The results to be concatenated with of shape [B, M, E]

    Returns:
        Tensor: The concatenated speaker embedding with the input x of of shape
        [B, M + 1, E]
    """
    return torch.cat([speaker_emb, x], dim=1)


def save_json(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


@lru_cache(maxsize=2)
def get_resampler(src_sr: int, target_sr: int):
    return torchaudio.transforms.Resample(
        orig_freq=src_sr,
        new_freq=target_sr
    )
