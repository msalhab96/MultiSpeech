import torch
import torchaudio
from torch import Tensor
from pathlib import Path
from typing import Tuple, Union
from utils import get_resampler
from interfaces import ITokenizer, IPipeline


class AudioPipeline(IPipeline):
    def __init__(
            self,
            sampling_rate: int,
            win_size: int,
            hop_size: int,
            n_mels: int,
            n_fft: int
            ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            n_mels=n_mels
        )

    def run(self, audio_path: Union[Path, str]) -> Tensor:
        x, sr = torchaudio.load(audio_path)
        x = get_resampler(sr, self.sampling_rate)(x)
        x = self.mel_spec(x)
        x = torch.squeeze(x)
        x = x.permute(1, 0)
        return x


class TextPipeline(IPipeline):
    def __init__(
            self,
            tokenizer: ITokenizer,
            ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def run(self, text: str) -> Tensor:
        text = text.lower()
        text = text.strip()
        result = self.tokenizer.tokens2ids(text)
        result.append(self.tokenizer.special_tokens.eos_id)
        return torch.LongTensor(result)


def get_audio_pipeline(aud_params):
    return AudioPipeline(**aud_params)


def get_pipelines(
        tokenizer: ITokenizer, aud_params: dict
        ) -> Tuple[IPipeline, IPipeline]:
    return (
        get_audio_pipeline(aud_params),
        TextPipeline(tokenizer=tokenizer)
    )
