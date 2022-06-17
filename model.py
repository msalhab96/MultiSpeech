import torch
import torch.nn as nn
from torch import Tensor
from layers import (
    Encoder,
    Decoder,
    SpeakerModule,
    PredModule,
    DecoderPrenet,
    PositionalEmbedding
    )
from interfaces import IDiagonalConstraint
from utils import cat_speaker_emb, get_positionals


class Model(nn.Module):
    def __init__(
            self,
            pos_emb_params: dict,
            encoder_params: dict,
            decoder_params: dict,
            speaker_mod_params: dict,
            prenet_params: dict,
            pred_params: dict,
            n_layers: int,
            device: str
            ) -> None:
        super().__init__()
        self.device = device
        self.enc_layers = nn.ModuleList([
            Encoder(**encoder_params)
            for _ in range(n_layers)
            ])
        self.dec_layers = nn.ModuleList([
            Decoder(**decoder_params)
            for _ in range(n_layers)
        ])
        self.dec_prenet = DecoderPrenet(
            **prenet_params
        )
        self.pos_emb = PositionalEmbedding(
            **pos_emb_params
            )
        self.speaker_mod = SpeakerModule(
            **speaker_mod_params
        )
        self.pred_mod = PredModule(
            **pred_params
        )

    def forward(
            self,
            x: Tensor,
            speakers: Tensor,
            y: Tensor,
            y_lengths: Tensor,
            diagonal_contraint: IDiagonalConstraint,
            ):
        enc_inp = self.pos_emb(x)
        speaker_emb = self.speaker_mod(speakers)
        prenet_out = self.dec_prenet(y)
        dec_input = cat_speaker_emb(speaker_emb, prenet_out)
        batch_size, max_len, d_model = dec_input.shape
        dec_input = dec_input + get_positionals(
            max_len, d_model
            ).to(self.device)
        center = torch.ones(batch_size)
        mel_results = None
        stop_results = None
        for i in range(1, max_len):
            dec_input_sliced = dec_input[:, :i, :]
            for enc_layer, dec_layer in zip(self.enc_layers, self.dec_layers):
                enc_inp = enc_layer(enc_inp)
                dec_input_sliced, att, temp_center = dec_layer(
                    x=dec_input_sliced,
                    encoder_values=cat_speaker_emb(speaker_emb, enc_inp),
                    center=center
                    )
                diagonal_contraint.add(att, y_lengths)
            center = temp_center
            mels, stop_props = self.pred_mod(dec_input_sliced[:, -1:, :])
            mel_results = mels if mel_results is None else \
                torch.cat([mel_results, mels], dim=1)
            stop_results = stop_props if stop_results is None else \
                torch.cat([stop_results, stop_props], dim=1)
        return mel_results, stop_results
