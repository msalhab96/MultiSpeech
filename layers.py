import torch.nn as nn
from torch import Tensor


class SpeakerModule(nn.Module):
    """Implements the Speaker Module in the model archticture

    Args:
        n_speakers (int): The number of speakers.
        emb_size (int): The embedding dimensionality of the embedding layer.
        fc_size (int): The fully connected layer size..
    """
    def __init__(self, n_speakers: int, emb_size: int, fc_size: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=n_speakers,
            embedding_dim=emb_size
        )
        self.fc = nn.Linear(
            in_features=emb_size,
            out_features=fc_size
        )
        self.soft_sign = nn.Softsign()

    def forward(self, x: Tensor) -> Tensor:
        """Given x of shape [B, S] where S is a valid speaker id return
        the embedding for each speaker.

        Args:
            x (Tensor): The input to the embedding layer.

        Returns:
            Tensor: The speakers' Embedding of shape [B, S, E].
        """
        out = self.emb(x)
        out = self.fc(out)
        out = self.soft_sign(out)
        return out
