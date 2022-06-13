import math
import torch
import torch.nn as nn
from typing import List
from torch import Tensor
from utils import get_positionals


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


class MHSA(nn.Module):
    """Implements the multi-head self attention module

    Args:
        d_model (int): The model dimensionality.
        dk (int): The size of each head.
        p_dropout (float): The dropout ratio.
        device (str): The device to map the tensors to.
    """
    def __init__(
            self,
            d_model: int,
            dk: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        assert d_model % dk == 0, 'd_model is not divisible by dk'
        self.d_model = d_model
        self.fc_key = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.fc_query = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.fc_value = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.lnorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)
        self.d_model = d_model
        self.dk = dk
        self.sqrt_dk = math.sqrt(dk)
        self.h = d_model // dk
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def _key_query_matmul(self, Q: Tensor, K: Tensor) -> Tensor:
        """Performs the Matmul operation in
        scaled Dot-Product Attention
        Args:
            Q (Tensor): The Query tensor of shape [B, M, dk, h]
            K (Tensor): The Key tensor of shape [B, M, dk, h]
        Returns:
            Tensor: The result of matmul operation of shape
            [B, M, dk, dk]
        """
        return torch.matmul(Q, K.permute(0, 1, 3, 2))

    def _get_scaled_att(
            self,
            Q: Tensor,
            K: Tensor
            ) -> Tensor:
        """Calculates the scaled attention map
        by calculating softmax(matmul(Q, K.T)/sqrt(dk))
        Args:
            Q (Tensor): The Query tensor of shape [B, M, dk, h]
            K (Tensor): The Key tensor of shape [B, M, dk, h]
        Returns:
            Tensor: The scaled attention map of shape
            [B, M, dk, dk]
        """
        result = self._key_query_matmul(Q, K)
        result = result / self.sqrt_dk
        return self.softmax(result)

    def perform_self_att(
            self,
            Q: Tensor,
            K: Tensor,
            V: Tensor
            ) -> Tensor:
        """Perform multi head scaled attention
        by calculating softmax(matmul(Q, K.T)/sqrt(dk)).V
        Args:
            Q (Tensor): The Query tensor of shape [B, M, dk, h]
            K (Tensor): The Key tensor of shape [B, M, dk, h]
            V (Tensor): The Value tensor of shape [B, M, dk, h]
        Returns:
            Tensor: The scaled attention map of shape
            [B, M, dk * h]
        """
        (b, m, *_) = Q.shape
        att = self._get_scaled_att(Q, K)
        result = torch.matmul(att, V)
        return result.view(b, m, -1)

    def _reshape(self, *args) -> List[Tensor]:
        """Reshabes all the given list of tensor
        from [B, M, N] to [B, M, dk, h]
        Returns:
            List[Tensor]: list of all reshaped tensors
        """
        return [
            item.view(-1, item.shape[1], self.dk, self.h)
            for item in args
        ]

    def forward(self, inp: Tensor) -> Tensor:
        """Passes the input into multi-head attention
        Args:
            inp (Tensor): The input tensor
        Returns:
            Tensor: The result after adding it to positionals
            and passing it through multi-head self-attention
        """
        out = self.lnorm(inp)
        K = self.fc_key(out)
        Q = self.fc_query(out)
        V = self.fc_value(out)
        max_length = inp.shape[1]
        positionals = get_positionals(max_length, self.d_model).to(self.device)
        out = out + positionals
        (Q, K, V) = self._reshape(Q, K, V)
        out = self.perform_self_att(Q, K, V)
        out = self.dropout(out)
        return inp + out


class FeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            hidden_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=d_model,
            out_features=hidden_size
        )
        self.fc2 = nn.Linear(
            in_features=hidden_size,
            out_features=d_model
        )
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.dropout(out)
        return out
