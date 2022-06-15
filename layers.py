import math
import torch
import torch.nn as nn
from typing import List, Tuple
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


class MHA(nn.Module):
    """Implements the multi-head attention module

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        p_dropout (float): The dropout ratio.
    """
    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        assert d_model % h == 0, 'd_model is not divisible by h'
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
        self.proj_fc = nn.Linear(
            in_features=2 * d_model,
            out_features=d_model,
        )
        self.dropout = nn.Dropout(p_dropout)
        self.d_model = d_model
        self.h = h
        self.dk = d_model // h
        self.sqrt_dk = math.sqrt(self.dk)
        self.softmax = nn.Softmax(dim=-1)

    def _get_scaled_att(
            self,
            Q: Tensor,
            K: Tensor
            ) -> Tensor:
        """Calculates the scaled attention map
        by calculating softmax(matmul(Q, K.T)/sqrt(dk))
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk]
            K (Tensor): The Key tensor of shape [h * B, dk, Tk]
        Returns:
            Tensor: The scaled attention weights of shape
            [B * h, Tq, Tk]
        """
        result = torch.matmul(Q, K)
        result = result / self.sqrt_dk
        return self.softmax(result)

    def perform_att(
            self,
            Q: Tensor,
            K: Tensor,
            V: Tensor
            ) -> Tensor:
        """Performs multi-head scaled attention
        by calculating softmax(matmul(Q, K.T)/sqrt(dk)).V
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk]
            K (Tensor): The Key tensor of shape [h * B, dk, Tk]
            V (Tensor): The Value tensor of shape [h * B, Tk, dk]
        Returns:
            Tuple[Tensor, Tensor]: The attention matrix of shape
            [B * h, Tq, Tk] and the scaled attention value of
            shape [B * h, Tq, dk].
        """
        att = self._get_scaled_att(Q, K)
        result = torch.matmul(att, V)
        return att, result

    def _reshape(self, *args) -> List[Tensor]:
        """Reshabes all the given list of tensor
        from [B, T, N] to [B, T, h, dk]
        Returns:
            List[Tensor]: list of all reshaped tensors
        """
        return [
            item.contiguous().view(-1, item.shape[1], self.h, self.dk)
            for item in args
        ]

    def _pre_permute(self, *args) -> List[Tensor]:
        """Permutes all the given list of tensors
        from [B, T, h, dk] to become [h, B, T, dk].

        Returns:
            List[Tensor]: List of all permuted tensors.
        """
        return [
            item.permute(2, 0, 1, 3)
            for item in args
        ]

    def _change_dim(self, *args) -> List[Tensor]:
        """Changes the dimensionality of all passed tensores
        from [B, T, N] to [B * h, T, dk]

        Returns:
            List[Tensor]: List of the modified tensors.
        """
        result = self._reshape(*args)  # [B, T, h, dk]
        result = self._pre_permute(*result)  # [h, B, T, dk]
        return [
            item.contiguous().view(-1, item.shape[2], item.shape[3])
            for item in result
        ]

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor
            ) -> Tuple[Tensor, Tensor]:
        """Performs multi-head attention on the provided key, query and value
        Args:
            key (Tensor): The key tensor of shape [B, Mt, d_model]
            query (Tensor): The query tensor of shape [B, Ms, d_model]
            value (Tensor): The value tensor of shape [B, Mt, d_model]
        Returns:
            Tuple[Tensor, Tensor]: A tuple of the attention matrix and the
            results after performing multi-head attention where the first of
            shape [h, B, Ms, Mt] and the second of shape [B, Tq, d_model].
        """
        [b, s, _] = query.shape
        K = self.fc_key(key)
        Q = self.fc_query(query)
        V = self.fc_value(value)
        (Q, K, V) = self._change_dim(Q, K, V)  # [h * B, T, dk]
        K = K.permute(0, 2, 1)  # [h, T, B, dk]
        att, result = self.perform_att(Q, K, V)
        result = result.view(self.h, b, s, self.dk)
        result = result.permute(1, 2, 0, 3)
        result = result.contiguous().view(b, s, -1)
        result = torch.cat([query, result], dim=-1)
        result = self.proj_fc(result)
        out = self.dropout(result)
        return att, out


class FeedForward(nn.Module):
    """Implements the feedforward Module in the model, where the input is
    scaled to a hidden_size and then back to the d_model.

    Args:
        d_model (int): The model dimensionality.
        hidden_size (int): the hidden size of the module.
        p_dropout (float): The dropout ratio.
    """
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


class AddAndNorm(nn.Module):
    """Implements the Add & Norm module where the input of the last module
    and the output of the last module added and then fed to Layernorm

    Args:
        d_model (int): The model dimensionality.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lnrom = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, out: Tensor):
        return self.lnrom(x + out)


class Encoder(nn.Module):
    """Implements the basic unit of the encoder and it contains the below:
        - multi-head self attention layer.
        - feed forward layer.
        - Residual add and layer normalization after each of the above.

    Args:
        d_model (int): The model dimensionality.
        dk (int): The size of each head.
        hidden_size (int): the hidden size of the feed forward module.
        p_dropout (float): The dropout ratio.
    """
    def __init__(
            self,
            d_model: int,
            dk: int,
            hidden_size: int,
            p_dopout: float
            ) -> None:
        super().__init__()
        self.mhsa = MHSA(
            d_model=d_model,
            dk=dk,
            p_dropout=p_dopout
            )
        self.mhsa_add_and_norm = AddAndNorm(
            d_model=d_model
            )
        self.ff = FeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            p_dropout=p_dopout
        )
        self.ff_add_and_norm = AddAndNorm(
            d_model=d_model
        )

    def forward(self, x: Tensor) -> Tensor:
        """Given the input of shape [B, M, d] performs self attention
        on the input and return back the result of shape [B, M, d]

        Args:
            x (Tensor): The input of shape [B, M, d]

        Returns:
            Tensor: The result out of the self attention of shape [B, M, d]
        """
        out = self.mhsa(x)
        out = self.mhsa_add_and_norm(x, out)
        ff_out = self.ff(out)
        out = self.ff_add_and_norm(out, ff_out)
        return out


class PositionalEmbedding(nn.Module):
    """Implemnts the Positional Embedding of the encoder.

    Args:
        d_model (int): The model dimensionality.
        vocab_size (int): The vocabulary size to be used in the
        embedding layer.
        pad_idx (int): The padding index.
        device (str): The device to map the tensors to.
        add_lnorm (int): A flag to either use a leyer norm or not.
    """
    def __init__(
            self,
            d_model: int,
            vocab_size: int,
            pad_idx: int,
            device: str,
            add_lnorm: bool
            ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
            )
        self.device = device
        self.add_lnorm = add_lnorm
        if add_lnorm is True:
            self.lnorm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Given the sequences input of shape [B, M]
        returns the positional embedding for each sequence.

        Args:
            x (Tensor): The input sequence of shape [B, M]

        Returns:
            Tensor: The positional embedding of shape [B, M, d]
        """
        max_length = x.shape[1]
        out = self.embedding(x)
        if self.add_lnorm is True:
            out = self.lnorm(out)
        pos = get_positionals(max_length, self.d_model).to(self.device)
        out = pos + out
        return out


class DecoderPrenet(nn.Module):
    def __init__(
            self,
            inp_size: int,
            bottleneck_size: int,
            d_model: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=inp_size,
            out_features=bottleneck_size
            )
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=bottleneck_size,
            out_features=bottleneck_size
            )
        self.fc3 = nn.Linear(
            in_features=bottleneck_size,
            out_features=d_model
            )

    def forward(self, x: Tensor):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out
