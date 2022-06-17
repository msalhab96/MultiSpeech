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


class MultiHeadAtt(nn.Module):
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
        dk (int): The number of heads.
        hidden_size (int): the hidden size of the feed forward module.
        p_dropout (float): The dropout ratio.
    """
    def __init__(
            self,
            d_model: int,
            h: int,
            hidden_size: int,
            p_dopout: float
            ) -> None:
        super().__init__()
        self.mhsa = MultiHeadAtt(
            d_model=d_model,
            h=h,
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
        out = self.mhsa(x, x, x)
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
    # TODO: add docstring
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


class MultiHeadSlidingAtt(nn.Module):
    """Implements The Multi-head with attention sliding window.

    Args:
        left_shift (int): The window size below the center.
        right_shift (int): The window size beyond the center.
        max_steps (int): The maximum step allowed for the window to take.
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        p_dropout (float): The dropout ratio.
    """
    def __init__(
            self,
            left_shift: int,
            right_shift: int,
            max_steps: int,
            d_model: int,
            h: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.left_shift = left_shift
        self.right_shift = right_shift
        self.window_size = self.right_shift - self.left_shift + 1
        self.max_steps = max_steps
        self.mha = MultiHeadAtt(
            d_model=d_model, h=h, p_dropout=p_dropout
            )

    def _get_range_matrix(self, center: Tensor) -> Tensor:
        """Given the cneter vector, returns the range matrix that has
        all the possible indices in the attention window.

        Args:
            center (Tensor): The center Tensor of shape [B,]

        Returns:
            Tensor: The range matrix of shape [B, win_size]
        """
        length = center.shape[0]
        range_matrix = torch.zeros(
            length, self.window_size, dtype=torch.int64
            )
        range_matrix[:, 0] = torch.max(
            center + self.left_shift, range_matrix[:, 0]
            )
        for i in range(self.window_size - 1):
            range_matrix[:, i + 1] = range_matrix[:, i] + 1
        return range_matrix

    def _govern_max_len(self, max_size: int, indices: Tensor) -> Tensor:
        batch_size = indices.shape[0]
        return torch.min(
            indices,
            torch.ones(batch_size, dtype=torch.int64) * max_size
            )

    def _govern_center(self, center: Tensor, updated_center: Tensor) -> Tensor:
        """Controls how fast the center moves, as mentioned in the paper
        we clip any movement greater than the max_steps.

        Args:
            center (Tensor): The center vector of shape [B, ]
            updated_center (Tensor): The new calculated center of shape [B, ]

        Returns:
            Tensor: The new updated center of shape [B, ]
        """
        mask = (updated_center - center) >= self.max_steps
        return ~mask * updated_center + mask * (center + 1)

    def _slice_range_matrix(self, range_matrix: Tensor, idx: int) -> Tensor:
        return range_matrix[:, idx]

    def _slice_from_values(self, indices: Tensor, values: Tensor) -> Tensor:
        [batch_size, _, d_model] = values.shape
        indices = torch.unsqueeze(indices, dim=1)
        indices = indices.repeat(1, d_model).view(batch_size, 1, d_model)
        return torch.gather(values, 1, indices)

    def _get_values(
            self,
            values: Tensor,
            range_matrix: Tensor,
            ) -> Tensor:
        """Given the values matrix (from the encoder) and the range matrix
        returns all the targeted indices in the range matrix.

        Args:
            values (Tensor): The values matrix of shape [B, Tt, d_model]
            range_matrix (Tensor): The range matrix of shape [B, win_size]

        Returns:
            Tensor: The sliced values of shape [B, win_size, d_model]
        """
        [batch_size, max_length, d_model] = values.shape
        slice = torch.zeros(
            batch_size, self.window_size, d_model
            )
        for i in range(self.window_size):
            indices = self._slice_range_matrix(range_matrix, i)
            indices = self._govern_max_len(max_length - 1, indices)
            slice[0:, i:i+1, 0:] = self._slice_from_values(indices, values)
        return slice

    def _update_center(self, range_matrix: Tensor, att: Tensor) -> Tensor:
        """Given the range matrix and the att matrix, updates the center
        vector by calculating floor(range * att) as mentioned in the paper.

        Args:
            range_matrix (Tensor): The range matrix of shape [B, window_size].
            att (Tensor): The resulted attention matrix of shape
            [B * h, Ts, window_size].

        Returns:
            Tensor: The updated center vector.
        """
        att = att[:, -1, :]  # The last one of Ts, [B, 1, win_size]
        print(att.shape)
        [bh, ws] = att.shape
        att = att.view(self.mha.h, bh // self.mha.h, ws)  # [h, B, win_size]
        att = att.sum(dim=0)  # [B, win_size]
        new_center = range_matrix * att
        new_center = torch.sum(new_center, dim=-1)  # [B,]
        new_center = new_center / self.mha.h
        new_center = torch.floor(new_center).to(torch.int)
        print(new_center)
        return new_center

    def forward(
            self, query: Tensor, values: Tensor, center: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs sliding attention.

        Args:
            query (Tensor): The query tensor of shape [B, Tq, d_model]
            values (Tensor): The values tensor of shape [B, Tk, d_model]
            center (Tensor): The latest center values of shape [B,]

        Returns:
            Tuple[Tensor, Tensor]: The attention wieghts The result of the
            attention and the updated center.
        """
        range_matrix = self._get_range_matrix(center)
        win_vals = self._get_values(values, range_matrix)
        att, out = self.mha(key=win_vals, query=query, value=win_vals)
        print(att.shape)
        center = self._update_center(range_matrix, att)
        return att, out, center


class Decoder(nn.Module):
    """Implements the basic unit of the decoder

    Args:
        d_model (int): The model dimensionality.
        h (int): The number of heads.
        p_dropout (float): The dropout ratio.
        left_shift (int): The window size below the center for the slided MHA.
        right_shift (int): The window size beyond the center for the
        slided MHA.
        max_steps (int): The maximum step allowed for the window to take for
        the slided MHA.
        hidden_size (int): the hidden size of the feed forward module.
    """
    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            left_shift: int,
            right_shift: int,
            max_steps: int,
            hidden_size: int
            ) -> None:
        super().__init__()
        # TODO: Add Masking here
        self.mhsa = MultiHeadAtt(
            d_model=d_model,
            h=h,
            p_dropout=p_dropout
        )
        self.add_and_norm_1 = AddAndNorm(d_model=d_model)
        self.slided_mha = MultiHeadSlidingAtt(
            left_shift=left_shift,
            right_shift=right_shift,
            max_steps=max_steps,
            d_model=d_model,
            h=h,
            p_dropout=p_dropout
        )
        self.add_and_norm_2 = AddAndNorm(d_model=d_model)
        self.ff = FeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            p_dropout=p_dropout
        )
        self.add_and_norm_3 = AddAndNorm(d_model=d_model)

    def forward(
            self,
            x: Tensor,
            encoder_values: Tensor,
            center: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pass the data into the decoder blocks which they are:
        - MMHA
        - ADD & NORM
        - MHA
        - ADD & NORM
        - Feed Forward
        - ADD & NORM

        Args:
            x (Tensor): The input tensor of shape [B, Td, d_model]
            encoder_values (Tensor): The encoder results of shape
            [B, Te, d_model]
            center (Tensor): The center vectors for the slided window attention

        Returns:
            Tuple[Tensor, Tensor, Tensor]: a tuple of the results, the
            attention weights and the center vector.
        """
        out = self.mhsa(x, x, x)
        out_1 = self.add_and_norm_1(x, out)
        att, out, center = self.slided_mha(
            query=out_1, values=encoder_values, center=center
        )
        out = self.add_and_norm_2(out_1, out)
        out_1 = self.ff(out)
        out = self.add_and_norm_3(out_1, out)
        return out, att, center


class PredModule(nn.Module):
    """Impelements the prediction module where it contains the Mel Linear and
    the Stop Linear layers.

    Args:
        d_model (int): The model dimensionality.
        n_mels (int): Number of mel filterbanks to be predicted.
    """
    def __init__(self, d_model: int, n_mels: int) -> None:
        super().__init__()
        self.mel_linear = nn.Linear(d_model, n_mels)
        self.stop_linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes the last decoder output through mel linear and stop linear

        Args:
            x (Tensor): The last decoder layer's output of shape
            [B, Ts, d_model].

        Returns:
            Tuple[Tensor, Tensor]: A tuple of the predicted mels and the stop
            prediction.
        """
        mels = self.mel_linear(x)
        stop_props = self.stop_linear(x)
        stop_props = self.sigmoid(stop_props)
        return mels, stop_props
