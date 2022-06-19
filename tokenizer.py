from __future__ import annotations
from dataclasses import dataclass
from typing import (
    List,
    Tuple,
    Union
    )
from data_loaders import JSONLoader
from decorators import check_token
from interfaces import ITokenizer
from utils import save_json
from pathlib import Path


PAD = '<PAD>'
EOS = '<EOS>'


@dataclass
class SpecialTokens:
    _pad: Tuple[str, int] = (None, None)
    _eos: Tuple[str, int] = (None, None)

    @property
    def pad_id(self):
        return self._pad[1]

    @property
    def pad_token(self):
        return self._pad[0]

    @property
    def eos_id(self):
        return self._eos[1]

    @property
    def eos_token(self):
        return self._eos[0]


class BaseTokenizer(ITokenizer):
    _pad_key = 'pad'
    _eos_key = 'eos'
    _token_to_id_key = 'token_to_id'
    _special_tokens_key = 'special_tokens'

    def __init__(self) -> None:
        super().__init__()
        self._token_to_id = dict()
        self._id_to_token = dict()
        self.special_tokens = SpecialTokens()

    @property
    def vocab_size(self):
        return len(self._token_to_id)

    def add_token(self, token: str):
        token_id = self.vocab_size
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        return token_id

    @check_token(PAD)
    def add_pad_token(self, token=PAD) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._pad = (token, token_id)
        return self

    @check_token(EOS)
    def add_eos_token(self, token=EOS) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._eos = (token, token_id)
        return self

    def _reset_id_to_token(self) -> None:
        self._id_to_token = dict(zip(
            self._token_to_id.values(),
            self._token_to_id.keys()
            ))

    def __set_special_tokens_dict(self, data: dict) -> None:
        if self._pad_key in data:
            self.special_tokens._pad = tuple(data[self._pad_key])
        if self._eos_key in data:
            self.special_tokens._eos = tuple(data[self._eos_key])

    def get_special_tokens_dict(self) -> dict:
        data = {}
        if self.special_tokens.pad_id is not None:
            data[self._pad_key] = list(self.special_tokens._pad)
        if self.special_tokens.eos_id is not None:
            data[self._eos_key] = list(self.special_tokens._eos)
        return data

    def load_tokenizer(
            self,
            tokenizer_path: Union[str, Path],
            *args,
            **kwargs
            ) -> ITokenizer:
        data = JSONLoader(tokenizer_path).load()
        self._token_to_id = data[self._token_to_id_key]
        self.__set_special_tokens_dict(data[self._special_tokens_key])
        self._reset_id_to_token()
        return self

    def set_tokenizer(self, data: List[str], *args, **kwargs) -> ITokenizer:
        all_tokens = self.get_tokens(data)
        _ = list(map(self.add_token, all_tokens))
        self._reset_id_to_token()
        return self

    def save_tokenizer(
            self,
            save_path: Union[str, Path],
            *args,
            **kwargs
            ) -> None:
        data = {
            self._token_to_id_key: self._token_to_id,
            self._special_tokens_key: self.get_special_tokens_dict()
        }
        save_json(save_path, data)

    def ids2tokens(self, ids: List[str]) -> List[str]:
        return list(map(lambda x: self._id_to_token[x], ids))

    def tokens2ids(self, sentence: str) -> List[int]:
        sentence = self.preprocess_tokens(sentence)
        return list(map(
            lambda x: self._token_to_id.get(x, self.special_tokens.pad_id),
            sentence)
            )

    def batch_tokenizer(self, data: List[str]) -> list:
        return list(map(self.tokens2ids, data))

    def batch_detokenizer(self, data: List[int]) -> list:
        return list(map(self.ids2tokens, data))


class CharTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()

    def get_tokens(self, data: List[str]):
        return set(''.join(data))

    def preprocess_tokens(self, sentence: str) -> List[str]:
        return list(sentence)
