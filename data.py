import torch
from typing import Union
from torch.utils.data import Dataset
from interfaces import IDataLoader, IPipeline, IPadder


SPK_ID = 0
PATH_ID = 1
TEXT_ID = 2
DURATION_ID = 3


class Data(Dataset):
    def __init__(
            self,
            data_loader: IDataLoader,
            aud_pipeline: IPipeline,
            text_pipeline: IPipeline,
            aud_padder: IPadder,
            text_padder: IPadder,
            batch_size: int,
            sep: str
            ) -> None:
        super().__init__()
        self.sep = sep
        self.aud_pipeline = aud_pipeline
        self.text_pipeline = text_pipeline
        self.aud_padder = aud_padder
        self.text_padder = text_padder
        self.batch_size = batch_size
        self.data = self.process(data_loader)
        self.max_speech_lens = []
        self.max_text_lens = []
        self.n_batches = len(self.data) // self.batch_size
        if len(self.data) % batch_size > 0:
            self.n_batches += 1
        self.__set_max_text_lens()

    def process(self, data_loader: IDataLoader):
        data = data_loader.load().split('\n')
        data = [item.split(self.sep) for item in data]
        data = sorted(data, key=lambda x: x[DURATION_ID], reverse=True)
        return data

    def __set_max_text_lens(self):
        for i, item in enumerate(self.data):
            idx = i // self.batch_size
            length = len(item[TEXT_ID])
            if idx >= len(self.max_text_lens):
                self.max_text_lens.append(length)
            else:
                self.max_text_lens[idx] = max(length, self.max_text_lens[idx])

    def __len__(self) -> int:
        return len(self.data)

    def _get_max_len(self, idx: int) -> Union[None, int]:
        bucket_id = idx // self.batch_size
        if bucket_id >= len(self.max_speech_lens):
            return None, self.max_text_lens[bucket_id] + 1
        return (
            self.max_speech_lens[bucket_id],
            self.max_text_lens[bucket_id] + 1
            )

    def __getitem__(self, idx: int):
        [spk_id, file_path, text, _] = self.data[idx]
        spk_id = int(spk_id)
        max_speech_len, max_text_len = self._get_max_len(idx)
        text = self.text_pipeline.run(text)
        text = self.text_padder.pad(text, max_text_len)
        speech = self.aud_pipeline.run(file_path)
        speech_length = speech.shape[0]
        mask = [True] * speech_length
        if max_speech_len is not None:
            mask.extend([False] * (max_speech_len - speech_length))
            speech = self.aud_padder.pad(speech, max_speech_len)
        else:
            self.max_speech_lens.append(speech_length)
        mask = torch.BoolTensor(mask)
        spk_id = torch.LongTensor([spk_id])
        return speech, speech_length, mask, text, spk_id
