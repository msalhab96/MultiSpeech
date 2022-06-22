from interfaces import ITrainer, IDataLoader
from torch.nn import Module
from pathlib import Path
from typing import Union
from torch import Tensor


class Trainer(ITrainer):
    def __init__(
            self,
            train_loader: IDataLoader,
            test_loader: IDataLoader,
            model: Module,
            criterion: Module,
            optimizer: object,
            save_dir: Union[str, Path],
            steps_per_ckpt: int,
            epochs: int,
            last_step: int,
            device: str
            ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.steps_per_ckpt = steps_per_ckpt
        self.last_step = last_step

    def set_train_mode(self):
        self.model = self.model.train()

    def set_test_mode(self):
        self.model = self.model.test()

    def _predict(self, text: Tensor, spk: Tensor, speech: Tensor):
        text = text.to(self.device)
        spk = spk.to(self.device)
        speech = speech.to(self.device)
        return self.model(
            text, spk, speech
        )

    def _train_step(
            self,
            speech: Tensor,
            speech_length: Tensor,
            mask: Tensor,
            text: Tensor,
            spk: Tensor
            ):
        print(text)
        mel_results, stop_results, alignments = self._predict(
            text, spk, speech
        )
        self.optimizer.zero_grad()
        loss = self.criterion(
            lengths=speech_length,
            mask=mask,
            stop_pred=stop_results,
            mels_pred=mel_results,
            mels_target=speech,
            alignments=alignments
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        total_train_loss = 0
        for item in self.train_loader:
            loss = self._train_step(*item)
            total_train_loss += loss
        return total_train_loss / len(self.train_loader)

    def test(self):
        total_test_loss = 0
        for (speech, speech_length, mask, text, spk) in self.test_loader:
            mel_results, stop_results, alignments = self._predict(
                text, spk, speech
            )
            total_test_loss += self.criterion(
                lengths=speech_length,
                mask=mask,
                stop_pred=stop_results,
                mels_pred=mel_results,
                mels_target=speech,
                alignments=alignments
                ).item()
        return total_test_loss / len(self.test_loader)

    def fit(self):
        for epoch in range(self.epochs):
            train_loss = self.train()
            test_loss = self.test()
