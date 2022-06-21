from interfaces import IDataLoader
from typing import Union
from pathlib import Path
import json


class JSONLoader(IDataLoader):
    def __init__(self, file_path: Union[str, Path]) -> None:
        super().__init__()
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data


class TextLoader(IDataLoader):
    def __init__(self, file_path: Union[str, Path]) -> None:
        super().__init__()
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r') as f:
            data = f.read()
        return data
