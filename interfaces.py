from abc import ABC, abstractmethod, abstractproperty


class ITokenizer(ABC):

    @abstractmethod
    def ids2tokens(self):
        pass

    @abstractmethod
    def tokens2ids(self):
        pass

    @abstractmethod
    def set_tokenizer(self):
        pass

    @abstractmethod
    def save_tokenizer(self):
        pass

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def add_token(self):
        pass

    @abstractmethod
    def preprocess_tokens(self):
        pass

    @abstractmethod
    def batch_tokenizer(self):
        pass

    @abstractproperty
    def vocab_size(self):
        pass

    @abstractmethod
    def get_tokens(self):
        pass


class IDataLoader(ABC):

    @abstractmethod
    def load(self):
        pass


class IPipeline(ABC):

    @abstractmethod
    def run():
        pass


class IPadder(ABC):

    @abstractmethod
    def pad():
        pass
