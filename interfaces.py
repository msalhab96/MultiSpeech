from abc import ABC, abstractmethod


class IDiagonalConstraint(ABC):

    @abstractmethod
    def reset():
        """To reset the total value
        """
    @abstractmethod
    def add():
        """Calculates the attention and add it to the total value.
        """
