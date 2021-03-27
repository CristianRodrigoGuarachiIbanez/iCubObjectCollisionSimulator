from zipfile import ZipFile
from typing import List, Tuple
from abc import ABC, abstractmethod
class BuilderInterface(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """
    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def produceRef(self) -> None:
        pass

    @abstractmethod
    def produceDataFrame(self, zipFileName: str) -> None:
        pass

    # @abstractmethod
    # def produce_part_c(self) -> None:
    #     pass