
from abc import ABC, abstractmethod
class BuilderInterface(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """
    @property
    @abstractmethod
    def csvProduct(self) -> None:
        pass

    @property
    @abstractmethod
    def imgArrayProduct(self) -> None:
        pass

    @abstractmethod
    def produceDataFrame(self, zipFileName: str, flag: str) -> None:
        pass

    @abstractmethod
    def produceImgArray(self, zipfileName: str, flag: str) -> None:
        pass