
from typing import List, Tuple, Any, Dict, TypeVar
class GroundTruthCollector:

    def __init__(self) -> None:
        self.__gtLeftHand: List[Tuple[List]] = list();
        self.__gtRightHand: List[Tuple[Any]] = list();
        self.__gtLeftArm: List[Tuple[List]] = list();
        self.__gtRightArm: List[Tuple[List]] = list();
        self.__gtLeftForeArm: List[Tuple[List]] = list();
        self.__gtRightForeArm: List[Tuple[List]] = list();

    def setLeftArm(self, inputs: Tuple[int, List[str], List[int]]) -> None:
        self.__gtLeftArm.append(inputs);
    def setRightArm(self, inputs:  Tuple[int, List[str], List[int]]) -> None:
        self.__gtRightArm.append(inputs);
    def setLeftHand(self, inputs:  Tuple[int, List[str], List[int]]) -> None:
        self.__gtLeftHand.append(inputs);
    def setRightHand(self, inputs:  Tuple[int, List[str], List[int]]) -> None:
        self.__gtRightHand.append(inputs);
    def setLeftForeArm(self, inputs: Tuple[int, List[str], List[int]]) -> None:
        self.__gtLeftForeArm.append(inputs);
    def setRightForeArm(self, inputs:  Tuple[int, List[str], List[int]]) -> None:
        self.__gtRightForeArm.append(inputs);

    def getLeftArm(self) -> List[Tuple[List]]:
        return self.__gtLeftArm;
    def getRightArm(self) -> List[Tuple[List]]:
        return self.__gtRightArm;
    def getLeftForeArm(self) -> List[Tuple[List]]:
        return self.__gtLeftForeArm;
    def getRightForeArm(self) -> List[Tuple[List]]:
        return self.__gtRightForeArm;
    def getLeftHand(self) -> List[Tuple[List]]:
        return self.__gtLeftHand;
    def getRightHand(self) -> List[Tuple[List]]:
        return self.__gtRightHand;