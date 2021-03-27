from typing import List, Tuple, Any, TypeVar
from pandas import DataFrame

class DataCollector:

    def __init__(self) -> None:
        self.__armLeft: List[DataFrame] = list();
        self.__armRight: List[DataFrame] = list();
        self.__forearmLeft: List[DataFrame] = list();
        self.__forearmRight: List[DataFrame] = list();
        self.__handLeft: List[DataFrame] = list();
        self.__handRight: List[DataFrame] = list();
        self.__handCoord: List[DataFrame] = list();
        self.__headCoord: List[DataFrame] = list();
        self.__jointCoord: List[DataFrame] = list();
        self.__objectCoord: List[DataFrame] = list();


    def setArmLeft(self, newArmLeftData: DataFrame) -> None:
        self.__armLeft.append(newArmLeftData);
    def setArmRight(self, newArmRightData: DataFrame) -> None:
        self.__armRight.append(newArmRightData);
    def setForeArmLeft(self, newForearmLeftData: DataFrame) -> None:
        self.__forearmLeft.append(newForearmLeftData);
    def setForearmRight(self, newForearmRightData: DataFrame) -> None:
        self.__forearmRight.append(newForearmRightData);
    def setHandLeftData(self, newHandLeftData: DataFrame) -> None:
        self.__handLeft.append(newHandLeftData);
    def setHandRightData(self, newHandRightData: DataFrame) -> None:
        self.__handRight.append(newHandRightData);
    def setHeadCoordData(self, newHeadCoordData: DataFrame) -> None:
        self.__headCoord. append(newHeadCoordData);
    def setJointCoordData(self, newJointCoordData: DataFrame) -> None:
        self.__jointCoord.append(newJointCoordData);
    def setObjectCoordData(self, newObjectCoordData: DataFrame) -> None:
        self.__objectCoord.append(newObjectCoordData);

    def getArmLeft(self) -> List[DataFrame]:
        return self.__armLeft;

    def getArmRight(self) -> List[DataFrame]:
        return self.__armRight;

    def getForeArmLeft(self) -> List[DataFrame]:
        return self.__forearmLeft;

    def getForearmRight(self) -> List[DataFrame]:
        return self.__forearmRight;

    def getHandLeftData(self) -> List[DataFrame]:
        return self.__handLeft;

    def getHandRightData(self) -> List[DataFrame]:
        return self.__handRight;

    def getHeadCoordData(self) -> List[DataFrame]:
        return self.__headCoord;

    def getJointCoordData(self) -> List[DataFrame]:
        return self.__jointCoord;

    def getObjectCoordData(self,) -> List[DataFrame]:
        return self.__objectCoord;
