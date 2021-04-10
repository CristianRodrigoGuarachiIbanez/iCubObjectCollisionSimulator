from numpy import ndarray, asarray, array
from typing import List
class ImgArrayProduct:
    def __init__(self) -> None:
        self.__binocularImgArray: List[ndarray] = list();
        self.__sceneImgArray: List[ndarray] = list();

    def getBinocularImgArray(self) -> List[ndarray]:
        return self.__binocularImgArray;

    def setBinoImgArray(self, imgArray: ndarray) -> None:
        assert(imgArray.size > 0), "binocular img array is equal to 0"
        self.__binocularImgArray.append(imgArray);

    def getSceneImgArray(self) -> List[ndarray]:
        return self.__sceneImgArray;

    def setSceneImgArray(self, imgArray: ndarray) -> None:
        assert(imgArray.size>0), 'scene img array is equal to 0'
        self.__sceneImgArray.append(imgArray);






