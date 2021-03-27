from typing import List, Tuple
from pandas import DataFrame

class DataConversionTool:

    def __init__(self, positiveClass: List[List[float]], negativeClass: List[List[float]]) -> None:

        self.__positiveClass: DataFrame = self.__convertToDataFrame(positiveClass);
        self.__negativeClass: DataFrame = self.__convertToDataFrame(negativeClass);

    def getPositiveClass(self) -> DataFrame:
        return self.__positiveClass;

    def getNegativeClass(self) -> DataFrame:
        return self.__negativeClass;

    def __convertToDataFrame(self, data: List[List]) -> DataFrame:
        return DataFrame(data);

