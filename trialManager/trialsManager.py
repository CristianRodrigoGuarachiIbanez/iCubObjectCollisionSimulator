from zipfile import ZipFile
import logging
from typing import Iterable, Iterator, List, Tuple, TypeVar, Any, Union
from io import StringIO
from csv import reader, register_dialect, QUOTE_ALL
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from dataCollector import DataCollector
from dataConversionTool import DataConversionTool
register_dialect("ownDialect", delimiter=",", skipinitialspace=True, quoting=QUOTE_ALL);
from imgArrayRecoverer import ImgArrayRecoverer
from numpy import ndarray

class TrialManager:
    listOfFileEndings: List[str] = [".csv", ".h5", "png"]

    def __init__(self, zipClass: ZipFile, sortedFileNames: List[List[str]]) -> None:
        self.__zipClass: ZipFile = zipClass;
        self.__fileNames: List[List[str]] = sortedFileNames;

        self.__collector: DataCollector = DataCollector();

    def copyCSVData(self, fileName: str, readLines: DataFrame) -> None:

        assert (len(readLines) > 0), "read files hat none extracted data, it es probably empty";

        if(fileName=="arm_left.csv"):
            self.__collector.setArmLeft(readLines);
        elif(fileName=="arm_right.csv"):
            self.__collector.setArmRight(readLines);
        elif(fileName =="forearm_left.csv"):
            self.__collector.setForeArmLeft(readLines);
        elif(fileName=="farearm_right.csv"):
            self.__collector.setForearmRight(readLines);
        elif(fileName=="hand_coordinates.csv"):
            self.__collector.setHeadCoordData(readLines);
        elif(fileName=="hand_left.csv"):
            self.__collector.setHandLeftData(readLines);
        elif(fileName=="hand_right.csv"):
            self.__collector.setHandRightData(readLines);
        elif(fileName=="head_coordinates.csv"):
            self.__collector.setHeadCoordData(readLines);
        elif(fileName=="joints_coordinates.csv"):
            self.__collector.setJointCoordData(readLines);
        elif(fileName=="object_data.csv"):
            self.__collector.setObjectCoordData(readLines);


    @staticmethod
    def __readDataCSV(file) -> DataFrame:

        df: DataFrame = None;
        try:
            df = read_csv(file, delim_whitespace=True)
        except EmptyDataError as e:
            print(e)
            df = DataFrame()

        return df

    def generateDataFrame(self, CSVfileName: str) -> DataFrame:
        """
         gets into every file and recovers the data according the specifications of the ground truth elements
         @:param fileName: string name of the file
         @:param zipClass: ZipFile data containing the zip file
         @:return: a data frame
        """

        with self.__zipClass.open(CSVfileName, "r") as lines:
            print(read_csv(lines))
            return self.__readDataCSV(lines)

    def generateLenRef(self) -> List[List[Any]]:

        groundTruth: List[List[Any]] = list();
        i: int = 0;
        for categories in self.__fileNames:
            for fileNamesWithExtension in categories:
                if(fileNamesWithExtension.endswith(".csv") and fileNamesWithExtension.startswith("gt_")):
                    trials, length = self.__getLenEveryTrials(fileNamesWithExtension, self.__zipClass) #type: List[str], List[int];
                    groundTruth.append([i ,trials, length]);
                    i += 1;
        return groundTruth;

    @staticmethod
    def __getLenEveryTrials(fileName: str, zipClass: ZipFile) -> Tuple[List[str], List[int]]:
        gtTrials: List[str] = [];
        lenTrials: List[int] =[];
        with zipClass.open(fileName, "r") as file:
            readLines: DataFrame = read_csv(file, delimiter="\n")
            print(readLines.values.tolist())
            readLines: List[str] = [col.split(",") for row in readLines.values.tolist() for col in row]
            print(readLines)
            #badChars: List[chr] = ['"[', '[', ']', ']"', '"']
            for trial in readLines:
                gtTrials.append(trial[-1])
                lenTrials.append(len(trial[2:len(trial)-1]))
                #print(len(gtProFrames))
                #gtProFrames: List[str] = [items.replace(i, "") for items in trial[2:len(trial)-1] for i in items if i in badChars ]
        print(gtTrials, lenTrials)
        return gtTrials, lenTrials



if __name__ == "__main__":
    # path: str = "iCubObjectCollisionSimulator/DGP_iCubSimulator/trials/ra_cube_trial6.zip"
    # file: str = "gt_right_hand.csv"
    # zip: TrialManager = TrialManager()
    # zip.openZipFile(path, file)
    pass




