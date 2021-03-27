
from zipfile import ZipFile
import logging
from typing import Iterable, Iterator, List, Tuple, TypeVar, Any, Union
from io import StringIO
from csv import reader, register_dialect, QUOTE_ALL
from pandas import read_csv, DataFrame
from filesExtensionClassifier import FilesExtensionClassifier
from dataCollector import DataCollector
from dataConversionTool import DataConversionTool
register_dialect("ownDialect", delimiter=",", skipinitialspace=True, quoting=QUOTE_ALL);
from imgArrayRecoverer import ImgArrayRecoverer
from numpy import ndarray

class TrialManager:
    listOfFileEndings: List[str] = [".csv", ".h5", "png"]

    def __init__(self) -> None:
        self.__readLines: DataFrame = None;
        self.__collector: DataCollector = DataCollector();
        self.imgArrays: ndarray = None;
        self.__imgCollector: ImgArrayRecoverer = ImgArrayRecoverer();



    def openZipFile(self, zipName: str, fileName: str) -> None:

        fileClassifier: FilesExtensionClassifier = FilesExtensionClassifier()
        with ZipFile(zipName, "r") as zipClass:
            print("succefully imported")
            #zip.printdir()
            zipFiles: List[str] = zipClass.namelist(); # names of the files in the Zip
            sortedZipFiles: List[List[str]] = fileClassifier.sortFilesAccordingToExtentions(zipFiles, self.listOfFileEndings); # zip file names sorting according to extension
            print(sortedZipFiles);
            gtTrials, lenTrials = self.__getGroundTruthParametersFromCSV(fileName, zipClass) #type: List[str], List[int]; # extract the number ground truths and the length for every trials
            #  ITERATE OVER EVERY SINGLE ELEMENT OF THE SORTED ZIP FILES
            for zipFiles in sortedZipFiles:
                for fileWithExtension in zipFiles:
                    print(fileWithExtension)
                    if(fileWithExtension.endswith(".csv")): # select only the certain extension
                        ##if(fileWithExtension == hand_coordinates): classi, kopieren und hinzufügen
                        # GET THE CONTENT OF FILES AND SEPARATE THEM ACCORDING TO EXTRACTED GROUND TRUTH ELEMENTS
                        pos, neg  = self.__classificateCSV(fileWithExtension, zipClass, gtTrials, lenTrials) # type: List[List[float]], List[List[float]];
                        data: DataConversionTool= DataConversionTool(pos, neg)

                        self.__copyCSVData(fileWithExtension, self.__readLines)
                        print("FROM OUTSIDE:",len(self.__collector.getArmLeft()));
                        print("FROM OUTSIDE:",len(self.__collector.getArmRight()));

                        print("NEGATIVE Trials:", len(data.getNegativeClass()))  # data.getNegativeClass())
                        print("POSITIVE Trials:", len(data.getPositiveClass()))  # , data.getPositiveClass())

                    elif fileWithExtension.endswith(".h5"):
                        if(fileWithExtension=="binocular_perception.h5"):
                            self.__copyH5Data(zipClass, fileWithExtension, "binocularPerception");
                        elif(fileWithExtension=="scene_records.h5"):
                            self.__copyH5Data(zipClass, fileWithExtension,'sceneRecords')

                    elif fileWithExtension.endswith(".png"):
                        pass

    def __copyCSVData(self, fileName: str, readLines: DataFrame) -> None:

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

    def __copyH5Data(self, zipClass: ZipFile, fileName: str, dataSetName: str) -> ndarray:
        """

        """
        with zipClass.open(fileName) as file:
            return self.__imgCollector.loadImageDataFromDataSet(file, dataSetName);

    def __copyImgData(self) -> None:
        pass
    def __classificateCSV(self, fileName: str, zipClass: ZipFile, gtTrials: List[str], lenTrials: List[int]) -> Tuple[List,List]:
        """
         gets into every file and recovers the data according the specifications of the ground truth elements
         @:param fileName: string name of the file
         @:param zipClass: ZipFile data containing the zip files
         @:param gtClass: List of stings contents the ground truth binary values
         @:param lenTrial: List of intgers contents the length of every trial
         @:return: a tuple of positive and negative class list
        """
        positiveClass: List[List[List[float]]] = list()
        negativeClass: List[List[List[float]]] = list()

        with zipClass.open(fileName, "r") as lines:

            self.__readLines = read_csv(lines, delimiter="\n")
            readLines: List[List[float]] = self.__readLines.values.tolist()
            # print(readLines)
            assert (len(gtTrials) == len(lenTrials)), "the length of extracted GTS and Trials are not identical";
            counter: int = 0;
            frames: List[List[float]] = None;
            for i in range(len(gtTrials)):
                try:
                    frames, counter = self.__getFramesData(counter, lenTrials[i], readLines) #type: List[List[float]], int;
                except Exception as e:
                    print(e)
                except IndexError as e:
                    print(e)
                    pass
                if(gtTrials[i]=="0"):
                    negativeClass.append(frames);
                elif(gtTrials[i]=="1"):
                    positiveClass.append(frames);
                else:
                    logging.info("the trial could not be successfully classified")
            #print("SEPARATED CLASSES:", len(positiveClass), len(negativeClass));
            return positiveClass, negativeClass;


    @staticmethod
    def __getFramesData(counter: int, numberOfFrames: int, data: List[List[float]]) -> Tuple[List[List[float]], int]:
        """
        iterates over a partition of the extracted file list. File Partition is given from the counter, which will be actualised at the End of the function
        @:param counter: digit that specifies the start point of the file list
        @:param numberOfFrames: length the partition
        @:param data: list of list extracted from the files, file list
        @:return: tuple with a list of list containing the partition of the file list
        """

        framesData: List[List[float]] = list();

        for row in range(numberOfFrames):
            framesData.append(data[row]);
        counter += numberOfFrames;
        return framesData, counter;

    @staticmethod
    def __getGroundTruthParametersFromCSV(fileName: str, zip: ZipFile) -> Tuple[List, List]:
        gtTrials: List[str] = [];
        lenTrials: List[int] =[];
        with zip.open(fileName, "r") as file:
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
    path: str = "iCubObjectCollisionSimulator/DGP_iCubSimulator/trials/ra_cube_trial6.zip"
    file: str = "gt_right_hand.csv"
    zip: TrialManager = TrialManager()
    zip.openZipFile(path, file)




