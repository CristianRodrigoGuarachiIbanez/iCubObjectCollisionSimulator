from zipfile import ZipFile
from h5py import File, Group
import logging
from typing import Iterable, Iterator, List, Tuple, TypeVar, Any, Union, IO
from io import StringIO, BytesIO, TextIOWrapper
from csv import reader, register_dialect, QUOTE_ALL
from pandas import read_csv, DataFrame, to_numeric
from pandas.errors import EmptyDataError
from csvDataProduct import CSVDataCollector
from groundTruthProduct import GroundTruthCollector
from imageProduct import ImgArrayProduct;
register_dialect("ownDialect", delimiter=",", skipinitialspace=True, quoting=QUOTE_ALL);
from numpy import ndarray, asarray, array, zeros, loadtxt
from imgArrayLoader import ImgArrayLoader
from cython import declare, locals, int, array, char
from os import getcwd, path, remove



class TrialManager:

    listOfFileEndings: List[str] = [".csv", ".h5", "png"]
    T: TypeVar = TypeVar('T', str, ndarray)

    def __init__(self) -> None:
        self.__zipClass: ZipFile = None;
        self.__csvCollector: CSVDataCollector = None;
        #
        self.__imgCollector: ImgArrayProduct = None;


    @property
    def ZipClass(self) -> ZipFile:
        return self.__zipClass
    @ZipClass.setter
    def ZipClass(self, zipClass: ZipFile) -> None:
        self.__zipClass = zipClass;

    @property
    def csvCollector(self) -> CSVDataCollector:
        return self.__csvCollector
    @csvCollector.setter
    def csvCollector(self, dataCollector: CSVDataCollector) -> None:
        self.__csvCollector = dataCollector;
    #
    @property
    def imgCollector(self) -> ImgArrayProduct:
        return self.__imgCollector;
    @imgCollector.setter
    def imgCollector(self, imgArrayProduct: ImgArrayProduct) -> None:
        self.__imgCollector = imgArrayProduct

    # -------------------------- img data ---------------------------------
    def saveRecoveredImgData(self, fileName: str) -> None:

        if(fileName =='binocular_perception.h5'):
            self.__imgCollector.setBinoImgArray(self.__generateBinocularImgArray(fileName));
            #print('binocular data set was sucessfully saved in product class!')
        elif(fileName == 'scene_records.h5'):
            self.__imgCollector.setSceneImgArray(self.__generateSceneImgArray(fileName));
            #print("scene data set was successfully saved in product class")

    def __generateBinocularImgArray(self, h5fileName:str ) -> T:
        try:
            return self.__readNumpyArrayData(h5fileName, dataSetName='binocularPerception') #########
        except Exception as e:
            logging.info(e);
            print('Exception:',e)
            return 'the binocular array could not be recovered'

    def __generateSceneImgArray(self, fileName: str) -> T:
        try:
            return self.__readNumpyArrayData(fileName, dataSetName='sceneRecords') #####
        except Exception as e:
            logging.info(e);
            return 'the scene array could not be recovered'
    @locals(h5fileName=char, dataSetName= char)
    def __readNumpyArrayData(self, h5fileName: str, dataSetName: str) -> ndarray:
        #imgLoader: ImgArrayLoader = ImgArrayLoader();
        #with self.__zipClass.read(h5fileName) as fileArray:
        direcPath: str = getcwd()
        filePath: str = path.join(direcPath, h5fileName)
        self.__zipClass.extract(h5fileName, direcPath);
        imgArray: ndarray = self.__extractImageDataFromDataSet(filePath,dataSetName);
        if(imgArray.size >0):
            remove(filePath)
            return imgArray

    @locals(filename=IO, datasetname=char)
    def __extractImageDataFromDataSet(self, filename: str, datasetName: str) -> ndarray:
        imgArray: ndarray = None;
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class", file.keys())
            try:
                imgArray = asarray(file.get(datasetName));
                logging.info("...dataset reached successfully");
                #print("...dataset reached successfully");
                print("Size of List", len(imgArray), "Size of Tuple", imgArray[0].size, "Size of Array",
                  imgArray[0][0].shape, imgArray[0][0].size)
            except Exception as e:
                print(e)
                print("the data set {} could not be opened".format(filename));

            file.close();
        return imgArray;

    # -------------------------- CSV data ----------------------------------
    def saveRecoveredCSVData(self, fileName: str, readLines: DataFrame) -> None:
        assert (len(readLines) > 0), "read files hat none extracted data, it es probably empty";

        if(fileName=="arm_left.csv"):
            self.__csvCollector.setArmLeft(readLines);
            print(fileName);
        elif(fileName=="arm_right.csv"):
            self.__csvCollector.setArmRight(readLines);
        elif(fileName =="forearm_left.csv"):
            self.__csvCollector.setForeArmLeft(readLines);
        elif(fileName=="forearm_right.csv"):
            self.__csvCollector.setForearmRight(readLines);
        elif(fileName=="hand_coordinates.csv"):
            self.__csvCollector.setHandCoordData(readLines);
        elif(fileName=="hand_left.csv"):
            self.__csvCollector.setHandLeftData(readLines);
        elif(fileName=="hand_right.csv"):
            self.__csvCollector.setHandRightData(readLines);
        elif(fileName=="head_coordinates.csv"):
            self.__csvCollector.setHeadCoordData(readLines);
        elif(fileName=="joints_coordinates.csv"):
            self.__csvCollector.setJointCoordData(readLines);
        elif(fileName=="object_data.csv"):
            self.__csvCollector.setObjectCoordData(readLines);

    @staticmethod
    @locals(file=IO)
    def __readDataCSV(file: IO) -> DataFrame:

        df: DataFrame = None;
        try:
            df = read_csv(file, delim_whitespace=True);
            #cols: DataFrame = df.select_dtypes(exclude=['float']).columns
            #df[cols] = df[cols].apply(to_numeric, downcast='float', errors='coerce')
        except EmptyDataError as e:
            print(e);
            try:
                df = read_csv(file, delimiter="\n");
                # cols: DataFrame = df.select_dtypes(exclude=['float']).columns
                # df[cols] = df[cols].apply(to_numeric, downcast='float', errors='coerce')
            except EmptyDataError as e:
                print(e);
                df = DataFrame();
        return df;
    @locals(CSVDataCollector=char)
    def generateDataFrame(self, CSVfileName: str) -> DataFrame:
        """
         gets into every csv file and recovers the data as a data frame
         @:param fileName: string name of the file
         @:param zipClass: ZipFile data containing the zip file
         @:return: a data frame
        """
        with self.__zipClass.open(CSVfileName, "r") as lines:
            #print(lines)
            return self.__readDataCSV(lines)


if __name__ == "__main__":
    # path: str = "iCubObjectCollisionSimulator/DGP_iCubSimulator/trials/ra_cube_trial6.zip"
    # file: str = "gt_right_hand.csv"




    pass