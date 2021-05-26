import time
from zipfile import ZipFile
from h5py import File, Group
import logging
from typing import Iterable, Iterator, List, Tuple, TypeVar, Any, Union, IO, Generator
from io import StringIO, BytesIO, TextIOWrapper
from csv import reader, register_dialect, QUOTE_ALL
from pandas import read_csv, DataFrame, to_numeric
from pandas.errors import EmptyDataError
from csvDataProduct import CSVDataCollector
from imageEditor import ImageEditor
from imageProduct import ImgArrayProduct;
register_dialect("ownDialect", delimiter=",", skipinitialspace=True, quoting=QUOTE_ALL);
from numpy import ndarray, asarray, array, zeros, loadtxt
from cython import declare, locals, int, array, char
from os import getcwd, path, remove
from os.path import isfile, isdir
from types import GeneratorType


class TrialManager:

    listOfFileEndings: List[str] = [".csv", ".h5", "png"]
    T: TypeVar = TypeVar('T', str, ndarray)

    def __init__(self) -> None:
        self.__zipClass: ZipFile = None;
        self.__csvCollector: CSVDataCollector = None;
        #
        self.__imgCollector: ImgArrayProduct = None;
        self.__imgEditor: ImageEditor = None;


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
    @property
    def imgEditor(self) -> ImageEditor:
        return self.__imgEditor;
    @imgEditor.setter
    def imgEditor(self, imgEditor: ImageEditor) -> None:
        self.__imgEditor = imgEditor;

    # -------------------------- img data ---------------------------------
    def saveRecoveredImgData(self, fileName: str) -> None:
        '''
        the image array will be reconstructed with the dimensions (None, 2, 120, 160, 1)
        '''
        img: Generator = None;
        if(fileName =='binocular_perception.h5'):
            # IMGARRAY: List[ndarray] = list();
            img = self.__generateBinocularImgArray(fileName);
            counter: int = 1;
            while True:
                BINOCULAR: List[ndarray] = list();
                binocularArray: ndarray = None;
                try:
                    while True:
                        if(counter>2):
                            counter = 1
                            break
                        else:
                            binocularArray = next(img);
                            BINOCULAR.append(binocularArray);
                            print('TRIAL MANAGER -> RECOVER IMAGE DATA',binocularArray.shape)
                            counter += 1;
                except StopIteration as e:
                    print('StopIteration:', e)
                    break
                self.__imgCollector.setBinoImgArray(asarray(BINOCULAR));
                #print(len(self.__imgCollector.getBinocularImgArray()))

        elif(fileName == 'scene_records.h5'):
            img = self.__generateSceneImgArray(fileName)
            while True:
                binocularArray = ndarray = None;
                try:
                    binocularArray = next(img);
                    self.__imgCollector.setSceneImgArray(binocularArray);
                except StopIteration as s:
                    print('StopIteration:', s);
                    break

    def __generateBinocularImgArray(self, h5fileName:str ) -> Generator:

        data: ndarray = self.__readNumpyArrayData(h5fileName, dataSetName='binocularPerception') #########
        assert (len(data.shape) ==5), 'the dimension of the image array is no large enough: {}'.format(data.shape)
        try:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    yield self.__imgEditor.editImagArray(data[i,j], equa_method='clahe', scale=50);
        except Exception as e:
            logging.info(e);
            print('EXCEPTION -> TRIAL MANAGER -> GENERATOR BINOCULAR ARRAY:',e)
            return None

    def __generateSceneImgArray(self, fileName: str) -> Generator:
        data: ndarray = self.__readNumpyArrayData(fileName, dataSetName='sceneRecords') #####
        assert(len(data.shape) ==4), 'the dimension of the image array is no large enough: {}'.format(data.shape)
        try:
            for i in range(len(data)):
                yield self.__imgEditor.editImagArray(data[i], equa_method='clahe', scale=50);
        except Exception as e:
            logging.info(e);
            return 'the scene array could not be recovered'

    @locals(h5fileName=char, dataSetName= char)
    def __readNumpyArrayData(self, h5fileName: str, dataSetName: str) -> ndarray:

        direcPath: str = getcwd()
        filePath: str = path.join(direcPath, h5fileName)

        self.__zipClass.extract(h5fileName, direcPath); # extract the file to the directory
        imgArray: ndarray = self.__extractImageDataFromDataSet(filePath,dataSetName);
        if (isfile(filePath)): #(isinstance(imgArray, GeneratorType)):     # ((imgArray.size >0) or
            remove(filePath) # delete the file from the directory
            return imgArray # next passes the data inside the generator along
        #else:
            #print('It is not a image array generator!')

    @locals(filename=IO, datasetname=char)
    def __extractImageDataFromDataSet(self, filename: str, datasetName: str) -> ndarray:
        imgArray: ndarray = None;
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class", file.keys())
            try:
                imgArray = asarray(file.get(datasetName));
                print("Size of List", len(imgArray), "Size of Tuple", imgArray[0].size, "Size of Array", imgArray[0][0].shape, imgArray[0][0].size)

            except Exception as e:
                print(e)
                print("the data set {} could not be opened".format(filename));
            file.close()
            return imgArray
        #return imgArray;

    # -------------------------- CSV data ----------------------------------
    def saveRecoveredCSVData(self, fileName: str, readLines: DataFrame) -> None:
        assert (len(readLines) > 0), "read files hat none extracted data, it es probably empty";

        if(fileName=="arm_left.csv"):
            self.__csvCollector.setArmLeft(readLines);
            #print(fileName);
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
    def __readDataCSV(file: IO) -> Generator:

        df: DataFrame = None;
        try:
            df = read_csv(file, delim_whitespace=True);
            #cols: DataFrame = df.select_dtypes(exclude=['float']).columns
            #df[cols] = df[cols].apply(to_numeric, downcast='float', errors='coerce')
        except Exception as e:
            print(e)
        except EmptyDataError as e:
            print(e);
            try:
                df = read_csv(file, delimiter="\n");
                # cols: DataFrame = df.select_dtypes(exclude=['float']).columns
                # df[cols] = df[cols].apply(to_numeric, downcast='float', errors='coerce')
            except Exception as e:
                print(e)
            except EmptyDataError as e:
                print(e);
                df = DataFrame();
        yield df;

    @locals(CSVDataCollector=char)
    def generateDataFrame(self, CSVfileName: str) -> DataFrame:
        """
         gets into every csv file and recovers the data as a data frame
         @:param fileName: string name of the file
         @:param zipClass: ZipFile data containing the zip file
         @:return: a data frame
        """
        with self.__zipClass.open(CSVfileName, "r") as lines:
            csvFile: Generator =  self.__readDataCSV(lines)
            return next(csvFile)


if __name__ == "__main__":
    # path: str = "iCubObjectCollisionSimulator/DGP_iCubSimulator/trials/ra_cube_trial6.zip"
    # file: str = "gt_right_hand.csv"


    pass