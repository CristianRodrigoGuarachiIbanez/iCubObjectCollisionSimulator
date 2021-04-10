from csvDataProduct import CSVDataCollector
from builderInterface import BuilderInterface
from zipfile import ZipFile
from io import BytesIO
from typing import List, Tuple, Dict
from trialManager import TrialManager
from filesExtensionClassifier import FilesExtensionClassifier
from pandas import read_csv, DataFrame, concat
from groundTruthProduct import GroundTruthCollector
from imageProduct import ImgArrayProduct;
from numpy import ndarray
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
from cython import declare, locals, int, char, array, bint



class ProductBuilder(BuilderInterface):
    listOfFileEndings: List[str] = [".csv", ".h5", "png"];
    _csvDataCollector = declare(CSVDataCollector);
    __trialsManager = declare(TrialManager)
    __imgArrayCollector = declare(ImgArrayProduct)
    def __init__(self) -> None:

        #self.__csvCollector: CSVDataCollector = CSVDataCollector();
        self.__reset();

    def __reset(self) -> None:

        self._csvDataCollector: CSVDataCollector = CSVDataCollector();
        self.__trialsManager: TrialManager = TrialManager()
        #
        self.__imgArrayCollector: ImgArrayProduct = ImgArrayProduct();

    @property
    def csvProduct(self) -> CSVDataCollector:
        '''
        saves a copy of the CSVDataCollector and outputs it
        :return: a CSVDataCollector
        '''
        product: CSVDataCollector = self._csvDataCollector;
        self.__reset();
        return product;

    @property
    def imgArrayProduct(self) -> ImgArrayProduct:
        '''
        saves a copy of the ImgArrayProduct and outputs it
        :return: ImgArrayProduct
        '''
        product: ImgArrayProduct = self.__imgArrayCollector;
        self.__reset()
        return product;

    # --------------------------- img data array ---------------
    @locals(zipFileName=char)
    def __produceImgArrayFromH5File(self, zipFileName: str) -> None:
        '''
        :param zipFileName: string;
        :param index: digit;
        '''
        fileClassifier: FilesExtensionClassifier = FilesExtensionClassifier();
        with ZipFile(zipFileName, "r") as zipClass:
            #print("succefully imported")
            # zip.printdir()
            zipFiles: List[str] = zipClass.namelist();  # names of the files in the Zip
            sortedZipFiles: List[str] = fileClassifier.grabFilesWithExtensionH5(zipFiles);  # zip file names sorting according to extension
            self.__trialsManager.ZipClass = zipClass;
            self.__trialsManager.imgCollector = self.__imgArrayCollector;
            # print(sortedZipFiles)
            for hp5files in sortedZipFiles:
               try:
                   self.__trialsManager.saveRecoveredImgData(hp5files);
               except Exception as e:
                   logging.info(e);

    # -------------------------  csv data --------------------------------------
    @locals(zipFileName=char)
    def __produceDataFrameFromCSV(self, zipFileName: str) -> None:
            '''

            :param zipFileName: string;
            :param index: digit;
            :param flag: boolen value
            '''
            fileClassifier: FilesExtensionClassifier = FilesExtensionClassifier()
            with ZipFile(zipFileName, "r") as zipClass:
                #print("succefully imported")
                # zip.printdir()
                zipFiles: List[str] = zipClass.namelist();  # names of the files in the Zip
                sortedZipFiles: List[List[str]] = fileClassifier.sortFilesAccordingToExtentions(zipFiles, self.listOfFileEndings);  # zip file names sorting according to extension
                self.__trialsManager.ZipClass = zipClass;
                self.__trialsManager.csvCollector = self._csvDataCollector;

                #  ITERATE OVER EVERY SINGLE ELEMENT OF THE SORTED ZIP FILES
                for zipFiles in sortedZipFiles:
                    for fileWithExtension in zipFiles:
                        #print(fileWithExtension)
                        if (fileWithExtension.endswith(".csv") and not fileWithExtension.startswith("gt_")):  # select only the certain extension
                            # GETS THE CONTENT OF FILES AND SAVES THEM ACCORDING TO EXTRACTED FILE ELEMENTS
                            try:
                                dataFrame: DataFrame = self.__trialsManager.generateDataFrame(fileWithExtension);
                                self.__trialsManager.saveRecoveredCSVData(fileWithExtension, dataFrame);
                            except Exception as e:
                                print(e)
                                print("data could not be saved")

    # ------------------------- extraction from Collectors ---------------------------------------------
    @locals(dataName=char)
    def readSpecificImgArray(self, dataName: str) -> List[ndarray]:
        '''
        accesses the img numpy product collector and recovers the list of array
        :param dataName: string name
        :return: a list of multi-dim numpy arrays
        '''
        listNumpyArrays: ImgArrayProduct = self.__imgArrayCollector;
        if(dataName=='binocular_img'):
            return listNumpyArrays.getBinocularImgArray();
        elif(dataName=='scene_img'):
            return listNumpyArrays.getBinocularImgArray();
        else:
            logging.info("the numpy array name specification is not in the preselected array names")
            print("introduce one of the following array names:")
            print("binocular_img, scene_img")

    @locals(dataName=char)
    def readSpecificDataFile(self, dataName: str) -> DataFrame:
        '''
        extracts a specific data frame from the specific product collector
        :param dataName: string name
        :return: a single data frama
        '''
        data: CSVDataCollector = self.csvProduct;
        if(dataName == "right_hand"):
            return self.__convertToDataFrame(data.getHandRightData());
        elif(dataName == "left_hand"):
            return self.__convertToDataFrame(data.getHandLeftData());
        elif(dataName == "left_forearm"):
            return self.__convertToDataFrame(data.getForeArmLeft());
        elif(dataName=="right_forearm"):
            return self.__convertToDataFrame(data.getForeArmRight());
        elif(dataName=="left_arm"):
            return self.__convertToDataFrame(data.getArmLeft());
        elif(dataName=='right_arm'):
            return self.__convertToDataFrame(data.getArmRight())
        elif(dataName=="joint_coord"):
            return self.__convertToDataFrame(data.getJointCoordData());
        elif(dataName=="hand_coord"):
            return self.__convertToDataFrame(data.getHandCoordData());
        elif(dataName=="head_coord"):
            return self.__convertToDataFrame(data.getHeadCoordData());
        elif(dataName=="object_coord"):
            return self.__convertToDataFrame(data.getObjectCoordData());
        else:

            logging.info("the data-frame name specification is not in the preselected data-frame names")
            print("introduce one of the following data-frame names:")
            print("left_hand, right_hand, left_forearm, right_forearm, left_arm, right_arm, joint_coord, hand_coord, head_coord, object_coord")
            raise AssertionError("the data-frame name specification is not in the preselected data-frame names");
    @staticmethod
    @locals(data=array)
    def __convertToDataFrame(data: List[DataFrame]) -> DataFrame:
        '''

        :param data:
        :return:
        '''
        dataframe: DataFrame = DataFrame();
        assert (len(data) > 0), "the list of Data frames ist empty";
        try:
            dataframe = concat(data, );
        except Exception as e:
            print(e);
            logging.info("the data could not be converted into data frame");
        return dataframe;


    # -----------------------------------------------------------------------------------
    #####################################################################################
    # ------------------------------------------------------------------------------------
    @locals(zipFileName=char, printLen=bint)
    def produceDataFrame(self, zipFileName: str, printLen: bool = False) -> None:
        '''
         executes the func to extract data frames
        :param zipFileName: string name
        :param index: digit
        :param printLen: boolean value as flag
        :return:
        '''
        self.__produceDataFrameFromCSV(zipFileName);
        if(printLen):
            data: CSVDataCollector = self.csvProduct;
            logging.info("OUTSIDE, ARMLEFT:", len(data.getArmLeft()));
            logging.info("OUTSIDE, ARMRIGHT:", len(data.getArmRight()));
            logging.info("OUTSIDE, HEAD:", len(data.getHeadCoordData()));
            logging.info("OUTSIDE, HAND:", len(data.getHandCoordData()));
            logging.info("OUTSIDE, HANDRIGHT:", len(data.getHandRightData()));
            logging.info("OUTSIDE, HANDLEFT:", len(data.getHandLeftData()));
            logging.info("OUTSIDE, FOREARMRIGHT:", len(data.getForeArmRight()));
            logging.info("OUTSIDE, FOREARMLEFT:", len(data.getForeArmLeft()));
            logging.info("OUTSIDE, JOINTS:", len(data.getJointCoordData()));

    def produceImgArray(self, zipFileName: str, flag: str = False) -> None:
        self.__produceImgArrayFromH5File(zipFileName);
        if(flag):
            data: ImgArrayProduct = self.imgArrayProduct;
            logging.info("GETTING BINOCULAR NUMPY ARRAY:", len(data.getBinocularImgArray()));
            logging.info("GETTING SCENE NUMPY ARRAY:", len(data.getSceneImgArray()));


if __name__ == "__main__":


    pass