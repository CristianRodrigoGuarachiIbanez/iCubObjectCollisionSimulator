
'''
    File name: buildDirector.py
    Author: Cristian R. Guarachi Ibanez
    Date created: 01.05.2021
    Date last modified: 10.06.2021
    Python Version: 3.8
'''

from productBuilder import ProductBuilder;
from os import listdir, getcwd;
from filesExtensionClassifier import FilesExtensionClassifier
from h5Writer import H5Writer
from pandas import DataFrame, to_numeric;
from typing import List, Tuple, TypeVar
from numpy import ndarray, float as Float, asarray;
import logging
class BuildDirector:
    """
       The Director is only responsible for executing the building steps in a
       particular sequence. It is helpful when producing products according to a
       specific order or configuration. Strictly speaking, the Director class is
       optional, since the client can control builders directly.
    """
    T: TypeVar = TypeVar("T", ndarray, DataFrame);
    def __init__(self) -> None:
        self._builder: ProductBuilder = None;

    @property
    def builder(self) -> ProductBuilder:
        return self._builder

    @builder.setter
    def builder(self, builder: ProductBuilder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled product.
        """
        self._builder = builder

    """
    The Director can construct several product variations using the same
    building steps.
    """

    def __recoverListOfDataFrame(self, direction: str) -> None:
        '''
        recovers a data frame of the CSV Files
        :param dataName: string name of the data frame
        :param direction: string indicating the direction of head and eyes and the arm being tested

        '''
        for file in listdir(getcwd()):
            if (direction):
                assert((direction =='la') or (direction=='ra')), 'select a valid selection key: "la" -> Link Arm, "ra" -> Rigth Arm or None'
                if (file.startswith(direction) and file.endswith('.zip')):
                    try:
                        self._builder.produceDataFrame(file);
                    except Exception as e:
                        logging.info('BUILD DIRECTOR -> RECOVER CSV DATA PRODUCT:', e);
            else:
                if file.endswith(".zip"):
                    try:
                        self._builder.produceDataFrame(file);
                    except Exception as e:
                        logging.info('BUILD DIRECTOR -> RECOVER CSV DATA PRODUCT:', e);

    def __recoverImgArrayProduct(self, h5pyfile: str, direction:str) -> None:
        '''
               recovers a arrary data from the images and saves it as a object class structure
               :param h5pyfile: string name of the h5 file
               :param direction: string indicating the direction of head and eyes and the arm being tested
        '''

        for file in listdir(getcwd()):
            if (direction):
                assert((direction =='la') or (direction=='ra')), 'select a valid selection key: "la" -> Link Arm, "ra" -> Rigth Arm or None'
                if (file.startswith(direction) and file.endswith('.zip')):
                    try:
                        self._builder.produceImgArray(file, h5pyfile=h5pyfile);
                    except Exception as e:
                        logging.info('BUILD DIRECTOR -> RECOVER IMG ARRAY PRODUCT:', e);
            else:
                if file.endswith(".zip"):
                    try:
                        self._builder.produceImgArray(file, h5pyfile=h5pyfile);
                    except Exception as e:
                        logging.info('BUILD DIRECTOR -> RECOVER IMG ARRAY PRODUCT:', e);
    # ------------------------- build the specific parts -----------------

    def buildCoordinateData(self,  dataName: str, direction: str, dtype: str = "numpy" ) -> T:

        self.__recoverListOfDataFrame(direction=direction)
        if(dtype =="numpy"):
            return self._builder.readSpecificDataFile(dataName).to_numpy();
        elif(dtype =="dataframe"):
            return self._builder.readSpecificDataFile(dataName)
        else:
            print('data type dtype is not valid. Introduce one of the following: numpy or dataframe')

    def buildImgArray(self, dataTORecover: str, dataToEdit: str, direction: str = None) -> List[ndarray]:
        self.builder: productBuilder = ProductBuilder()
        self.__recoverImgArrayProduct(h5pyfile=dataToEdit, direction=direction);
        if(dataTORecover =='binocular_img'):
            return self._builder.readSpecificImgArray(dataTORecover);
        elif(dataTORecover == 'scene_img'):
            return self._builder.readSpecificImgArray(dataTORecover);




if __name__ == "__main__":
    # --------------------------- CALL BUILDER
    buildDirector: BuildDirector = BuildDirector();
    #productBuilder: ProductBuilder = ProductBuilder();
    #buildDirector.builder = productBuilder;

    # --------------------------- WRITER
    writer: H5Writer = H5Writer('training_data.h5');

    # --------------------------- prepare image data
    biL: ndarray = asarray(buildDirector.buildImgArray("binocular_img", dataToEdit='binocular_perception.h5', direction='la'));
    biR: ndarray = asarray(buildDirector.buildImgArray("binocular_img", dataToEdit='binocular_perception.h5', direction='ra'));
    scL: ndarray = asarray(buildDirector.buildImgArray('scene_img', dataToEdit='scene_records.h5', direction='la'));
    scR: ndarray = asarray(buildDirector.buildImgArray('scene_img', dataToEdit='scene_records.h5', direction='ra'));
    print("IMG:", biL.shape);
    print("IMG:", biR.shape);
    print('Scene:', scL.shape);
    print('Scene:', scR.shape);
    # --------------------------- save data
    imgData: List[ndarray] = [biL, biR, scL, scR]
    datasetnames: List[str] = ['binocularDataLeft', 'binocularDataRight', 'sceneDataLeft', 'sceneDataRight']
    groupname: str = 'features_data'
    writer.saveImgDataIntoGroup(imgData, groupname, datasetnames)
    writer.closingH5PY()
    # --------------------------- prepare CSV data
    #coord: ndarray = buildDirector.buildCoordinateData("left_forearm", direction='la');
    # print(len(coord))
    #coord1: ndarray = buildDirector.buildCoordinateData('left_forearm', direction='la')
    # print(len(coord1))

    # filename: str = 'training_data.h5'
    # gen = writer.loadImgDataFromGroup(filename, groupName='features_data')
    # for g in gen:
    #     print(g.shape)
