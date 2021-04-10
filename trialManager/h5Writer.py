#import cv2 as cv
from typing import List, Dict, Tuple, TypeVar, Any, Callable, Union, KeysView
import logging
import sys
#import numpy as np
from numpy import ndarray, asarray
import pandas as pd
from h5py import File, Group
from PIL import Image as im
from numpy import ndarray

# ----------------------------------------- CAMERAS COORDINATES WRITER -----------------------------------------------
class CamerasCoordinatesWriter:
    L = TypeVar("L", ndarray, List)
    #__H5PY: h5py.File = h5py.File(fileobj=None, mode=None)
    @classmethod
    def saveImgDataIntoDataSet(cls, imgData: L, filename: str, datasetName: str) -> None:
        """
        @:param imgData: a numpy array or list of floats data
        @:param filename: a string name of the file
        @:param datasetName: a string name of the data set
        """
        with File(filename, "w") as file:
            file.create_dataset(datasetName, data = imgData)
            print("... dateset created und data saved")
            cls.__closingH5PY(file)
    @classmethod
    def loadImageDataFromDataSet(cls, filename: str, datasetName: str) ->ndarray:
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class",file.keys())
            try:

                imgArray: ndarray = asarray(file.get(datasetName))
                print("...dataset riched successfully ")
                #print("printing from inside the function:")

            except Exception as e:
                print("the data set {} could not be opened".format(filename));

            print("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array",
            imgArray[0][0].shape, imgArray[0][0].size)
            cls.__closingH5PY(file)
        return imgArray
    @classmethod
    def saveImgDatasetIntoGroup(cls, imgData: L, filename: str, groupName: str, datasetName: str) -> None:
        with File(filename, 'w') as file:
            g1: Group = file.create_group(groupName)
            print('... group was created successfully!')
            g1.create_dataset(datasetName, data=imgData)
            print('... dataset was created successfully!')
    @classmethod
    def loadImgDataFromGroup(cls, mgData: L, filename: str, groupName: str, datasetName: str ) -> None:
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class",file.keys())
            group: Group = file.get(groupName)  # group2 = hf.get('group2/subfolder')
            print("...dataset riched successfully:", group.items()) # [(u'data3', <HDF5 dataset "data3": shape (100, 3333), type "<f8">)]
            imgArray: ndarray = asarray(group.get(datasetName)) # n1 = group1.get('data1') \n np.array(n1).shape
            print("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array",
            imgArray[0][0].shape, imgArray[0][0].size)
            cls.__closingH5PY(file)
    @staticmethod
    def __closingH5PY(file: File) -> None:
        file.close()
    @staticmethod
    def imgFromArray(array: ndarray, filename:str) -> None:
        # creating image object of
        # above array
        data = im.fromarray(array)
        # saving the final output
        # as a PNG file
        data.save(filename)
class GroundTruthWriter:
    @classmethod
    def saveGroundTruthtoCVS(cls, data: List[Tuple], filename: str = "output_object", mode: chr = 'w', fileType: str = 'csv' ) -> None:
        '''
        convert a list of list in a dataframe and saves it into a csv or xlsx file
        @:param ObjectCoord: list of list with floating numbers as coordinates
        @:param filename: a string
        @:param mode: a chart data type indicating to update the file
        @:param fileType: a string
        '''
        try:
            df: pd.DataFrame = cls.__convertListToDataframe(data)
            # print(df)
            if (fileType == "csv"):
                df.to_csv(filename, mode=mode)
            elif (fileType == "xlsx"):
                df.to_excel(filename)
        except Exception as e:
            print(e)
            sys.exit('the ground truth raw data could not be converted to data frame!')
    @staticmethod
    def __convertListToDataframe(data: List[Tuple]) -> pd.DataFrame:
        return pd.DataFrame(data, columns =['NumTrial', 'CollisionPerTrial', 'GroundTruth'])
if __name__ == "__main__":
    pass