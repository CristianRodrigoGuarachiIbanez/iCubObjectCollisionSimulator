from abc import ABC, abstractmethod, abstractproperty
import h5py
from h5py import File, Group
from numpy import ndarray, asarray
from typing import List, Tuple, TypeVar, Any
from PIL import Image as img
import logging

class ImgArrayInterface(ABC):
    L = TypeVar("L", ndarray, List[List[Any]])
    @abstractmethod
    def saveImgDataIntoDataSet(cls, imgData: L, filename: str, datasetName: str ) -> None:
        """
        @:param imgData: a numpy array or list of floats data
        @:param filename: a string name of the file
        @:param datasetName: a string name of the data set
        """
        pass

    @abstractmethod
    def loadImageDataFromDataSet(cls, filename: str, datasetName: str) -> ndarray:
        """
        recovers the array image data.
        @:param filename: string name of the file
        @:param datasetName: string name of the data set
        @:return: multi dim array
        """
        pass

    @abstractmethod
    def loadImgDataFromGroup(cls, mgData: L, filename: str, groupName: str, datasetName: str) -> None:
        pass


class ImgArrayRecoverer(ImgArrayInterface):
    L = TypeVar("L", ndarray, List[List[Any]]);

    @classmethod
    def saveImgDataIntoDataSet(cls, imgData: L, filename: str, datasetName: str) -> None:
        """
        @:param imgData: a numpy array or list of floats data
        @:param filename: a string name of the file
        @:param datasetName: a string name of the data set
        """
        with File(filename, "w") as file:
            file.create_dataset(datasetName, data=imgData);
            print("... dateset created und data saved");
            cls.__closingH5PY(file);

    @classmethod
    def loadImageDataFromDataSet(cls, filename: str, datasetName: str) -> ndarray:
        """
        recovers the array image data.
        @:param filename: string name of the file
        @:param datasetName: string name of the data set
        @:return: multi dim array
        """
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class", file.keys())
            imgArray: ndarray = asarray(file.get(datasetName));
            print("...dataset reached successfully");
            print("printing from inside the function:");
            print("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array", imgArray[0][0].shape, imgArray[0][0].size);
            cls.__closingH5PY(file);
        return imgArray;

    @classmethod
    def loadImgDataFromGroup(cls, mgData: L, filename: str, groupName: str, datasetName: str ) -> None:
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class",file.keys())
            group: Group = file.get(groupName)  # group2 = hf.get('group2/subfolder')
            print("...dataset riched successfully:", group.items()) # [(u'data3', <HDF5 dataset "data3": shape (100, 3333), type "<f8">)]
            imgArray: ndarray = asarray(group.get(datasetName)) # n1 = group1.get('data1') \n np.array(n1).shape

            print("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array", imgArray[0][0].shape, imgArray[0][0].size)
            cls.__closingH5PY(file)

    @staticmethod
    def __closingH5PY(file: File) -> None:
        file.close();

    @staticmethod
    def imgFromArray(array: ndarray, filename:str) -> None:
        # creating image object of
        # above array
        data: img = img.fromarray(array);

        # saving the final output
        # as a PNG file
        data.save(filename);