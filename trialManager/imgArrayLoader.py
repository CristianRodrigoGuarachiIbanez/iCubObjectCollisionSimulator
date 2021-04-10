from numpy import ndarray, asarray
from h5py import File, Group
from zipfile import ZipFile
from typing import List, IO, TypeVar
from os import getcwd, path, remove
import logging

from cython import declare, locals, int, array, char
class ImgArrayLoader:


    @locals(filename=IO, datasetname= char)
    def loadImageDataFromDataSet(self, filename: IO, datasetName: str) -> ndarray:
        imgArray: ndarray = None;
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class", file.keys())
            try:
                imgArray = asarray(file.get(datasetName));
                logging.info("...dataset reached successfully");
                print("...dataset reached successfully");
            except Exception as e:
                logging.info("the data set {} could not be opened".format(filename));
                print("the data set {} could not be opened".format(filename));

            logging.info("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array", imgArray[0][0].shape, imgArray[0][0].size)
            self.__closingH5PY(file)
        return imgArray;

    @classmethod
    def loadImgDataFromGroup(cls, filename: str, groupName: str, datasetName: str) -> ndarray:
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class", file.keys())
            try:

                group: Group = file.get(groupName)  # group2 = hf.get('group2/subfolder')
                print("...dataset riched successfully:",
                      group.items())  # [(u'data3', <HDF5 dataset "data3": shape (100, 3333), type "<f8">)]
            except Exception as e:
                logging.info(e)
            try:
                imgArray: ndarray = asarray(group.get(datasetName))  # n1 = group1.get('data1') \n np.array(n1).shape
                print("array was successfully loaded")
                logging.info("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array",
                  imgArray[0][0].shape, imgArray[0][0].size)
            except Exception as e:
                logging.info(e)
            cls.__closingH5PY(file)
            return imgArray;

    @staticmethod
    def __closingH5PY(file: File) -> None:
        file.close();

if __name__ == '__main__':
    # imgArray: ImgArrayLoader = ImgArrayLoader();
    #img: ndarray = imgArray.loadImageDataFromDataSet('binocular_perception.h5', 'binocularPerception');
    #print(len(img[0]))
    # import time
    # direc = getcwd()
    # print(direc)
    # newPath: str = path.join(direc, "binocular_perception.h5")
    # print(newPath)
    # with ZipFile('ra_cube_trial5.zip', 'r') as file:
    #     file.extract('binocular_perception.h5', getcwd())
    #     img: ndarray = imgArray.loadImageDataFromDataSet('binocular_perception.h5', 'binocularPerception');
    #     print(len(img[0]))
    #     remove(newPath)
    pass


