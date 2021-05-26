from numpy import ndarray, asarray
from h5py import File, Group
from zipfile import ZipFile
from typing import List, IO, TypeVar, Dict, Any
from os import getcwd, path, remove
import logging
from pickle import dump, load
from cython import declare, locals, int, array, char
class DataStorage:
    T: TypeVar = TypeVar('T', List[ndarray], List[Dict[str, int]])
    @staticmethod
    def storeData(pickelFileName: str, data: T,  mode: str = 'ab'):

        with open(pickelFileName, mode) as file:
            dump(data, file)
            print('... successfully saved')
            file.close()
    @staticmethod
    def loadData(pickelFileName: str, mode: str = 'rb'):
        with open(pickelFileName,mode ) as file:
            data: List= load(file);
            print('... successfully recovered')
            try:
                for i in range(len(data)):
                    yield data[i]
            except StopIteration as s:
                print(s)
                file.close()
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


