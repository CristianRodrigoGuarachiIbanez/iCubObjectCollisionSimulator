import numpy as np
import cv2 as cv
import h5py
from typing import Tuple, List, TypeVar
from PIL import Image as im

class camerasCoordinatesWriter:
    L = TypeVar("L", np.ndarray, List)
    #__H5PY: h5py.File = h5py.File(fileobj=None, mode=None)

    @classmethod
    def saveImgDataIntoDataSet(cls, imgData: L, filename: str, datasetName: str) -> None:
        with h5py.File(filename, "w") as file:
            file.create_dataset(datasetName, data = imgData)
            print("... dateset created und data saved")
            cls.__closingH5PY(file)
    @classmethod
    def loadImageDataFromDataSet(cls, filename: str, datasetName: str) ->np.ndarray:
        with h5py.File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class",file.keys())
            imgArray: np.ndarray = np.asarray(file.get(datasetName))
            print("...dataset riched successfully ")
            print("printing from inside the function:")
            print("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array",
            imgArray[0][0].shape, imgArray[0][0].size)
            cls.__closingH5PY(file)
        return imgArray
    @classmethod
    def saveImgDatasetIntoGroup(cls, imgData: L, filename: str, groupName: str, datasetName: str) -> None:
        with h5py.File(filename, 'w') as file:
            g1 =file.create_group(groupName)
            print('... group was created successfully!')
            g1.create_dataset(datasetName, data=imgData)
            print('... dataset was created successfully!')
    @classmethod
    def loadImgDataFromGroup(cls, mgData: L, filename: str, groupName: str, datasetName: str ) -> None:
        with h5py.File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class",file.keys())
            group: h5py.Group = file.get(groupName)  # group2 = hf.get('group2/subfolder')
            print("...dataset riched successfully:", group.items()) # [(u'data3', <HDF5 dataset "data3": shape (100, 3333), type "<f8">)]
            imgArray: np.ndarray = np.asarray(group.get(datasetName)) # n1 = group1.get('data1') \n np.array(n1).shape

            print("Size of List", len(imgArray), "Size of Tuple", len(imgArray[0]), "Size of Array",
            imgArray[0][0].shape, imgArray[0][0].size)
            cls.__closingH5PY(file)
    @staticmethod
    def __closingH5PY(file: h5py.File) -> None:
        file.close()

    @staticmethod
    def imgFromArray(array: np.ndarray, filename:str) -> None:
        # creating image object of
        # above array
        data = im.fromarray(array)

        # saving the final output
        # as a PNG file
        data.save(filename)

L = TypeVar("L", np.ndarray, List)
def __readImage(pfade: str = "scene_None.png" ) -> cv:
    return cv.imread(pfade, 1)
def __readImageReturnedAsTuple(img1: cv, img2:cv) -> Tuple:
    return img1, img2

def __readImageReturnsList(img12: Tuple, img34: Tuple) -> List[Tuple]:
    return [img12, img34]

def __saveImgArray(imgArr: L) -> None:
    hf: h5py.File = h5py.File('data.h5', 'w')
    hf.create_dataset('first_img', data=imgArr)
    print('numpy array saved!')
    hf.close()
def __readImgArray(pfade: str = "data.h5") -> np.ndarray:
    with h5py.File(pfade, 'r') as imgArr:
        imgArr.keys()
        imgArr_1: np.ndarray = np.asarray(imgArr.get('first_img'))
        imgArr.close()
        print("printing from inside the function:",imgArr)
        return imgArr_1

import progressbar
from progress.bar import Bar
from time import sleep


if __name__ == "__main__":

    # img11: Tuple = __readImageReturnedAsTuple(__readImage("scene.png"), __readImage("scene_0.png"))
    # img34: Tuple = __readImageReturnedAsTuple(__readImage("scene_1.png"), __readImage("scene_2.png"))
    # imgs: List[Tuple] = __readImageReturnsList( img11, img34 )
    # print("Size of List", len(imgs),"Size of Tuple",len(imgs[0]), "Size of Array", imgs[0][0].shape, imgs[0][0].size)
    # # __saveImgArray(imgs)
    # # retrivedData: np.ndarray = __readImgArray("data.h5")
    # # #print(retrivedData)
    # # print("Size of List", len(retrivedData),"Size of Tuple",len(retrivedData[0]), "Size of Array", retrivedData[0][0].shape, retrivedData[0][0].size)
    #
    #
    # # class
    # writer: camerasCoordinatesWriter = camerasCoordinatesWriter()
    # writer.saveImgData(imgs, 'fourImgAsList.h5','trial0')
    #
    # recovered: np.ndarray = writer.loadImageData("fourImgAsList.h5", 'trial0')
    # print("Size of List", len(recovered), "Size of Tuple", len(recovered[0]), "Size of Array",
    #       recovered[0][0].shape, recovered[0][0].size)
    #
    #
    # #main(recovered[0][0])

    # bar: progressbar.ProgressBar = progressbar.ProgressBar(maxval=11, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()
    # for i in range(0,11):
    #     print(i, end='\n')
    #     print(end='\n')
    #     bar.update(i + 1)
    #     sleep(0.1)
    # bar.finish()

    bar: Bar = Bar('Processing', max=20, fill='=', suffix='%(percent)d%%')
    for i in range(20):
        # Do some work
        print("" , i)
        sleep(.1)
        bar.next()
    bar.finish()