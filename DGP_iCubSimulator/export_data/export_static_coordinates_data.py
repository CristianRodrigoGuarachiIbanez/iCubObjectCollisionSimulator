"""
Created on Fr Dic 4 2020
@author: Cristian Rodrigo Guarachi Ibanez

Export static coordinates data
"""


#import cv2 as cv
from typing import List, Dict, Tuple, TypeVar, Any, Callable, Union, KeysView
import logging
import sys
import numpy as np
import pandas as pd
import h5py
from PIL import Image as im
from numpy import ndarray

# ------------------------------------ HAND COORDINATES WRITER ----------------------------------------------
class HandCoordinatesWriter:
    L = TypeVar("L", List[List], np.ndarray)

    @classmethod
    def HandCoordinatesToCSV(cls, rowData: List[Dict], export: str = "csv", mode: chr ="w", filename: str = "hand_coordinates.csv") -> None:
        '''
        convert a list of list in a dataframe and saves it into a csv or xlsx file
        @:param ObjectCoord: list of list with floating numbers as coordinates
        @:param filename: a string
        @:param mode: a chart data type indicating to update the file
        @:param fileType: a string
        '''
        try:
            data: np.ndarray = cls.__concatenateFloatArrays(cls.__rowDataDictToList(rowData))
            #print("DATA ARRAY", data)
            df: pd.DataFrame = cls.__convertToDataframe(data, cls.__putHeaderToList())
            #print("DATA FRAME", df)
            if(export=='csv'):
                df.to_csv(filename, mode=mode, float_format= '%2f')
            elif(export =='xlsx'):
                df.to_excel(filename)
        except Exception as e:
            print(e)
            sys.exit('the hand raw data could not be converted to a data frame!')

    @classmethod
    def JointsCoodinatesToCSV(cls, RowData: List[List[float]], mode: chr = 'w', export: str = 'csv', filename: str = 'hand_joints.csv') -> None:
        """
        convert a list of list in a dataframe and saves it into a csv or xlsx file
        @:param ObjectCoord: list of list with floating numbers as coordinates
        @:param filename: a string
        @:param mode: a chart data type indicating to update the file
        @:param fileType: a string
        """
        headers: List[str] = ["Joint_" + str(i) for i in range(1, len(RowData[0])+1)]
        try:
            df: pd.DataFrame = cls.__convertToDataframe(np.asarray(RowData), headers)
            #print(df)
            if(export == 'csv'):
                df.to_csv(filename, mode=mode, float_format='%2f')
            elif(export == 'xlsx'):
                df.to_csv(filename)
        except Exception as e:
            print(e)
            sys.exit('the joint raw data could not be converted to dataframe!')

    @staticmethod
    def __putHeaderToList(concatenatedCoord: List[List[float]] =  None) -> List[str]:
        positionHeader: List[str] = ['pose_X', 'pose_Y', 'pose_Z']
        orientationHeader: List[str] = ['orient_1', 'orient_2', 'orient_3', 'orient_4']
        AllHeaders: List[str] = [y for x in [positionHeader, orientationHeader] for y in x]

        #concatenatedCoord.insert(0, AllHeaders)
        return AllHeaders

    @staticmethod
    def __convertToDataframe(DataArray: L, header: List[str] = None) -> pd.DataFrame:
        """

        """
        #print("DATA:", DataArray)
        if (header):
            return pd.DataFrame(DataArray[:, :], index=[x for x in range(1,len(DataArray)+1)], columns=header)

        return pd.DataFrame(DataArray[1:, :], index=[x for x in range(1, len(DataArray))], columns=DataArray[0])

    @classmethod
    def __concatenateFloatArrays(cls, raw_data: Tuple[List, List]) -> ndarray:
        """separate a multidimentional array in 2 2D-arrays concatenating them on the axis 1
        @:parameter: takes a Tuple of Lists
        @:return: numpy array"""
        print("DATA TUPLE:", raw_data)
        position: np.ndarray = np.zeros((len(raw_data[0]), len(raw_data[0][0])))
        orientation: np.ndarray = np.zeros((len(raw_data[1]), len(raw_data[1][0])))
        for row in range(len(position)):
            for col in range(len(position[row])):
                position[row][col] = raw_data[0][row][col]
        for row in range(len(orientation)):
            for col in range(len(orientation[row])):
                orientation[row][col] = raw_data[1][row][col]
        return cls.__joint_pose_orient(position, orientation)

    @staticmethod
    def __joint_pose_orient(arr1: np.ndarray, arr2: np.ndarray) -> ndarray:
        """concatenate 2d array
        @parameters take 2 arrays from similar number of rows
        @returns a array"""
        output: np.ndarray = np.empty((len(arr1), len(arr1[0])))

        if (len(arr1) == len(arr2)):
            output: ndarray = np.concatenate((arr1, arr2), axis=1)
        else:
            print("the length of the arrays is not similar")
        return output

    @staticmethod
    def __rowDataDictToList(rowdata: List[Dict]) -> Tuple[List, List]:
        handPosition: List[List[float]] = []
        handOrientation: List[List[float]] = []
        for dicts in rowdata:
            for index, values in dicts.items():
                if(index=='RightHandPosi'):
                    handPosition.append([float(x.replace(',', '.')) for x in values.split("|")])
                elif(index == 'RightHandOrient'):
                    handOrientation.append([float(x.replace(',', '.')) for x in values.split("|")])

        return handPosition, handOrientation

# @staticmethod
# def __jointStringArrays(strings: List[str], strings2: List[str]) -> List[str]:
#     return [y for x in [strings, strings2] for y in x]

# --------------------------------- HEAD COORDINATES WRITER ---------------------------------------

class HeadCoordinatesWriter:
    L = TypeVar("L", List[List], np.ndarray)

    @classmethod
    def outputToCSV(cls, CoordRow: List, filename: str = "head_coordinates.csv") -> None:
        """
        convert a list of list in a dataframe and saves it into a csv or xlsx file
        @:param ObjectCoord: list of list with floating numbers as coordinates
        @:param filename: a string
        @:param mode: a chart data type indicating to update the file
        @:param fileType: a string
        """
        data: np.ndarray = cls.__convertListToArray(cls.__putHeadersToTheList(CoordRow))
        #print(data)
        try:
            data: pd.DataFrame = pd.DataFrame(data[1:,:], index=[x for x in range(1, len(data))], columns=data[0])
            #print(data)
            data.to_csv(filename)
            #print("the Numpy array was saved as .csv file")
        except Exception as e:
            print(e)
            sys.exit('the head raw data could not be converted to data frame!')

    @staticmethod
    def __convertListToArray(CoordWithHeaders: List) -> np.ndarray:
        return np.asarray(CoordWithHeaders)

    @staticmethod
    def __putHeadersToTheList(Coord: List)-> List:
        headers: List = ["HorizontalRotationn", "LateralTranslation", "VerticalRotation", "GazeDirectionLeftRight", "GazedirectionUpDown", "EyeOpossiteVertRotation"]
        Coord.insert(0, headers)
        #print(Coord)
        return Coord

# ---------------------------------------------- SENSOR COORDINATES WRITER ----------------------------------------

class SensorCoordinatesWriter:

    @classmethod
    def outputDataToCSV(cls, rowData: List[List[float]], filename: str = "arm_sensor_ouputs.csv", mode: str = 'w', fileType: str = 'csv') -> None:
        """
        convert a list of list in a dataframe and saves it into a csv or xlsx file
        @:param ObjectCoord: list of list with floating numbers as coordinates
        @:param filename: a string
        @:param mode: a chart data type indicating to update the file
        @:param fileType: a string
        """
        try:
            df: pd.DataFrame = cls.__arrayDataToDataframe(cls.__convertListToArray(rowData), cls.__putHeadersToList(rowData))
            if(fileType == 'csv'):
                df.to_csv(filename, mode=mode, float_format ='%2f')
            elif(fileType == 'xlsx'):
                df.to_excel(filename)
            #print(df)
        except Exception as e:
            print(e)
            sys.exit('the sensor row data could not be converted to data frame!')

    @staticmethod
    def __arrayDataToDataframe(sensorsCoord: np.ndarray, headers: List[str]) -> pd.DataFrame:
        return pd.DataFrame(sensorsCoord[:,:], index=[x for x in range(1, len(sensorsCoord)+1)], columns=headers)

    @staticmethod
    def __convertListToArray(SensorsCoord: List[List[float]]) -> np.ndarray:
        return np.asarray(SensorsCoord)
    @staticmethod
    def __putHeadersToList(SensorsCoord: List[List[float]]):
        return ["sensor_" + str(i) for i in range(1, len(SensorsCoord[0])+1)]


# ------------------------------------- OBJECT COORDINATES WRITER ------------------------------------------

class ObjectCoordinatesWriter:
    @classmethod
    def outputDataToCSV(cls, ObjectCoord: List[List[float]], filename: str = "output_object", mode: chr = 'w', fileType: str = 'csv' ) -> None:
        '''
        convert a list of list in a dataframe and saves it into a csv or xlsx file
        @:param ObjectCoord: list of list with floating numbers as coordinates
        @:param filename: a string
        @:param mode: a chart data type indicating to update the file
        @:param fileType: a string
        '''
        try:
            df: pd.DataFrame = cls.__convertArrayToDataframe(cls.__convertListToArray(ObjectCoord), cls.__putHeaders())
            #print(df)
            if(fileType == "csv"):
                df.to_csv(filename, mode=mode)
            elif(fileType =="xlsx"):
                df.to_excel(filename)
        except Exception as e:
            print(e)
            sys.exit('the object raw data could not be converted to data frame!')
    @staticmethod
    def __putHeaders(headers: List[str] = ['axisX', 'axisY', 'axisZ'])-> List[str]:
        """
        :param headers: a list of strings
        :return: a list of string with the headers in the 3 axis
        """
        return headers

    @staticmethod
    def __convertListToArray(ObjectCoord: List[List[float]]) -> np.ndarray:
        """

        """
        return np.asarray(ObjectCoord)
    @staticmethod
    def __convertArrayToDataframe(ObjectCoord: np.ndarray, headers: List[str] = None) -> pd.DataFrame:
        """

        """
        if(headers):
            return pd.DataFrame(ObjectCoord[:,:], index=[x for x in range(1, len(ObjectCoord)+1)], columns=headers)
        return pd.DataFrame(ObjectCoord[1:,:], index=[x for x in range(len(ObjectCoord))], columns=ObjectCoord[0])

# ----------------------------------------- CAMERAS COORDINATES WRITER -----------------------------------------------

class camerasCoordinatesWriter:
    L = TypeVar("L", np.ndarray, List)
    #__H5PY: h5py.File = h5py.File(fileobj=None, mode=None)

    @classmethod
    def saveImgDataIntoDataSet(cls, imgData: L, filename: str, datasetName: str) -> None:
        """
        @:param imgData: a numpy array or list of floats data
        @:param filename: a string name of the file
        @:param datasetName: a string name of the data set
        """
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
            g1: h5py.Group = file.create_group(groupName)
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
    """lista: List = [[['left_pose_X', 'left_pose_Y', 'left_pose_Z'], [-0.210739, 0.070214, 0.198735],
                    [-0.365226, -0.113365, 0.201576]],
                   [['left_orient_1', 'left_orient_2', 'left_orient_3', 'left_orient_4'],
                    [-0.064876, -0.357082, 0.931817, 1.906142], [0.144769, 0.528344, 0.836597, 3.026171]]]

    head: List = [["HorizontalRotationn", "LateralTranslation", "VerticalRotation", "GazeDirection", "EyeOpossiteVertRotation", "GazeDIrectionHor"],
                  [-33.75082808055433, -9.643059269213834, -24.107615535308224, 6.40766993729216e-09, -3.7185543320855826e-05, 9.999873106412087]]

    output: WriterNewPosition = WriterNewPosition()
    output.HandCooordinatesToCsv("handOutput.csv.csv", lista)
    output.HeadCoordinatesToCsv("headOutput.csv", head) """

"""
    hand = [{'RightHandPosi': '-0.365246, 0.113735, 0.201235', 'RightHandOrient': '-0.001321,-0.985973, 0.166899, 2.818362'},
     {'RightHandPosi': '-0.365246, 0.113735, 0.201235', 'RightHandOrient': '-0.001321,-0.985973, 0.166899, 2.818362'},
     {'RightHandPosi': '-0.365246, 0.113735, 0.201235', 'RightHandOrient': '-0.001321,-0.985973, 0.166899, 2.818362'},
     {'RightHandPosi': '-0.364462, 0.114432, 0.202460', 'RightHandOrient': '-0.001978,-0.985947, 0.167049, 2.814664'},
     {'RightHandPosi': '-0.365245, 0.113739, 0.201235', 'RightHandOrient': '-0.001326,-0.985973, 0.166899, 2.818360'},
     {'RightHandPosi': '-0.365245, 0.113739, 0.201235', 'RightHandOrient': '-0.001326,-0.985973, 0.166899, 2.818360'},
     {'RightHandPosi': '-0.364462, 0.114431, 0.202460', 'RightHandOrient': '-0.001977,-0.985946, 0.167050, 2.814664'}]

    head= [[-35.00006259372558, -10.000001527978846, -29.999994515124854, 3.445175354442912e-11, -2.993695810902318e-13, 10.000000000000002],
              [-35.00006259912176, -10.000001505203386, -29.99999455710475, -1.1346907913775506e-11, 8.74652162547822e-15, 9.999999999999996],
              [-35.00006286126668, -10.000001373845924, -29.999994409331816, -3.0450707674485265e-11, -7.752598713492058e-14, 9.999999999999966],
              [-35.00006285101373, -10.000001596050472, -29.99999439467196, -1.2470614218681909e-11, -5.685239056560843e-14, 10.000000000000027], [-35.000062599242206, -10.00000150531351, -29.999994557075997, -9.847207137128671e-12, 5.565968307122503e-15, 10.000000000000004], [-35.00006114759337, -10.000000693259725, -29.99999492679807, -5.616865413822535e-11, 1.6300335756573046e-13, 9.999999999999995], [-35.00006288090415, -10.000001604578324, -29.999994378398362, 8.447549613552782e-12, -5.526211390643057e-14, 10.000000000000025]]

    #head: List = [[-33.75082808055433, -9.643059269213834, -24.107615535308224, 6.40766993729216e-09, -3.7185543320855826e-05, 9.999873106412087],
    #             [-33.75082808055433, -9.643059269213834, -24.107615535308224, 6.40766993729216e-09, -3.7185543320855826e-05, 9.999873106412087]]
    writer: HeadCoordinatesWriter = HeadCoordinatesWriter()
    writer.outputToCSV(head)

    handWriter: HandCoordinatesWriter = HandCoordinatesWriter()
    handWriter.HandCoordinatesToCSV(hand)

    joints = [np.array([-90.80726927, 23.02473216, 79.61576799, 16.31147315,
            7.83639391, -13.06761053, 6.12093528, 58.9999748,
            19.99999157, 19.99999135, 19.99999157, 9.9999959,
            9.99999587, 9.99999592, 9.99999588, 9.99999683]),
     np.array([-2.49984864e+01, 1.99991370e+01, 1.27923898e-04, 4.99993218e+01,
            4.13405900e-05, 1.07571026e-06, 8.74775044e-06, 5.89999750e+01,
            1.99999915e+01, 1.99999915e+01, 1.99999915e+01, 9.99999574e+00,
            9.99999574e+00, 9.99999575e+00, 9.99999574e+00, 9.99999592e+00]),
     np.array([-2.49984685e+01, 1.99991450e+01, 1.27297952e-04, 4.99993136e+01,
            4.19457700e-05, 1.00865758e-06, 8.86794584e-06, 5.89999750e+01,
            1.99999915e+01, 1.99999915e+01, 1.99999915e+01, 9.99999574e+00,
            9.99999574e+00, 9.99999574e+00, 9.99999574e+00, 9.99999591e+00]),
    np.array([-2.49984894e+01, 1.99991337e+01, 1.28276255e-04, 4.99993254e+01,
            4.15016051e-05, 1.11299927e-06, 8.68796576e-06, 5.89999750e+01,
            1.99999915e+01, 1.99999915e+01, 1.99999915e+01, 9.99999575e+00,
            9.99999574e+00, 9.99999575e+00, 9.99999574e+00, 9.99999593e+00]),
    np.array([-2.49984864e+01, 1.99991370e+01, 1.27923898e-04, 4.99993218e+01,
            4.18606857e-05, 1.07570634e-06, 8.74777100e-06, 5.89999750e+01,
            1.99999915e+01, 1.99999915e+01, 1.99999915e+01, 9.99999574e+00,
            9.99999574e+00, 9.99999575e+00, 9.99999574e+00, 9.99999592e+00]),
    np.array([-2.49984736e+01, 1.99991405e+01, 1.27590664e-04, 4.99993146e+01,
            4.24589290e-05, 1.02769952e-06, 8.85761087e-06, 5.89999750e+01,
            1.99999915e+01, 1.99999915e+01, 1.99999915e+01, 9.99999574e+00,
            9.99999574e+00, 9.99999575e+00, 9.99999574e+00, 9.99999592e+00]),
    np.array([-2.49984936e+01, 1.99991324e+01, 1.28193497e-04, 4.99993275e+01,
            4.06535096e-05, 1.12779535e-06, 8.65599132e-06, 5.89999750e+01,
            1.99999915e+01, 1.99999915e+01, 1.99999915e+01, 9.99999575e+00,
            9.99999574e+00, 9.99999575e+00, 9.99999574e+00, 9.99999593e+00])]
    #print(joints)
    print(len(joints[0]))
    handWriter.JointsCoodinatesToCSV(joints)
    #--------------------------------------- sensor change the name of the files ----------------------------------------------
    sensors = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0]]


    sensor: SensorCoordinatesWriter = SensorCoordinatesWriter()
    sensor.outputDataToCSV(sensors, filename = "forearm_sensor_ouput.csv")

    sensorArm = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    sensor.outputDataToCSV(sensorArm)

    object = [[0.0, 1.0, 0.3], [-0.06666666666666667, 0.6000000000000001, 0.36500000000000005],
     [-0.03333333333333333, 0.75, 0.38250000000000006], [0.0, 0.9, 0.4], [0.0, 0.9, 0.4],
     [-0.03333333333333333, 0.75, 0.38250000000000006], [-0.06666666666666667, 0.6000000000000001, 0.36500000000000005]]

    objectWriter: ObjectCoordinatesWriter = ObjectCoordinatesWriter()
    objectWriter.outputDataToCSV(object)"""