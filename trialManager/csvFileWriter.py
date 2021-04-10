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
        #print("DATA TUPLE:", raw_data)
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
