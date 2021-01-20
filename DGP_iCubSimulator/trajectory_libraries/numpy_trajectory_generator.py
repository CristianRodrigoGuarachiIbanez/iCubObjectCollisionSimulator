"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
3dNumpy trajectory generator
"""

import sys
sys.path.insert(0, "~/PycharmProjects/projectPy3.8/iclub_simulator_tools/python_scripts/trajectory_generation")

import numpy as np
from random import choice#
from typing import List, Dict, TypeVar, Any, Tuple
from tabulate import tabulate


class CoordinatesGenerator:

    A = TypeVar("A", List, np.ndarray)
    @staticmethod
    def range_selector(beginn: float, end: float) -> float:
        range_num: Any = np.linspace(beginn, end, endpoint=False, num=20, retstep=True)

        return choice(range_num[0])

    @staticmethod
    def generator(steps: int, step: float) -> np.array:
        path: np.ndarray = np.zeros((3, steps))
        # print(path)
        for i in range(steps):
            x, y, z = np.random.rand(3)
            sgnX: float = (x - 0.5) / abs(x - 0.5)
            sgnY: float = (y - 0.5) / abs(y - 0.5)
            sgnZ: float = (z - 0.5) / abs(z - 0.5)
            a = np.array([step * sgnX, step * sgnY, step * sgnZ])
            path[:, i] = path[:, i - 1] + a
            # print(path)
        return path

    @staticmethod
    def rev(l: A):
        if len(l) == 0: return []
        return [l[-1]] + CoordinatesGenerator.rev(l[:-1])


class NumpyTrajectoryGenerator(CoordinatesGenerator):
    T = TypeVar("T", np.ndarray, List)
    def __init__(self, steps: int, step: float):
        super().__init__()
        self.rangeLinksRechts: Tuple = (-0.3, 0.3)
        self.rangeUpDown: Tuple = (0.4, 0.7)
        self.rangeForeBack: Tuple = (0.1, 0.4)
        self._trajectoryCoordinates: np.ndarray = self._trajectoryCorrections(steps, step) # man kann sich aussuchen welcher Datentyp aus den in T definierten Datentypen hier angegeben werden sollte
        self._data_len: int = len(self._trajectoryCoordinates)

    def __getitem__(self, item: int):
        return self._trajectoryCoordinates[item]

    def __len__(self):
        return self._data_len

    def __iter__(self):
        for item in self._trajectoryCoordinates:
            yield item

    def __repr__(self):
        return tabulate(self._trajectoryCoordinates, headers=["X_Coordinates", "Y_Coordinates", "Z_Coordinates"], showindex="always", tablefmt="github")
    @staticmethod
    def __searchInArray(arr: np.ndarray, item:float) -> bool:
        for itemArr in arr:
            if itemArr == item:
                return True
            else:
                return False

    def _replaceArraysItems(self, arr: np.ndarray, minValue: float, maxValue: float) -> np.ndarray:
        """search for the values which are smaller that a min value and bigger that a max value, selecting the index of those values
         Then, it returns a np.ndarray with the changed values that exided the min and max values """

        foundIndex: np.ndarray = np.where((arr <= minValue) | (arr >= maxValue))
        #print(foundIndex)
        for index in range(len(foundIndex[0])):

            newValue: float = self.range_selector(minValue + 0.2, maxValue)

            while self.__searchInArray(arr, newValue):
                newValue = self.range_selector(minValue + 0.2, maxValue)


            arr[foundIndex[0][index]] = newValue


        return arr


    def _trajectoryCorrections(self, steps: int, step: float) -> T:
        """takes a number of colums (steps) and the distance of the values in the generated numpy array. it converts the steps in the number of rows of the output array
         It returns a array with a number of rows equal to steps and a predeterminated number of rows as X,Y,Z"""
        points: List = [self.generator(steps, step) for _ in range(1)]
        points_reverse: np.ndarray = self.rev(points)
        points: np.ndarray = np.concatenate((points, points_reverse), axis=1)

        #print("POINTS:", points)
        #print(points.shape)  # (1, 6, 20) first array, arrows, columns

        trajectories: List = [(point[0, 0:steps], point[1, 0:steps], point[2, 0:steps]) for point in points]

        #print(trajectories)
        #print("LEN DER ERSTEN LIST[ARRAYS]:", len(trajectories[0]), len(trajectories[0][1]))

        output: np.ndarray = np.zeros((steps, len(trajectories[0])), dtype=np.float_)

        LinkRechts: np.ndarray = self._replaceArraysItems(trajectories[0][0], self.rangeLinksRechts[0], self.rangeLinksRechts[1])
        UpDown: np.ndarray = self._replaceArraysItems(trajectories[0][1], self.rangeUpDown[0], self.rangeUpDown[1])
        ForeBack: np.ndarray = self._replaceArraysItems(trajectories[0][2], self.rangeForeBack[0], self.rangeForeBack[1])

        for i in range(len(LinkRechts)):
            output[i: i + 1] = [LinkRechts[i], UpDown[i], ForeBack[i]]

        return output


if __name__ == "__main__":
    generator: NumpyTrajectoryGenerator = NumpyTrajectoryGenerator(20, 0.1)
    print(generator._trajectoryCoordinates)
    print(generator)
    print(len(generator))
    # for item in range(len(generator)):
    #     for i in range(len(generator[item])):
    #         if (i > 0):
    #             break
    #         else:
    #             print(i, generator[item][i], generator[item][i+1], generator[item][i+2])

