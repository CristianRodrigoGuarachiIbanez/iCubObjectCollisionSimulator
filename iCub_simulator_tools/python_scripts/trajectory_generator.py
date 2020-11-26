"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
3dNumpy trajectory generator
"""

import numpy as np
from random import choice#
from typing import List, Dict, TypeVar, Any, Tuple, Callable
from tabulate import tabulate
from numpy_trajectory_generator import NumpyTrajectoryGenerator
import random

class CoordinatesGenerator:

    def __init__(self, start: float, end: float, stepsize: float, num: int = 10, dtyp: Callable = np.float_, linspace: bool = False):

        self._start: float = start
        self._end: float = end
        self._num: int = num
        self._stepsize: float = stepsize
        self._dtyp: Callable = dtyp
        self._linspace: bool = linspace

        self._linspaceTrajectory, self._linsStepsize = self._CreateLinspaceTrajectory() # type: np.ndarray, float
        self._arangeTrajectory: np.ndarray = self._CreateArrayTrajectory()

    def __getitem__(self, item: int):
        if(self._linspace):
            return self._linspaceTrajectory[item]
        return self._arangeTrajectory[item]

    def __len__(self):
        if (self._linspace):
            return len(self._linspaceTrajectory)
        return len(self._arangeTrajectory)

    # def __iter__(self):
    #     for item in self._linspaceTrajectory:
    #         yield item

    def __repr__(self) -> str:
        if(self._linspace):
            return str(self._linspaceTrajectory)
        return str(self._arangeTrajectory)

    def _CreateArrayTrajectory(self) -> np.ndarray:
        arr: np.ndarray = np.arange(self._start, self._end, self._stepsize, dtype=self._dtyp)
        arr: np.ndarray = np.resize(arr, (1, self._num))
        arr: np.ndarray = arr.flatten()
        return arr

    def _CreateLinspaceTrajectory(self) -> Tuple[np.ndarray, float]:
        arr, stepsize = np.linspace(self._start, self._end, num=self._num, dtype=self._dtyp, retstep= True) # type: np.ndarray, float
        #print(x)
        return arr, stepsize

class DirectionSelector:

    def __init__(self, right: bool = True):

        self._right: bool = right

        self._starts: NumpyTrajectoryGenerator = NumpyTrajectoryGenerator(20, 0.06666666666666667)
        self._rightLeft: np.ndarray = self._starts[:, 0:1].flatten()

        self._upDown: np.ndarray = self._starts[:,1:2].flatten()
        self._foreBack: np.ndarray = self._starts[:, 2:].flatten()


    def _StartFromRight(self) -> float:
        """fom -0.3 to 0."""
        startRight: List[float] = [x for x in self._rightLeft if x <= 0]
        if(startRight):
            return float(random.choice(startRight))
        return random.choice(np.linspace(-0.3, 0.0, 5))

    def _StartFromLeft(self) -> float:
        """ from 0 t0 0.3"""
        startLeft: List[float] = [x for x in self._rightLeft if x >= 0]
        if(startLeft):
            return float(random.choice(startLeft))
        return random.choice(np.linspace(-0., 0.3, 5))

class StartSelector(DirectionSelector):
    def __init__(self, right: bool, upDOwn: bool = False):
        super().__init__(right)
        self._upDown: bool = upDOwn

    def __float__(self) -> float:
        if (self._right) and not (self._upDown):
            return self._StartFromRight()

        elif not (self._right) and not (self._upDown):
            return self._StartFromLeft()

        elif not (self._right) and (self._upDown):
            return self._StartFromUPDown()

        elif (self._right) and (self._upDown):
            return self._StartFromForeBack()


    def _StartFromUPDown(self) -> float:
        return float(random.choice(self._upDown))

    def _StartFromForeBack(self) -> float:
        return float(random.choice(self._foreBack))



class IntervalGenerator:

    def __init__(self, start: int, stop: int, step: int):
        self._start: int = start
        self._stop: int = stop
        self._step: int = step

    def IndexSelector(self):

        indexes: List[int] = [x for x in range(9,1000,10)]
        return  indexes





if __name__ == "__main__":
    generator: CoordinatesGenerator = CoordinatesGenerator(-0.3, 0.3, stepsize=0.0006006006006006006, num=10, linspace=False)
    print("arange:",generator._arangeTrajectory)
    print(generator._linsStepsize)
    print("linspace:",generator._linspaceTrajectory)
    # i: int = 0
    # for item in generator:
    #     print(i,item)
    #     i += 1
    start: StartSelector = StartSelector(False, False)
    print(float(start))





