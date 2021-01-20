
"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
Simulation Main Loop
"""


######################################################################
########################## Import modules  ###########################
######################################################################


import numpy as np



import sys
sys.path.insert(0, "~/PycharmProjects/projectPy3.8/iclub_simulator_tools/python_scripts/")

from hand_trajectory_tracker import *
from tabulate import tabulate
from typing import List, Dict, Tuple, TypeVar, Callable
from numpy import ndarray
from Python_libraries.iCubSim_world_controller import WorldController

################ Import parameter from parameter file ################
from examples.example_parameter import CLIENT_PREFIX, GAZEBO_SIM, ROBOT_PREFIX

############# Import control classes for both simulators #############
if ROBOT_PREFIX or CLIENT_PREFIX:

    import Python_libraries.iCubSim_world_controller as iCubSim_ctrl




yarp.Network.init()  # Initialise YARP

class ObjectMovementExecution:
    A = TypeVar("A", List[str], str)
    def __init__(self, typObj: A ="ssphere") -> None:
        """ create simple objects objects= sbox, scyl oder ssphere"""

        self.__worldController: WorldController = iCubSim_ctrl.WorldController() # WorldController()
        assert type(typObj) == str, 'the object name should be a string'
        if(self.__checkObject(typObj)):
            # create the object with a iCub Object-ID
            self.__object: int = self.__iCubSimObjectID(typObj)


        self.__finalObjectCoordinates: List[List] = [["X Coordinates", "Y Coordinates", "Z Coordinates"],]


    def __repr__(self):
        if (self.__finalObjectCoordinates):
            return tabulate(self.__finalObjectCoordinates, headers=['Right/Left (X)', 'Up/Down (Y)', 'Forward/Backward (Z)'], showindex="always")
        elif not (self.__finalObjectCoordinates):
            return "None Data was saved inside the class"


    def object_movement(self, Xaxis: float, Yaxis: float, Zaxis: float) -> None:
        """moves the Object in 3 Simulation World
        @:parameter: the  X, Y, Z Axis where the object will be moved to"""

        # -------------- move the object -----------------------------
        self.__worldController.move_object(self.__object, [Xaxis, Yaxis, Zaxis])
        print(self.__worldController.get_object_location(self.__object).tolist())

    def getOnGoingObjectCoordinatesFromOutside(self) -> List[float]:
        """ method that could be called from outside the class
        @:return: a list with the location coordinates of the object only"""
        return self.__worldController.get_object_location(self.__object).tolist()

    def deleteAll(self) -> None:
        self.__worldController.del_all()
        del self.__worldController

    def getSaveObjectCoordinatesInsideClassVariable(self) -> List[List]:
        """run the get save object coordinates function
        @return a list of lists with the location coordinates of the object while saving these inside the class
        """
        return self.__getSaveObjectCoordinatesInsideTheClass(self.__object, self.__worldController, self.__saveObjectCoordinatesInsideTheClass())

    def __iCubSimObjectID(self, typObj: str) -> int:
        """
        @:param typObj: a string varaible: "sbox": [1, 0, 0] or "scyl": [0, 0.5, 0.3], "ssphere"
        @:return: digit representing a Object ID
        """
        if(typObj == "sbox"):
            return self.__worldController.create_object(typObj,[0.1, 0.1, 0.1], [0., 0.7, 0.3], [0.4, 0.4, 0.5])
        elif(typObj == "scyl"):
            return self.__worldController.create_object(typObj, [0.1, 0.1], [0., 0.7, 0.3], [0.4, 0.4, 0.5])
        elif(typObj == "ssphere" ):
            return  self.__worldController.create_object(typObj, [0.05], [0., 0.7, 0.3], [0.4, 0.4, 0.5])

    @staticmethod
    def __getSaveObjectCoordinatesInsideTheClass(ObjID: int, __worldController: WorldController, __saveOb: Any) -> List[List]:
        """saves the location coordinates of the object inside the class as final coordinates
        @:param ObjID: a single digit integer
        @:param __worldController: the instance of world controller
        @:param __saveOb: function which saves the location coordinates of the object
        @:return: a list of list of the location coordinates of the object"""
        __saveOb()
        return __worldController.get_object_location(ObjID).tolist()

    def __saveObjectCoordinatesInsideTheClass(self) -> None:
        """saves the location coordinates of the object only"""

        self.__finalObjectCoordinates.append(self.__worldController.get_object_location(self.__object).tolist())

    @staticmethod
    def __checktype(obj: List[Any]):
        """
        check if the list is a list of strings or not
        @:param obj: a list of any data structure
        @:return: a bool
        """
        return bool(obj) and all(isinstance(elem, str) for elem in obj)

    @staticmethod
    def __checkObject(obj: str):
        '''
        check if the string belongs to one of the predeterminated string names
        @:param obj: a string name
        @:return: a bool value
        '''
        predertiminatedObjects: List[str] = ["sbox", "scyl", "ssphere"]
        if(obj in predertiminatedObjects):
            return True
        return False

if __name__ == '__main__':
    pass