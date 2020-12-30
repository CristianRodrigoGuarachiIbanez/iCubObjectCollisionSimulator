"""
Created on Fr Dic 4 2020
@author: Cristian Rodrigo Guarachi Ibanez

Head new coordinates
"""


import sys
import numpy as np
import yarp
from typing import Any, List, Tuple, TypeVar
############ Import modules with specific functionalities ############
import Python_libraries.YARP_motor_control as mot
from Python_libraries.YARP_motor_control import motor_init, get_joint_position, motor_init_cartesian

################ Import parameter from parameter file ################
from examples.example_parameter import CLIENT_PREFIX, ROBOT_PREFIX

######################################################################
######################### Init YARP network ##########################
######################################################################
#mot.motor_init("head", "position", ROBOT_PREFIX, CLIENT_PREFIX)
yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')

class MoveHead:
    N = TypeVar("N", List, np.ndarray)
    def __init__(self, newCoord: N ):



        print('---------------- Init motor control for the head  ----------------')
        self.__iCtrl, self.__iEnc, self.__jnts, self.__driver = motor_init("head")

        self.__newPos: yarp.Vector = yarp.Vector(self.__jnts)

        print("------------------------Go to head zero position------------------")
        mot.goto_zero_head_pos(self.__iCtrl, self.__iEnc, self.__jnts)

        print("-------------------- Print head joints position ----------------------")
        self.__initialHeadJointsPosition: List[List] = [["HorizontalRotationn, LateralTranslation, VerticalRotation", "GazeDirectionVerti", "EyeOpossiteVertRotation", "GazeDIrectionHor"],
                                                        ]
        if not (newCoord):
            print("Usage of the class is appropiate for countinously moving the head")
        elif (newCoord):
            print("Usage of the class takes appropiately just a final coordinates")

            if(isinstance(newCoord, list)):
                self.__newCoord: np.ndarray = np.array(newCoord)
            self.__moveToGoalCoordinatesInsideTheClass(newCoord)


    def __repr__(self) -> str:
        return str(self.__initialHeadJointsPosition)

    def moveHeadToNewPositionFromOutside(self, newCoord: List[float]) -> None:
        """takes the head to the new position coordinates"""
        self.__moveToGoalCoordinatesInsideTheClass(newCoord)

    def getHeadJointsCoordinatesFromOutside(self):
        """ @returns the head eye coordinates OUTSIDE THE CLASS only"""
        return self.__getHeadJointsCoordinatesInsideClass()

    def closing_programm(self) -> None:
        """ finishes the yarp conection """
        print('--------------- Close control devices and opened ports --------------')
        self.__driver.close()
        yarp.Network.fini()

    def getSaveHeadJointsCoordinatesFromOutside(self) -> np.ndarray:
        """ @returns the Head Eye Coordinates as a numpy array OUTSIDE THE CLASS while saving the position coordinates into the class's constructor """
        return self.__getSaveHeadJointsCoordinates(self.__initialHeadJointsPosition, self.__getHeadJointsCoordinatesInsideClass(), self.__saveHeadJointsCoordinatesInsideClass())

    @staticmethod
    def __getSaveHeadJointsCoordinates(initList: List[List], __getCoord: np.ndarray, __save: Any) -> np.ndarray:
        """ @returns the Head Eye Coordinates as a numpy array INSIDE THE CLASS while saving the position coordinates into the class's constructor """
        initList.append(__getCoord.tolist())
        return __getCoord

    def __saveHeadJointsCoordinatesInsideClass(self) -> None:
        """ saves the new coordinates into the class's constructor """
        self.__initialHeadJointsPosition.append(self.__getHeadJointsCoordinatesInsideClass())

    def __moveToGoalCoordinatesInsideTheClass(self, newCoord: List[float], motion: bool = False) -> None:
        """1.- Move the head to predefined position"""
        # x:upDown, y: rightLeft \|/ z: rotation RightLeft, a: eyedirection (same direction) leftRight, ag: eyedirection (opposite directio)  inside/outside
        #pos: np.ndarray = np.array([-40., -10., -20., 0., 0., 10.])
        if not (len(newCoord) > 0):
            raise AssertionError()
        elif(isinstance(newCoord, List)):
            newCoord: np.ndarray = np.array(newCoord)
        print("---------------- Move the head to predefined position ---------------")
        for i in range(self.__jnts):
            self.__newPos.set(i, newCoord[i])
        self.__iCtrl.positionMove(self.__newPos.data())

        # optional, for blocking while moving the joints
        #motion = False
        while not motion:
            motion = self.__iCtrl.checkMotionDone()

    def __getHeadJointsCoordinatesInsideClass(self) -> np.ndarray:
        """read the head joints coordinates inside the class
        @returns a np array with the coordinates of the head and eye direction"""

        self.__iEnc.getEncoders(self.__newPos.data())
        vector: np.ndarray = np.zeros(self.__newPos.length(), dtype=np.float64)
        for i in range(self.__newPos.length()):
            vector[i] = self.__newPos.get(i)
        #print(vector)
        return vector

if __name__ == "__main__":
    # pos: np.ndarray = np.array([-35., -10., -25., 0., 0., 10.])
    # head: MoveHead = MoveHead(pos)
    # for _ in range(10):
    #     print(head.getHeadJointsCoordinatesFromOutside())
    pass