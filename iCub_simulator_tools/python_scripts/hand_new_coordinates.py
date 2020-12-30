"""
Created on Fr Dic 4 2020
@author: Cristian Rodrigo Guarachi Ibanez

hand new coordinates
"""


#############################run previously ##########################
#bash start_environment.sh
#bash start_cartesian_control_modules.sh left_arm right_arm

########################## Import modules  ###########################
######################################################################

import sys
import time

import numpy as np
import yarp
from hand_tracker.hand_trajectory import ford_and_backwards
from typing import List, Dict, Tuple, Any, TypeVar, Callable
#from export_arm_movement_data import Writer
from tabulate import tabulate
from hand_tracker.joints_tracker.joints_trajectory_tracker import jointTrajectory
############ Import modules with specific functionalities ############
import Python_libraries.YARP_motor_control as mot
from Python_libraries.YARP_motor_control import motor_init, get_joint_position, motor_init_cartesian

################ Import parameter from parameter file ################
from examples.example_parameter import (Transfermat_robot2world, Transfermat_world2robot,
                                orientation_robot_hand, pos_hand_world_coord, pos_hand_world_coord_new,
                                CLIENT_PREFIX, ROBOT_PREFIX)

######################################################################
######################### Init YARP network ##########################
######################################################################

yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')

class HandMovementExecution:
    L = TypeVar("L", List[float], np.ndarray)
    T = TypeVar("T", List, Dict)
    def __init__(self, ArmSide: str = "right_arm", CurrCoordinates: L = None):

        # --------------------------------------------------------------
        # ------------------- Init Arm Side and Joints -----------------
        if (ArmSide == "right_arm"):
            self.__HandSide: Tuple = "Right Hand", "RightHand"
        elif (ArmSide == "left_arm"):
            self.__HandSide: Tuple = "Left Hand", "LeftHand"

        # ------------------------------------------------------------------
        # print('-------------------- Init YARP variables ---------------------')
        yarp.Network.init()
        self._T_r2w: Transfermat_robot2world = Transfermat_robot2world
        self._T_w2r: Transfermat_world2robot = Transfermat_world2robot

        self._orient_hand: yarp = mot.npvec_2_yarpvec(orientation_robot_hand)

        self._pos: yarp = yarp.Vector(3)
        self._orient: yarp = yarp.Vector(4)

        # ------------------ Init cartesian controller for arm --------------
        # ----------------- Prepare a property object --------------------
        # print("-------------- Create remote driver: Driver-----------------")
        # print("------------- Query motor control interfaces: iCart and iEnc----------")

        self.__iCart, self.__driver = motor_init_cartesian(ArmSide)  # "right_arm" or "left_arm"
        # print(f"{self.__HandSide[0]} (driver, iCart):", self.__driver, self.__iCart)
        time.sleep(1)

        # ------------------------------ create a holder variable ---------------------
        try:
            self.__HandCoordinatesHolder: List[List[Dict]] = [self.__initHandCoordinates()]
        except Exception as e:
            print(e)

        if not (CurrCoordinates):
            print("Usage of the class is appropiate for countinously moving the handd")
        elif (CurrCoordinates):
            print("Usage of the class takes appropiately just a final coordinates")
            if(isinstance(CurrCoordinates, List)):
                    CurrCoordinates: np.ndarray = np.array(CurrCoordinates)
            self.__MoveToGivenPositionCoordinates(CurrCoordinates)

    def getHandFinalCoodinatesFromInside(self) -> List[List[Dict]]:
        """@returns the final list (holder) of saved dicts (coordinates)  """
        return self.__HandCoordinatesHolder

    def moveHandToNewPositionCooridnatesfromOuside(self, NewCoord: List[float]) -> None:
        """ take the hannd to new position coordinates
        @parameters list of floating numbers """
        self.__MoveToGivenPositionCoordinates(NewCoord)

    # def MoveToDestinationCoordinatesInsideTheClass(self, pos_hand_world_coord_new: L) -> Dict[str,str]:
    #     """ @parameters recieves a new set of coordinates to execute a movement and save these into the holder at the constructor
    #     @returns the current registered coordinates """
    #     self.__MoveToGivenPositionCoordinates(pos_hand_world_coord_new)
    #     return self.__getSaveOnGoingCoordinatesInsideTheClass(self.__readOnGoingHandCoordinates(), self.__saveOnGoingHandDataInsideTheClass())

    def readOnGoingHandCoordinatesFromOutside(self):
        """ @returns a Dict with the current world coordinates of the position and orientation of the hand """
        return self.__readOnGoingHandCoordinates()

    def closingProgramm(self) -> None:
        """ finishes the Yarp programm"""
        print('--------------- close control devices and opened ports [Hand] ---------------')
        self.__driver.close()
        yarp.Network.fini()

    @staticmethod
    def __getSaveOnGoingCoordinatesInsideTheClass(__readHandCoord: Dict, __saveData: Any) -> Dict[str,str]:
        """saves the location coordinates of the hand inside the class as holder
        @parameter callable function which gives the position and orientation coordinates of the hand
        @returns a dict of strings with the position and orientation coordinates of the hand"""
        __saveData()
        return __readHandCoord

    def __saveOnGoingHandDataInsideTheClass(self) -> None:
        """saves the current coordinates into the holder at the constructor"""
        self.__HandCoordinatesHolder.append(self.__readOnGoingHandCoordinates())

    def __readOnGoingHandCoordinates(self) -> Dict[str, str]:
        """0 .- get the position and orientation of the hand vectors
        @returns a Dict of strings with the coordinates of the hand vectors """
        self.__iCart.getPose(self._pos, self._orient)
        # print(f'{self.__HandSide[0]} Position:', self._pos.toString())
        # print(f'{self.__HandSide[0]} Orientation:', self._orient.toString())
        HandPos, HandOri = self._pos.toString(), self._orient.toString()  # type: str, str
        #return {f"{self.__HandSide[1]}Posi": HandPos, f"{self.__HandSide[1]}Orient": HandOri}
        return {f"{self.__HandSide[1]}Posi": ",".join(x for x in HandPos.split("\t")), f"{self.__HandSide[1]}Orient": ",".join(x for x in HandOri.split("\t"))}

    def __initHandCoordinates(self) -> List[Dict]:
        """" 1.- Move hand to Initial Coordinates (position and orientation)
        @returns a list of ditionary with the initial coordinates where the hand have to be moved to """
        # print('--------------------- Hand Movement to Initial Pose ---------------')
        HandCoord: List[Dict] = []
        #JointsCoord: List[List] = []
        welt_pos = pos_hand_world_coord
        init_hand_pos_np = np.dot(self._T_w2r, welt_pos.reshape((4, 1))).reshape((4,))
        init_hand_pos_yarp = mot.npvec_2_yarpvec(init_hand_pos_np[0:3])

        self.__iCart.goToPoseSync(init_hand_pos_yarp, self._orient_hand)
        self.__iCart.waitMotionDone(timeout=5.0)
        time.sleep(0.5)
        HandCoord.append(self.__readOnGoingHandCoordinates())
        return HandCoord

    def __MoveToGivenPositionCoordinates(self, pos_hand_world_coord_new: L) -> None:
        """2.-  Move hand to the new given position
            @parameter recive a new array or list with X,Y,Z Coordinates
            @returns a List with Dict of the coordinates of the given arm side """

        # print("---------------------- Hand Movement to the New Static Coordinates ----------------------")
        if not (len(pos_hand_world_coord_new) > 0):
            raise AssertionError()
        elif(isinstance(pos_hand_world_coord_new, List)):
             pos_hand_world_coord_new: np.ndarray = np.array(pos_hand_world_coord_new) #np.array([-0.111577, 0.27158, 0.501089, 0.1])

        welt_pos_n: np.ndarray = pos_hand_world_coord_new   # erster: links/rechts, zweiter: oben/unten, dritter: vorn/hinten
        new_hand_pos_np: np.dot = np.dot(self._T_w2r, welt_pos_n.reshape((4, 1))).reshape((4,))
        new_hand_pos_yarp: mot.npvec_2_yarpvec = mot.npvec_2_yarpvec(new_hand_pos_np[0:3])

        # ----------------------------------------------------------------
        # ------------- Get Hand to Position/Orientation -----------------
        self.__iCart.goToPoseSync(new_hand_pos_yarp, self._orient_hand)
        self.__iCart.waitMotionDone(timeout=5.0)
        # --------------------------------------------------------------------
        #print(self.__readOnGoingHandCoordinates())



class GettingOnGoingJointCoordinates:
    def __init__(self, ArmSide: str = "left_arm"):
        # ------------------- Init Arm Side and Joints -----------------
        if (ArmSide == "right_arm"):
            self.__HandSide: Tuple = "Right Hand", "RightHand"
        elif (ArmSide == "left_arm"):
            self.__HandSide: Tuple = "Left Hand", "LeftHand"

        #---------------------Init encoder for get_join_position ---------
        # --------------------- Init encoder for get join position ----------------
        # print('---------------- Init arm joint position ----------------')
        self.__iCtrl, self.__iEnc, self.__jnts, self.__driver = motor_init(ArmSide)
        if not self.__driver:
            sys.exit("Motor initialization failed!")

        # print(f"{self.__HandSide[0]} (iCtrl, iEnc, jnts):") # self.__iEnc, self.__jnts)
        # print("iCtrl:", self.__iCtrl,"iEnc:", self.__iEnc, "jnts:", self.__jnts)

    def readOnGoingJointsCoordinates(self) -> np.ndarray:
        """ @returns a List of Lists with the coordinates of the arm's joints """
        return get_joint_position(self.__iEnc, self.__jnts, as_np=True)





def __ReadOnGoingJointCoordinates(ArmSide: str = "left_arm"):
    iCtrl, iEnc, jnts, driver = motor_init(ArmSide)
    # print(f"{ArmSide} (iEnc, jnts):")  # self.__iEnc, self.__jnts)
    # print("iEnc:", iEnc, "jnts:", jnts)
    return get_joint_position(iEnc, jnts, as_np=True)


if __name__ == "__main__":
    # continuous: List[float] = [0.111577, 0.27158, 0.501089, 0.1]
    #
    # moving: HandMovementExecution = HandMovementExecution(ArmSide="left_arm", CurrCoordinates=continuous)
    # a = []
    # lista = []
    # for i in range(10):
    #     print(i)
    #     print()
    #     moving.MoveToDestinationCoordinatesInsideTheClass([0.11 * i, 0.27 * i, 0.56 * i, 0.1])
    #     lista.append(moving.readOnGoingHandCoordinatesFromOutside())
    #     a.append(__ReadOnGoingJointCoordinates())
    #
    # print("reading from class:", moving.getHandFinalCoodinatesFromInside())
    # moving.closingProgramm()
    # print("reading from list",lista)
    # print(a)
    pass



    #output: Writer = Writer(handMov, output="position")
    #1.- liefert eine Tabelle der ausgewählten Daten Position/Orientation aus dem __repr__ zurück
    #print(output)
    #2.- liefert die Flugbahn der Gelenke zurück
    #print(hand_mov.joints_trajectories)
    #3.- liefert die DataFrame von Position oder Orientation züruck
    #print(output.data)





