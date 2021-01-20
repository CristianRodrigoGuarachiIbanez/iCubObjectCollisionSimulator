"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
Simulation Main Loop
"""


######################################################################
########################## Import modules  ###########################
######################################################################

import os
import time
import numpy as np

import yarp

import sys
sys.path.insert(0, "~/PycharmProjects/projectPy3.8/iclub_simulator_tools/DGP_iCubSimulator/DGP_iCubSimulator")

from hand_trajectory_tracker import *
from typing import List, Dict, Tuple, TypeVar, Callable
from numpy import ndarray


from visual_perception_tracker import VisualPerception
from tactil_perception_tracker import TactilCollisionDetector
from export_data.export_static_coordinates_data import HandCoordinatesWriter, HeadCoordinatesWriter, SensorCoordinatesWriter, ObjectCoordinatesWriter, camerasCoordinatesWriter, GroundTruthWriter
from scene_recorder import SceneRecorder
from trajectory_generator import CoordinatesGenerator, StartSelector, TrajectoryGenerator, __regulatingValuesEqualToLimit

from object_movement_execution import ObjectMovementExecution
from hand_new_coordinates import HandMotionExecutor, __ReadOnGoingJointCoordinates, GettingOnGoingJointCoordinates
from head_new_position import MoveHead
from groundTruthGenerator import GroundTruthGenerator
import logging





class SimulationEventModulator:
    def __init__(self, object: str, armSide: str, newHandCoord: List[float], newHeadCoord: List[float]):

        # -------------------------- Object Movement---------------------------------------------
        print(" -------------- init object movement executer and tracker-------------------")
        self.__object: ObjectMovementExecution = ObjectMovementExecution(object) # default sphere

        # ------------------------------ Hand's position and Orientation ------------------------------
        print(" --------------- init hand movement executer and tracker ------------------")
        self.__handMovement: HandMotionExecutor = HandMotionExecutor(ArmSide=armSide, CurrCoordinates=newHandCoord)

        # ------------------------------ joints Position --------------------------------------------
        self.__jointsPosition: GettingOnGoingJointCoordinates = GettingOnGoingJointCoordinates(armSide)

        # ------------------------------ Head's and eyes position ------------------------------------
        print("------------------- init head movement executer and tracker ---------------------")
        self.__newHeadCoordinates: MoveHead = MoveHead(newHeadCoord)

        # ------------------------------ sensors for the skin perception -----------------------------------
        print(" ------------------ init collision detector -----------------")
        self.__collisionDect_hand: TactilCollisionDetector = TactilCollisionDetector(armSec=0)
        self.__collisionDect_forearm: TactilCollisionDetector = TactilCollisionDetector(armSec=1)
        self.__collisionDect_arm: TactilCollisionDetector = TactilCollisionDetector(armSec=2)

        # ----------------------------- scene and visual perception cameras ----------------------------------------
        print("-------------- Init visual perception----------- ")
        self.__visualPerc: VisualPerception = VisualPerception()

        print("---------------init scene recorder---------------")
        self.__sceneRecorder: SceneRecorder = SceneRecorder()

    def translationObjectCoordinatesFromOutside(self, X: float, Y: float, Z: float) -> None:
        """
        moves the generated object to the new given coordinates
         @:param X: a single floating number
         @:param Y: a single floating number
         @:param Z: a single floating number
        """
        self.__object.object_movement(X,Y,Z)
    def translationHandCoordinatesFromOutside(self, newCoord: List[float]) -> None:
        """
        moves the Hand to new given coordinates
         @:param newCoord: a list of floats containing the coordinates of the hand to be moved
        """
        self.__handMovement.moveHandToNewPositionCooridnatesfromOuside(newCoord)

    def translationHeadCoordinatesFromOutside(self, newCoord: List[float]) -> None:
        """
        moves the Head to new given coordinates
        @:param newCoord: a list of floats containing the coordinates of the head to be moved
        """
        self.__newHeadCoordinates.moveHeadToNewPositionFromOutside(newCoord)

    def startObjectCoordinatesFromOutside(self) -> List[float]:
        """
        gives the first coordinates for the object and starts the recollection of the current coordinated of the object
        @:return: a list of list with the location coordinates (X,Y,Z) of the object
        """
        return self.__object.getOnGoingObjectCoordinatesFromOutside()

    def startHandJointsCoordinatesFromOutside(self) -> Dict[str,str]:
        """
        pass the first position coordinates for the hand and begins the gathering of hand coordinates
        @:return: a dict with the hand location and orientation coordinates
        """
        return self.__handMovement.readOnGoingHandCoordinatesFromOutside()

    def startJointsCoordinatesFromOutside(self) -> List[float]:
        """
        @:return: a list of floating numbers
        """
        return self.__jointsPosition.readOnGoingJointsCoordinates()

    def startVisualPerceptionCoordFromOutside(self, index: int = None) -> Tuple[ndarray, ndarray]:
        """
        starts the recording process on the eye cameras
        @:return: a tuple of numpy array matrixes with the visual data of the perceived object
        """
        return self.__visualPerc.readCameraImagesFromOutside(index)
    def startSceneRecorderCoordsFromOutside(self, index: int = None) -> ndarray:
        """
        starts the recording process on the scene camera
        @:return: a numpy array matrix with the visual data (coordinates) of the perceived """
        return self.__sceneRecorder.readSceneCamera(index)
    def startCollisionSensorDataFromOutside(self) -> Tuple[Tuple,Tuple,Tuple]:
        """
        starts the activity over the skin of the robot
        @:return: a tuple of tuples with the corresponds lists for right and left collision coordinates"""
        return self.__collisionDect_hand.skin_sensor_reader(), self.__collisionDect_forearm.skin_sensor_reader(), self.__collisionDect_arm.skin_sensor_reader()

    def startHeadEyeCoordinatesFromOutside(self) -> List[float]:
        """
        pass the first position coordinates of the head and eyes and get the correspondenly data
        @:return: a List of floats with the coordinates of the head position"""
        headEyeCoord: List = self.__newHeadCoordinates.getHeadJointsCoordinatesFromOutside().tolist()
        return headEyeCoord


    def __finishVisualPerceptionConnToYARPFromImside(self) -> None:
        """ Closing the connection with the eyes camaras
        """
        self.__visualPerc.closing_program()
    def __finishSceneRecorderConnToYARPFromInside(self) -> None:
        """ Closing the connection with the scene cameras"""
        self.__sceneRecorder.closing_program()
    def __finishCollisionSensorDataConnToYARPFromInside(self) -> None:
        """ Closing the connection with the collision sensors of the skin"""
        self.__collisionDect_hand.closing_programm(), self.__collisionDect_forearm.closing_programm(), self.__collisionDect_arm.closing_programm()
    def __finishHandMovementConnToYARPFromInside(self) -> None:
        """  Closing the YARP connection with hand coordinates"""
        self.__handMovement.closeYARPProgrammHand()

    def __finishJointsDataReadingToYARPFromInside(self) -> None:
        """ closing the YARP Connection"""
        self.__jointsPosition.closeYARPProgrammJoints()
    def __finishObjectMovementConnToYARPFromInside(self) -> None:
        self.__object.deleteAll()

    def __finishHeadMovementToYARPFromInside(self) -> None:
        """  Closing the YARP conexcion with head coordinates"""
        self.__newHeadCoordinates.closing_programm()

    def finishProgrammConectionFromOutside(self) -> None:
        """
        close the conection to yarp
        """
        self.__finishHeadMovementToYARPFromInside()
        self.__finishHandMovementConnToYARPFromInside()

        self.__finishJointsDataReadingToYARPFromInside()
        self.__finishObjectMovementConnToYARPFromInside()
        self.__finishCollisionSensorDataConnToYARPFromInside()
        self.__finishSceneRecorderConnToYARPFromInside()
        self.__finishVisualPerceptionConnToYARPFromImside()


def __initializeArmSide(ArmSide: bool) -> Tuple[float, float, str, List[float], List[float]]:
    start_X, end_X, armSide, NewHandCoord, NewHeadCoord = None, None, None, None, None # type: float, float, str, List[float], List[float]
    if (ArmSide):
        start_X = __regulatingValuesEqualToLimit(0.0) #StartSelector(ArmSide, False)  # handSide == True -> right hand
        end_X = 0.0
        armSide = "right_arm"
        NewHandCoord = [-0.111577, 0.27158, 0.501089, 0.1]
        NewHeadCoord = [-35., -10., -30., 0., 0., 10.]

    elif not (ArmSide):
        start_X = __regulatingValuesEqualToLimit(0.3) #StartSelector(ArmSide, False)  # handSide == False -> left Hand
        end_X = 0.3
        armSide = "left_arm"
        NewHandCoord = [0.111577, 0.27158, 0.501089, 0.1]
        NewHeadCoord = [-35., 10., 30., 0., 0., 10.]

    return start_X, end_X, armSide, NewHandCoord, NewHeadCoord


if __name__ == '__main__':
    pass

    # # ----------------------- create the directory wherein the imgs will be saved ---------------------------
    # path: str = os.path.dirname(os.path.abspath(__file__)) + "/trials"
    # if not os.path.exists(path):
    #     os.mkdir(path)
    #     pathBinocular: str = os.path.dirname(os.path.abspath(__file__)) + "/trials/image_outputs"
    #     pathScene: os.path = os.path.dirname(os.path.abspath(__file__)) + "/trials/scene_outputs"
    #
    #     if not os.path.exists(pathBinocular):
    #         os.mkdir(pathBinocular)
    #
    #
    #     elif not os.path.exists(pathScene):
    #         os.mkdir(pathScene)
    #
    #
    # # parameters for the trajectory
    # ArmSide: bool = True
    # steps: int = 5 ######
    #
    # # ---------------------- Anzahl der Events: Objektbewegung, Augenbilder, Scenebilder ------------------------
    # start_X, end_X, armSide, NewHandCoord, NewHeadCoord = __initializeArmSide(ArmSide)  # type: float, float, str, List[float], List[float]
    # start_Y: float = __regulatingValuesEqualToLimit(0.7, False, True ) #StartSelector(False, True)
    # start_Z: float = __regulatingValuesEqualToLimit(0.3, True, True) #StartSelector(True, True)
    #
    # X: TrajectoryGenerator = TrajectoryGenerator(start_X, end_X, stepsize=0.05, dtyp=np.float_, num=steps, linspace=True)
    # Y: TrajectoryGenerator = TrajectoryGenerator(start_Y, 0.8, stepsize=0.05, dtyp=np.float_,  num=steps, linspace=True)
    # Z: TrajectoryGenerator = TrajectoryGenerator(start_Z, 0.4, stepsize=0.05, dtyp=np.float_, num=steps, linspace=True)
    # X_trajectory: ndarray = X.ConcatenateTrajectoryArrays()
    # Y_trajectory: ndarray = Y.ConcatenateTrajectoryArrays()
    # Z_trajectory: ndarray = Z.ConcatenateTrajectoryArrays()
    #
    # # ------------------------------ instantiate the main class ----------------------------------
    # # "sbox", "scyl", "ssphere"
    # dataGeneration: SimulationEventModulator = SimulationEventModulator(object= 'sbox', armSide = armSide, newHandCoord = NewHandCoord, newHeadCoord = NewHeadCoord)
    #
    # # ----------------------------- initial coordinates values --------------------------------
    # tracked_collision_hand_right: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[0][0]]
    # tracked_collision_hand_left: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[0][1]]
    #
    # tracked_collision_forearm_right: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[1][0]]
    # tracked_collision_forearm_left: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[1][1]]
    #
    # tracked_collision_arm_right: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[2][0]]
    # tracked_collision_arm_left: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[2][1]]
    #
    # object_coordinates: List[List[float]] = [dataGeneration.startObjectCoordinatesFromOutside()]
    #
    # #joint_coordinates: List[ndarray] = [__ReadOnGoingJointCoordinates(armSide)]
    # joint_coordinates: List[List[float]] = [dataGeneration.startJointsCoordinatesFromOutside()]
    # hand_coordinates: List[Dict] = [dataGeneration.startHandJointsCoordinatesFromOutside()]
    #
    # visual_perception_right: List[ndarray] = [dataGeneration.startVisualPerceptionCoordFromOutside()[0]]
    # visual_perception_left: List[ndarray] = [dataGeneration.startVisualPerceptionCoordFromOutside()[1]]
    # visual_binocular_perception: List[Tuple[ndarray, ndarray]] = [dataGeneration.startVisualPerceptionCoordFromOutside()]
    # scene_records: List[ndarray] = [dataGeneration.startSceneRecorderCoordsFromOutside()]
    #
    # head_eyes_coordinates: List[List] = [dataGeneration.startHeadEyeCoordinatesFromOutside()]
    #
    # # ----------------------- Create the Ground Truth -------------------------------------
    #
    # GroundTruth: GroundTruthGenerator = GroundTruthGenerator()
    # GroundTruthForRightHand: List[Tuple] = []
    # GroundTruthForLeftHand: List[Tuple] = []
    #
    # GroundTruthForRightForearm: List[Tuple] = []
    # GroundTruthForLeftForearm: List[Tuple] = []
    #
    # GroundTruthForRightArm: List[Tuple] = []
    # GroundTruthForLeftArm: List[Tuple] = []
    #
    # # ---------------------- Parameters for the while loop ---------------------------------
    # start: int = 9
    # STEPSIZE: int = 10
    # numPause: int = len(X_trajectory)
    # PAUSE: List[int] = [x for x in range(start, numPause, STEPSIZE)]
    # print(PAUSE)
    # i: int = 0
    # trialsCounter: int = 0
    # while True:
    #
    #     try:
    #         logging.info("   ", f"index:{i},left hand:{len(tracked_collision_hand_left)}, left forearm:{len(tracked_collision_forearm_left)}, left arm:{len(tracked_collision_arm_left)},"
    #                      f" right hand:{len(tracked_collision_hand_right)}, right forearm: {len(tracked_collision_forearm_right)}, right arm: {len(tracked_collision_arm_right)} ",
    #                     f" object coord: {len(object_coordinates)}, hand coord:{len(hand_coordinates)}, joints coord:{len(joint_coordinates)}, head coord:{len(head_eyes_coordinates)}")
    #         # -------------- move the object -----------------------------
    #         dataGeneration.translationObjectCoordinatesFromOutside(X_trajectory[i], Y_trajectory[i], Z_trajectory[i])
    #         #iCubSim_wrld_ctrl.move_object(self.__object, [X[i], Y[i], Z[i]])
    #         # print(iCubSim_wrld_ctrl.get_object_location(objID).tolist())
    #
    #         # -------------- Head, Hand and Eyes Coordinates ------------
    #         hand_coordinates.append(dataGeneration.startHandJointsCoordinatesFromOutside())
    #         # joint_coordinates.append(__ReadOnGoingJointCoordinates(armSide))
    #         joint_coordinates.append(dataGeneration.startJointsCoordinatesFromOutside())
    #         head_eyes_coordinates.append(dataGeneration.startHeadEyeCoordinatesFromOutside())
    #
    #         # --------------append visual perception data ---------------
    #         visual_perception_right.append(dataGeneration.startVisualPerceptionCoordFromOutside(i)[0])
    #         time.sleep(0.5)
    #         visual_perception_left.append(dataGeneration.startVisualPerceptionCoordFromOutside(i)[1])
    #         time.sleep(0.5)
    #
    #         # --------------append scene data ----------------------------
    #         scene_records.append(dataGeneration.startSceneRecorderCoordsFromOutside(i))
    #
    #         # ------------- append collision data of the arm skin -----------------------
    #         tracked_collision_hand_right.append(dataGeneration.startCollisionSensorDataFromOutside()[0][0])
    #         time.sleep(0.5)
    #         tracked_collision_hand_left.append(dataGeneration.startCollisionSensorDataFromOutside()[0][1])
    #         time.sleep(0.5)
    #
    #         tracked_collision_forearm_right.append(dataGeneration.startCollisionSensorDataFromOutside()[1][0])
    #         time.sleep(0.5)
    #         tracked_collision_forearm_left.append(dataGeneration.startCollisionSensorDataFromOutside()[1][1])
    #         time.sleep(0.5)
    #
    #         tracked_collision_arm_right.append(dataGeneration.startCollisionSensorDataFromOutside()[2][0])
    #         time.sleep(0.5)
    #         tracked_collision_arm_left.append(dataGeneration.startCollisionSensorDataFromOutside()[2][1])
    #         time.sleep(0.5)
    #
    #         # ------------- append location data of object ------------------------
    #         object_coordinates.append(dataGeneration.startObjectCoordinatesFromOutside())
    #         time.sleep(0.5)
    #
    #         if (i in PAUSE):
    #             time.sleep(2)
    #             #print(tracked_collision_hand_right)
    #             GroundTruthForRightHand.append(GroundTruth.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, tracked_collision_hand_right))
    #             GroundTruthForLeftHand.append(GroundTruth.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, tracked_collision_hand_left))
    #             GroundTruthForRightForearm.append(GroundTruth.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, tracked_collision_forearm_right))
    #             GroundTruthForLeftForearm.append(GroundTruth.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, tracked_collision_forearm_left))
    #             GroundTruthForRightArm.append(GroundTruth.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, tracked_collision_arm_right))
    #             GroundTruthForLeftArm.append(GroundTruth.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, tracked_collision_arm_left))
    #             time.sleep(2)
    #             trialsCounter +=1
    #         else:
    #             pass
    #
    #     except Exception as e:
    #         print(e)
    #         break
    #     except KeyboardInterrupt:
    #         break
    #
    #     i += 1
    #
    # dataGeneration.finishProgrammConectionFromOutside()
    #
    #
    # print("right hand:",len(tracked_collision_hand_right))
    # print("left hand:",len(tracked_collision_hand_left))
    # print("right forearm:",len(tracked_collision_forearm_right))
    # print("left forearm",len(tracked_collision_forearm_left))
    # print("right arm",len(tracked_collision_arm_right))
    # print("left arm",len(tracked_collision_arm_left))
    #
    # print("joint's coordinates",len(joint_coordinates))
    # print("hand coordinates", len(hand_coordinates))
    #
    # print("visual right perception:",len(visual_perception_right))
    # print("visual left perception:", len(visual_perception_left))
    #
    # print("scene records:", len(scene_records))
    #
    # print("head eye coordinates:",len(head_eyes_coordinates))
    # print("object coordinates:", len(object_coordinates))
    # print('JOINT COORDINATES', joint_coordinates)
    #
    #
    # print("x axis", X_trajectory)
    # print("y axis", Y_trajectory)
    # print("z axis", Z_trajectory)
    #
    # print('Number of trials:', len(PAUSE), 'Number of Ground Truth:',len(GroundTruthForLeftHand))
    # print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(GroundTruthForRightHand))
    # print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(GroundTruthForLeftForearm))
    # print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(GroundTruthForRightForearm))
    # print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(GroundTruthForLeftArm))
    # print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(GroundTruthForRightArm))
    #
    # # ---------------------------------- SAVING DATA --------------------------------------
    #
    # __DATAOBJECT: ObjectCoordinatesWriter = ObjectCoordinatesWriter()
    # __DATAOBJECT.outputDataToCSV(object_coordinates, filename=path + '/' + 'object_data.csv')
    #
    # __DATAHAND: HandCoordinatesWriter = HandCoordinatesWriter()
    # __DATAHAND.HandCoordinatesToCSV(hand_coordinates, filename=path + '/' + 'hand_coordinates.csv')
    # __DATAHAND.JointsCoodinatesToCSV(joint_coordinates, filename=path + '/' + 'joints_coordinates.csv')
    #
    # __DATAHEAD: HeadCoordinatesWriter = HeadCoordinatesWriter()
    # __DATAHEAD.outputToCSV(head_eyes_coordinates, filename=path + '/' + 'head_coordinates.csv')
    #
    # __DATASENSORS: SensorCoordinatesWriter = SensorCoordinatesWriter()
    # __DATASENSORS.outputDataToCSV(tracked_collision_arm_right, filename=path + '/' + 'arm_right.csv')
    # __DATASENSORS.outputDataToCSV(tracked_collision_arm_left, filename= path + '/' + 'arm_left.csv')
    # __DATASENSORS.outputDataToCSV(tracked_collision_forearm_right, filename= path + '/' + 'forearm_right.csv')
    # __DATASENSORS.outputDataToCSV(tracked_collision_forearm_left, filename= path + '/' + 'forearm_left.csv')
    # __DATASENSORS.outputDataToCSV(tracked_collision_hand_right, filename= path + '/' + 'hand_right.csv')
    # __DATASENSORS.outputDataToCSV(tracked_collision_hand_left, filename= path + '/' + 'hand_left.csv')
    #
    #
    # __DATACAMERAS: camerasCoordinatesWriter = camerasCoordinatesWriter()
    # __DATACAMERAS.saveImgDataIntoDataSet(visual_binocular_perception, datasetName='binocularPerception', filename=path + '/' + 'binocular_perception.h5')
    #
    #
    # __DATACAMERAS.saveImgDataIntoDataSet(scene_records,datasetName= 'sceneRecords', filename= path + '/' + 'scene_records.h5')
    #
    #
    # __GROUNDTRUTH: GroundTruthWriter = GroundTruthWriter()
    #
    # __GROUNDTRUTH.saveGroundTruthtoCVS(GroundTruthForRightHand, filename=path + '/' + 'gt_right_hand.csv')
    # __GROUNDTRUTH.saveGroundTruthtoCVS(GroundTruthForLeftHand, filename= path + '/' + 'gt_left_hand.csv')
    #
    # __GROUNDTRUTH.saveGroundTruthtoCVS(GroundTruthForRightForearm, filename= path + '/' + 'gt_right_forearm.csv')
    # __GROUNDTRUTH.saveGroundTruthtoCVS(GroundTruthForLeftForearm, filename= path + '/' +'gt_left_forearm.csv')
    #
    # __GROUNDTRUTH.saveGroundTruthtoCVS(GroundTruthForRightArm, filename= path + '/' + 'gt_right_arm.csv')
    # __GROUNDTRUTH.saveGroundTruthtoCVS(GroundTruthForLeftArm, filename= path + '/' + 'gt_left_arm.csv')
