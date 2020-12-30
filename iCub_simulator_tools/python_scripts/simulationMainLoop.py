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
sys.path.insert(0, "~/PycharmProjects/projectPy3.8/iclub_simulator_tools/python_scripts/")

from hand_trajectory_tracker import *
from tabulate import tabulate
from typing import List, Dict, Tuple, TypeVar, Callable
from progress.bar import Bar


from visual_perception_tracker import VisualPerception
from tactil_perception_tracker import TactilCollisionDetector
from export_data.export_collision_data import Writer, Writer2
from export_data.export_static_coordinates_data import HandCoordinatesWriter, HeadCoordinatesWriter, SensorCoordinatesWriter, ObjectCoordinatesWriter, camerasCoordinatesWriter
from scene_recorder import SceneRecorder
from trajectory_generator import CoordinatesGenerator, StartSelector, TrajectoryGenerator, __regulatingValuesEqualToLimit
from hand_new_coordinates import HandMovementExecution, __ReadOnGoingJointCoordinates
from head_new_position import MoveHead

################ Import parameter from parameter file ################
from examples.example_parameter import CLIENT_PREFIX, GAZEBO_SIM, ROBOT_PREFIX

############# Import control classes for both simulators #############
if ROBOT_PREFIX or CLIENT_PREFIX:

    import Python_libraries.iCubSim_world_controller as iCubSim_ctrl


class ObjectMovementExecution:
    A= TypeVar("A", List, str)
    def __init__(self, typObj: A ="ssphere"):
        """ create simple objects objects= sbox, scyl oder ssphere"""

        self.__iCubSim_wrld_ctrl: iCubSim_ctrl.WorldController = iCubSim_ctrl.WorldController()
        if isinstance(typObj, str):
            # create the object
            self.__object: iCubSim_ctrl.WorldController = self.__objects_iCubSim_ID(typObj)
            time.sleep(1.)

        self.__finalObjectCoordinates: List[List] = [["X Coordinates", "Y Coordinates", "Z Coordinates"],]


    def __repr__(self):
        if (self.__finalObjectCoordinates):
            return tabulate(self.__finalObjectCoordinates, headers=['Right/Left (X)', 'Up/Down (Y)', 'Forward/Backward (Z)'], showindex="always")
        elif not (self.__finalObjectCoordinates):
            return "None Data was saved inside the class"


    def object_movement(self, Xaxis: float, Yaxis: float, Zaxis: float) -> None:
        """moves the Object in 3 Simulation World
        @parameters the  X, Y, Z Axis where the object will be moved to"""

        # -------------- move the object -----------------------------
        self.__iCubSim_wrld_ctrl.move_object(self.__object, [Xaxis, Yaxis, Zaxis])
        print(self.__iCubSim_wrld_ctrl.get_object_location(self.__object).tolist())

    def getOnGoingObjectCoordinatesFromOutside(self) -> List[List]:
        """ method that could be called from outside the class
        @returns the location coordinates of the object only"""
        return self.__iCubSim_wrld_ctrl.get_object_location(self.__object).tolist()

    def deleteAll(self) -> None:
        self.__iCubSim_wrld_ctrl.del_all()
        del self.__iCubSim_wrld_ctrl

    def getSaveObjectCoordinatesInsideClassVariable(self) -> List[List]:
        """run the get save object coordinates function
        @return a list of lists with the location coordinates of the object while saving these inside the class
        """
        return self.__getSaveObjectCoordinatesInsideTheClass(self.__object, self.__iCubSim_wrld_ctrl, self.__saveObjectCoordinatesInsideTheClass())

    def __objects_iCubSim_ID(self, typObj:str) -> iCubSim_ctrl.WorldController:
        """ create a object ID inisde the class according to "sbox": [1, 0, 0], "scyl": [0, 0.5, 0.3], "ssphere"
        @returns a Object ID"""
        if(typObj):
             return  self.__iCubSim_wrld_ctrl.create_object(typObj, [0.05], [0, 1., 0.3], [1, 1, 1])

    @staticmethod
    def __getSaveObjectCoordinatesInsideTheClass(ObjID: iCubSim_ctrl.WorldController, __iCubSim_wrld_ctrl: iCubSim_ctrl.WorldController, __saveOb: Any ) -> List[List]:
        """saves the location coordinates of the object inside the class as final coordinates
        @parameter object id, world controller, function which saves the location coordinates of the object
        @returns a list of list of the location coordinates of the object"""
        __saveOb()
        return __iCubSim_wrld_ctrl.get_object_location(ObjID).tolist()

    def __saveObjectCoordinatesInsideTheClass(self) -> None:
        """saves the location coordinates of the object only"""
        self.__finalObjectCoordinates.append(self.__iCubSim_wrld_ctrl.get_object_location(self.__object).tolist())



class PutTogetherAllExecProgramms:
    def __init__(self, ArmSide: str, NewHandCoord: List[float], NewHeadCoord: List[float]):

        # -------------------------- Object Movement---------------------------------------------
        print(" -------------- init object movement executer and tracker-------------------")
        self.__object: ObjectMovementExecution = ObjectMovementExecution() # default sphere

        # ------------------------------ Hand's position and Orientation ------------------------------
        print(" --------------- init hand movement executer and tracker ------------------")
        self.__handMovement: HandMovementExecution = HandMovementExecution(ArmSide=ArmSide, CurrCoordinates=NewHandCoord)

        # ------------------------------ Head's and eyes position ------------------------------------
        print("------------------- init head movement executer and tracker ---------------------")
        self.__newHeadCoordinates: MoveHead = MoveHead(NewHeadCoord)

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
        """ moves the generated object to the new given coordinates"""
        self.__object.object_movement(X,Y,Z)
    def translationHandCoordinatesFromOutside(self, newCoord: List[float]) -> None:
        """ moves the Hand to new given coordinates"""
        self.__handMovement.moveHandToNewPositionCooridnatesfromOuside(newCoord)

    def translationHeadCoordinatesFromOutside(self, newCoord: List[float]) -> None:
        """ moves the Head to new given coordinates"""
        self.__newHeadCoordinates.moveHeadToNewPositionFromOutside(newCoord)

    def startObjectCoordinatesFromOutside(self) -> List[List]:
        """@returns a list of list with the location coordinates of the object"""
        return self.__object.getOnGoingObjectCoordinatesFromOutside()
    def startHandJointsCoordinatesFromOutside(self) -> Dict[str,str]:
        """@returns a dict with the hand location and orientation coordinates """
        return self.__handMovement.readOnGoingHandCoordinatesFromOutside()
    def startVisualPerceptionCoordFromOutside(self, index: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """@returns a list of list with the visual data of the perceived object"""
        return self.__visualPerc.readCameraImagesFromOutside(index)
    def startSceneRecorderCoordsFromOutside(self, index: int = None) -> np.ndarray:
        """@returns a list of lists with the visual data (coordinates) of the perceived """
        return self.__sceneRecorder.readSceneCamera(index)
    def startCollisionSensorDataFromOutside(self) -> Tuple[Tuple,Tuple,Tuple]:
        """@returns a tuple of tuples with the corresponds lists for right and left collision coordinates"""
        return self.__collisionDect_hand.skin_sensor_reader(), self.__collisionDect_forearm.skin_sensor_reader(), self.__collisionDect_arm.skin_sensor_reader()

    def startHeadEyeCoordinatesFromOutside(self) -> List[float]:
        """@returns a List of floats with the coordinates of the head position"""
        headEyeCoord: List = self.__newHeadCoordinates.getHeadJointsCoordinatesFromOutside().tolist()
        return headEyeCoord


    def __endVisualPerceptionCoordFromImside(self) -> None:
        """ Closing the coneccion with the eyes camaras"""
        self.__visualPerc.closing_program()
    def __endSceneRecorderCoordsFromInside(self) -> None:
        """ Closing the conexion with the scene cameras"""
        self.__sceneRecorder.closing_program()
    def __endCollisionSensorDataFromInside(self) -> None:
        """ Closing the conexcion with the collision sensors of the skin"""
        self.__collisionDect_hand.closing_programm(), self.__collisionDect_forearm.closing_programm(), self.__collisionDect_arm.closing_programm()
    def __endHandCoordinatesFromInside(self) -> None:
        """  Closing the YARP conexcion with hand coordinates"""
        self.__handMovement.closingProgramm()
    def __endHeadCoordinatesFromInside(self) -> None:
        """  Closing the YARP conexcion with head coordinates"""
        self.__newHeadCoordinates.closing_programm()

    def EndProgrammConectionFromOutside(self) -> None:

        self.__endHeadCoordinatesFromInside()
        self.__endCollisionSensorDataFromInside()
        self.__endHandCoordinatesFromInside()
        self.__endSceneRecorderCoordsFromInside()
        self.__endVisualPerceptionCoordFromImside()

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

    # ----------------------- create the directory wherein the imgs will be saved ---------------------------
    path: str = os.path.dirname(os.path.abspath(__file__)) + "/trials"
    if not os.path.exists(path):
        os.mkdir(path)
        pathBinocular: str = os.path.dirname(os.path.abspath(__file__)) + "/trials/image_outputs"
        pathScene: os.path = os.path.dirname(os.path.abspath(__file__)) + "/trials/scene_outputs"

        if not os.path.exists(pathBinocular):
            os.mkdir(pathBinocular)


        elif not os.path.exists(pathScene):
            os.mkdir(pathScene)


    #parameters for the trajectory
    ArmSide: bool = True
    steps: int = 3 ######

    # ---------------------- Anzahl der Events: Objektbewegung, Augenbilder, Scenebilder ------------------------
    start_X, end_X, armSide, NewHandCoord, NewHeadCoord = __initializeArmSide(ArmSide)  # type: float, float, str, List[float], List[float]
    start_Y: float = __regulatingValuesEqualToLimit(0.6, False, True )#StartSelector(False, True)
    start_Z: float = __regulatingValuesEqualToLimit(0.3, True, True)#StartSelector(True, True)

    X: TrajectoryGenerator = TrajectoryGenerator(start_X, end_X, stepsize=0.05, dtyp=np.float_, num=steps, linspace=True)
    Y: TrajectoryGenerator = TrajectoryGenerator(start_Y, 0.9, stepsize=0.05, dtyp=np.float_,  num=steps, linspace=True)
    Z: TrajectoryGenerator = TrajectoryGenerator(start_Z, 0.4, stepsize=0.05, dtyp=np.float_, num=steps, linspace=True)
    X: np.ndarray = X.ConcatenateTrajectoryArrays()
    Y: np.ndarray = Y.ConcatenateTrajectoryArrays()
    Z: np.ndarray = Z.ConcatenateTrajectoryArrays()

    # ------------------------------ instantiate the main class ----------------------------------
    dataGeneration: PutTogetherAllExecProgramms = PutTogetherAllExecProgramms(ArmSide = armSide, NewHandCoord = NewHandCoord, NewHeadCoord = NewHeadCoord)

    # ----------------------------- initial coordinates values --------------------------------
    tracked_collision_hand_right: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[0][0]]
    tracked_collision_hand_left: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[0][1]]

    tracked_collision_forearm_right: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[1][0]]
    tracked_collision_forearm_left: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[1][1]]

    tracked_collision_arm_right: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[2][0]]
    tracked_collision_arm_left: List[List[float]] = [dataGeneration.startCollisionSensorDataFromOutside()[2][1]]

    object_coordinates: List[List] = [dataGeneration.startObjectCoordinatesFromOutside()]

    joint_coordinates: List[np.ndarray] = [__ReadOnGoingJointCoordinates(armSide)]
    hand_coordinates: List[Dict] = [dataGeneration.startHandJointsCoordinatesFromOutside()]

    visual_perception_right: List[np.ndarray] = [dataGeneration.startVisualPerceptionCoordFromOutside()[0]]
    visual_perception_left: List[np.ndarray] = [dataGeneration.startVisualPerceptionCoordFromOutside()[1]]
    visual_binocular_perception: List[Tuple[np.ndarray, np.ndarray]] = [dataGeneration.startVisualPerceptionCoordFromOutside()]
    scene_records: List[np.ndarray] = [dataGeneration.startSceneRecorderCoordsFromOutside()]

    head_eyes_coordinates: List[List] = [dataGeneration.startHeadEyeCoordinatesFromOutside()]

    # ---------------------- Parameters for the while loop ---------------------------------
    start: int = 9
    STEPSIZE: int = 10
    PAUSE: List[int] = [x for x in range(start, len(X), STEPSIZE)]
    bar: Bar = Bar('Processing', max=steps, fill='=', suffix='%(percent)d%%')
    i: int = 0
    while True:


        try:
            print("   ", f"index:{i},left hand:{len(tracked_collision_hand_left)}, left forearm:{len(tracked_collision_forearm_left)}, left arm:{len(tracked_collision_arm_left)},"
                         f" right hand:{len(tracked_collision_hand_right)}, right forearm: {len(tracked_collision_forearm_right)}, right arm: {len(tracked_collision_arm_right)} ",
                        f" object coord: {len(object_coordinates)}, hand coord:{len(hand_coordinates)}, joints coord:{len(joint_coordinates)}, head coord:{len(head_eyes_coordinates)}")
            # -------------- move the object -----------------------------
            dataGeneration.translationObjectCoordinatesFromOutside(X[i], Y[i], Z[i])
            #iCubSim_wrld_ctrl.move_object(self.__object, [X[i], Y[i], Z[i]])
            # print(iCubSim_wrld_ctrl.get_object_location(objID).tolist())

            # -------------- Head, Hand and Eyes Coordinates ------------
            hand_coordinates.append(dataGeneration.startHandJointsCoordinatesFromOutside())
            joint_coordinates.append(__ReadOnGoingJointCoordinates(armSide))
            head_eyes_coordinates.append(dataGeneration.startHeadEyeCoordinatesFromOutside())
            # --------------append visual perception data ---------------
            visual_perception_right.append(dataGeneration.startVisualPerceptionCoordFromOutside(i)[0])
            time.sleep(0.5)
            visual_perception_left.append(dataGeneration.startVisualPerceptionCoordFromOutside(i)[1])
            time.sleep(0.5)

            # --------------append scene data ----------------------------
            scene_records.append(dataGeneration.startSceneRecorderCoordsFromOutside(i))

            # ------------- append collision data of the arm skin -----------------------
            tracked_collision_hand_right.append(dataGeneration.startCollisionSensorDataFromOutside()[0][0])
            time.sleep(0.5)
            tracked_collision_hand_left.append(dataGeneration.startCollisionSensorDataFromOutside()[0][1])
            time.sleep(0.5)

            tracked_collision_forearm_right.append(dataGeneration.startCollisionSensorDataFromOutside()[1][0])
            time.sleep(0.5)
            tracked_collision_forearm_left.append(dataGeneration.startCollisionSensorDataFromOutside()[1][1])
            time.sleep(0.5)

            tracked_collision_arm_right.append(dataGeneration.startCollisionSensorDataFromOutside()[2][0])
            time.sleep(0.5)
            tracked_collision_arm_left.append(dataGeneration.startCollisionSensorDataFromOutside()[2][1])
            time.sleep(0.5)

            # ------------- append location data of object ------------------------
            object_coordinates.append(dataGeneration.startObjectCoordinatesFromOutside())
            time.sleep(0.5)

            if (i in PAUSE):
                time.sleep(3)
            else:
                pass

        except Exception as e:
            print(e)
            break
        except KeyboardInterrupt:
            break
        bar.next()

        i += 1
    bar.finish()
    dataGeneration.EndProgrammConectionFromOutside()


    print("right hand:",len(tracked_collision_hand_right))
    print("left hand:",len(tracked_collision_hand_left))
    print("right forearm:",len(tracked_collision_forearm_right))
    print("left forearm",len(tracked_collision_forearm_left))
    print("right arm",len(tracked_collision_arm_right))
    print("left arm",len(tracked_collision_arm_left))

    print("joint's coordinates",len(joint_coordinates))
    print("hand coordinates", len(hand_coordinates))

    print("visual right perception:",len(visual_perception_right))
    print("visual left perception:", len(visual_perception_left))

    print("scene records:", len(scene_records))

    print("head eye coordinates:",len(head_eyes_coordinates))
    print("object coordinates:", len(object_coordinates))
    # print(hand_coordinates)
    # print(head_eyes_coordinates)
    # print(joint_coordinates)
    # print(visual_perception_left)
    # print(visual_perception_left[0].shape)
    # print(visual_perception_left[0].size)
    # print(visual_perception_left[0].ndim)

    print("x axis", X)
    print("y axis", Y)
    print("z axis", Z)
    # ---------------------------------- SAVING DATA --------------------------------------

    __DATAOBJECT: ObjectCoordinatesWriter = ObjectCoordinatesWriter()
    __DATAOBJECT.outputDataToCSV(object_coordinates, filename=path + '/' + 'object_data.csv')

    __DATAHAND: HandCoordinatesWriter = HandCoordinatesWriter()
    __DATAHAND.HandCoordinatesToCSV(hand_coordinates, filename=path + '/' + 'hand_coordinates.csv')
    __DATAHAND.JointsCoodinatesToCSV(joint_coordinates, filename=path + '/' + 'joints_coordinates.csv')

    __DATAHEAD: HeadCoordinatesWriter = HeadCoordinatesWriter()
    __DATAHEAD.outputToCSV(head_eyes_coordinates, filename=path + '/' + 'head_coordinates.csv')

    __DATASENSORS: SensorCoordinatesWriter = SensorCoordinatesWriter()
    __DATASENSORS.outputDataToCSV(tracked_collision_arm_right, filename=path + '/' + 'arm_right.csv')
    __DATASENSORS.outputDataToCSV(tracked_collision_arm_left, filename= path + '/' + 'arm_left.csv')
    __DATASENSORS.outputDataToCSV(tracked_collision_forearm_right, filename= path + '/' + 'forearm_right.csv')
    __DATASENSORS.outputDataToCSV(tracked_collision_forearm_left, filename= path + '/' + 'forearm_left.csv')
    __DATASENSORS.outputDataToCSV(tracked_collision_hand_right, filename= path + '/' + 'hand_right.csv')
    __DATASENSORS.outputDataToCSV(tracked_collision_hand_left, filename= path + '/' + 'hand_left.csv')


    __DATACAMERAS: camerasCoordinatesWriter = camerasCoordinatesWriter()
    __DATACAMERAS.saveImgDataIntoDataSet(visual_binocular_perception, datasetName='binocularPerception', filename=path + '/' + 'binocular_perception.h5')


    __DATACAMERAS.saveImgDataIntoDataSet(scene_records,datasetName= 'sceneRecords', filename= path + '/' + 'scene_records.h5')
