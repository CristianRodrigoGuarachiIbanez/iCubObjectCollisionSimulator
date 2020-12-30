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
from typing import List, Dict, Tuple, TypeVar


from visual_perception_tracker import VisualPerception
from tactil_perception_tracker import TactilCollisionDetector
from export_data.export_collision_data import Writer, Writer2
from export_data.export_static_coordinates_data import NewCoordinatesWriter
from scene_recorder import SceneRecorder
from trajectory_generator import CoordinatesGenerator, StartSelector, IntervalGenerator, TrajectoryGenerator
from hand_new_coordinates import GettingTheHandToMove, __ReadOnGoingJointCoordinates
from head_new_position import MoveHead

################ Import parameter from parameter file ################
from examples.example_parameter import CLIENT_PREFIX, GAZEBO_SIM, ROBOT_PREFIX

############# Import control classes for both simulators #############
if ROBOT_PREFIX or CLIENT_PREFIX:

    import Python_libraries.iCubSim_world_controller as iCubSim_ctrl
    import Python_libraries.iCubSim_model_groups_definition as mdl_define
    import Python_libraries.iCubSim_model_groups_control as mdl_ctrl


class ObjectTrajectory:
    A= TypeVar("A", List, str)
    def __init__(self, typObj: A ="ssphere", ArmSide: bool = False, stepsize: int= 50):
        """ create simple objects objects= sbox, scyl oder ssphere"""

        self.__iCubSim_wrld_ctrl: iCubSim_ctrl.WorldController = iCubSim_ctrl.WorldController()

        print(" ------------------ init collision detector -----------------")
        self.__collisionDect_hand: TactilCollisionDetector = TactilCollisionDetector(armSec=0)
        self.__collisionDect_foreamr: TactilCollisionDetector = TactilCollisionDetector(armSec=1)
        self.__collisionDect_arm: TactilCollisionDetector = TactilCollisionDetector(armSec=2)

        print("-------------- Init visual perception----------- ")
        self.__visualPerc: VisualPerception = VisualPerception()

        print("---------------init scene recorder---------------")
        self.__sceneRecorder: SceneRecorder = SceneRecorder()

        print("--------------- Init Object Trajectory around the Hand Side----------")
        if(ArmSide):
            start_X: StartSelector = StartSelector(ArmSide, True) # handSide == True -> right hand
            end_X: float = 0.0
            armSide: str = "right_arm"
            NewHandCoord: List[float] = [-0.111577, 0.27158, 0.501089, 0.1]
            NewHeadCoord: List[float] = [-35., -10., -30., 0., 0., 10.]

        else:
            start_X: StartSelector = StartSelector(ArmSide, False) # handSide == False -> left Hand
            end_X: float = 0.3
            armSide: str = "left_arm"
            NewHandCoord: List[float] = [0.111577, 0.27158, 0.501089, 0.1]
            NewHeadCoord: List[float] = [-35., 10., 30., 0., 0., 10.]

        start_Y: StartSelector = StartSelector(False, True)
        start_Z: StartSelector = StartSelector(True, True)
        # for 1000 0.0006006006006006006
        self.__X: TrajectoryGenerator = TrajectoryGenerator(float(start_X), end_X, stepsize= 0.05, dtyp= np.float_,num=stepsize, linspace=True)
        self.__Y: TrajectoryGenerator = TrajectoryGenerator(float(start_Y), 0.9, stepsize= 0.05, dtyp= np.float_, num=stepsize, linspace=True)
        self.__Z:  TrajectoryGenerator = TrajectoryGenerator(float(start_Z), 0.4, stepsize= 0.05, dtyp= np.float_, num=stepsize, linspace=True)

        print("------------------ init Arm/Hand Position -----------------")
        self.__newHandCoordinates: GettingTheHandToMove = GettingTheHandToMove(ArmSide=armSide, CurrCoordinates=NewHandCoord)
        print(self.__newHandCoordinates.readOnGoingHandCoordinates())
        print("------------------- init Head Position ---------------------")
        self.__newHeadCoordinates: MoveHead = MoveHead(NewHeadCoord)
        print(self.__newHeadCoordinates.OutputNewPositionOrientation())
        # -------------------------- Object Movement---------------------------------------------
        if isinstance(typObj, str):
            object: iCubSim_ctrl.WorldController = self._objects_iCubSim_ID(typObj)
            time.sleep(1.)
            self.location, self.collision = self._object_iCubSim(object)
        print("insert one of the ")

    def __repr__(self):
        return tabulate(self.location, headers=['Right/Left (X)', 'Up/Down (Y)', 'Forward/Backward (Z)'], showindex="always")

    def _objects_iCubSim_multi_IDs(self, typObj:List) -> List:
        """create multiple objects IDs"""
        obj_IDs: List = []
        for obj in typObj:
            obj_id: iCubSim_ctrl.WorldController = self._objects_iCubSim_ID(obj)
            obj_IDs.append(obj_id)
        return obj_IDs

    def _object_iCubSim(self, object: iCubSim_ctrl.WorldController) -> Tuple[List, Tuple]:
        """execute the introduced type of object as argument and returns a list of coordinates where the object appeared
         running the privat staticmethod _object_movement"""
        objectCoordTrajectory: Tuple = self.__X, self.__Y, self.__Z
        contactSensors: Tuple = self.__collisionDect_hand, self.__collisionDect_foreamr, self.__collisionDect_arm
        return self._object_movement(object, self.__iCubSim_wrld_ctrl, objectCoordTrajectory, contactSensors, self.__visualPerc, self.__sceneRecorder)

    def _objects_iCubSim_ID(self, typObj:str) -> iCubSim_ctrl.WorldController:
        """ create a object ID according to "sbox": [1, 0, 0], "scyl": [0, 0.5, 0.3], "ssphere" """
        if(typObj):
            sphere_id = self.__iCubSim_wrld_ctrl.create_object(typObj, [0.05], [0, 1., 0.3], [1, 1, 1])
            return sphere_id

    @staticmethod
    def _object_movement(objID: iCubSim_ctrl.WorldController, iCubSim_wrld_ctrl: iCubSim_ctrl.WorldController, coordTrajectory: Tuple, skinTracker: Tuple,
                         visual_tracker: VisualPerception, scene_recorder: SceneRecorder) -> Tuple[List, Tuple]:
        """create random area corrected trajectory"""

        tracked_collision_hand: List[List,...] = []
        tracked_collision_forearm : List[List,...] = []
        tracked_collision_arm: List[List,...] = []

        tracked_coord: List[List,...] = []

        JointCoordinates: List[np.ndarray] = []
        HandCoordinates: List[Dict] = []

        # create the directory wherein the imgs will be saved
        path: str = os.path.dirname(os.path.abspath(__file__)) + "/img"
        if not os.path.exists(path):
            os.mkdir(path)
        # ---------------------- Anzahl der Events: Objektbewegung, Augenbilder, Scenebilder ------------------------

        X: np.ndarray = coordTrajectory[0].ConcatenateTrajectoryArrays()
        Y: np.ndarray = coordTrajectory[1].ConcatenateTrajectoryArrays()
        Z: np.ndarray = coordTrajectory[2].ConcatenateTrajectoryArrays()

        start: int = 9
        STEPSIZE: int = 10
        PAUSE: List[int] = [x for x in range(start, len(X), STEPSIZE)]

        # ---------------------initial phase: appending location data -------------
        tracked_coord.append(iCubSim_wrld_ctrl.get_object_location(objID).tolist())

        # ---------------------- Head, Hand and Eyes Coordinates --------------------


        # -------------- initial phase: appending the visual perception --------------
        visual_tracker.read_camera_images()

        # --------------------initial phase: appending the Scene ------------------
        scene_recorder.read_scene()

        # ---------------------initial phase: appending the collision data----------

        tracked_collision_hand.append(skinTracker[0].skin_sensor_reader())
        #time.sleep(0.5)
        tracked_collision_forearm.append(skinTracker[1].skin_sensor_reader())
        #time.sleep(0.5)
        tracked_collision_arm.append(skinTracker[2].skin_sensor_reader())

        i: int = 0

        while True:

            try:

                # -------------- move the object -----------------------------
                iCubSim_wrld_ctrl.move_object(objID, [X[i], Y[i], Z[i]])
                #print(iCubSim_wrld_ctrl.get_object_location(objID).tolist())

                # -------------- Head, Hand and Eyes Coordinates ------------
                #JointCoordinates.append(__ReadOnGoingJointCoordinates())

                # --------------append visual perception data ---------------
                visual_tracker.read_camera_images(i)
                time.sleep(0.5)

                # --------------append scene data ----------------------------
                scene_recorder.read_scene(i)
                time.sleep(0.5)

                # ------------- append collision data of the arm skin -----------------------
                tracked_collision_hand.append(skinTracker[0].skin_sensor_reader())
                time.sleep(0.5)
                tracked_collision_forearm.append(skinTracker[1].skin_sensor_reader())
                time.sleep(0.5)
                tracked_collision_arm.append(skinTracker[2].skin_sensor_reader())
                time.sleep(0.5)

                #------------- append location data of object ------------------------
                tracked_coord.append(iCubSim_wrld_ctrl.get_object_location(objID).tolist())
                time.sleep(0.5)

                if(i in PAUSE):
                    time.sleep(3)
                else:
                    pass

            except Exception as e:
                print(e)
                break
            except KeyboardInterrupt:
                break

            i += 1

        visual_tracker.closing_program()
        scene_recorder.closing_program()

        skinTracker[0].closing_programm()
        skinTracker[1].closing_programm()
        skinTracker[2].closing_programm()

        iCubSim_wrld_ctrl.del_all()
        del iCubSim_wrld_ctrl

        return tracked_coord, (tracked_collision_hand, tracked_collision_forearm, tracked_collision_arm)

if __name__ == '__main__':

    obj_1: ObjectTrajectory = ObjectTrajectory(ArmSide=True)
    dfLoc: Writer2 = Writer2(obj_1.location)
    print("------------Exporting Location Data---------------")
    dfLoc.export_dataframe("object_location.csv", "csv")
    dfColli_hand: Writer = Writer(obj_1.collision[0])
    dfColli_forearm: Writer = Writer(obj_1.collision[1])
    dfColli_arm: Writer = Writer(obj_1.collision[2])
    print("------------Exporting Collition Data---------------")
    dfColli_hand.export_dataframe("collision_right_hand.csv", "collision_left_hand.csv", "csv")
    dfColli_forearm.export_dataframe("collision_right_forearm.csv", "collision_left_forearm.csv", "csv")
    dfColli_arm.export_dataframe("collision_right_arm.csv", "collision_left_arm.csv", "csv")


