"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
Object trajectory tracker
"""


######################################################################
########################## Import modules  ###########################
######################################################################

import os
import time
import numpy as np
from random import choice
import yarp

import sys
sys.path.insert(0, "~/PycharmProjects/projectPy3.8/iclub_simulator_tools/python_scripts/")

from hand_trajectory_tracker import *
from tabulate import tabulate
from typing import List, Dict, Tuple, TypeVar
import random

from visual_perception_tracker import VisualPerception
from tactil_perception_tracker import TactilCollisionDetector
from export_collision_data import Writer, Writer2
from scene_recorder import SceneRecorder
from trajectory_generator import CoordinatesGenerator, StartSelector, IntervalGenerator

################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, GAZEBO_SIM, ROBOT_PREFIX

############# Import control classes for both simulators #############
if ROBOT_PREFIX or CLIENT_PREFIX:

    import Python_libraries.iCubSim_world_controller as iCubSim_ctrl
    import Python_libraries.iCubSim_model_groups_definition as mdl_define
    import Python_libraries.iCubSim_model_groups_control as mdl_ctrl


class ObjectTrajectory:
    A= TypeVar("A", List, str)
    def __init__(self, typObj: A ="ssphere"):
        """ create simple objects objects= sbox, scyl oder ssphere"""

        self.__iCubSim_wrld_ctrl: iCubSim_ctrl.WorldController = iCubSim_ctrl.WorldController()

        print("---------- init collision detector -----------")
        self.__collisionDect_hand: TactilCollisionDetector = TactilCollisionDetector(armSec=0)
        self.__collisionDect_foreamr: TactilCollisionDetector = TactilCollisionDetector(armSec=1)
        self.__collisionDect_arm: TactilCollisionDetector = TactilCollisionDetector(armSec=2)

        print("--------------Init visual perception----------- ")
        self.__visualPerc: VisualPerception = VisualPerception()

        print("---------------init scene recorder---------------")
        self.__sceneRecorder: SceneRecorder = SceneRecorder()

        print("--------------- Init Object Trajectory ----------")
        startRight_X: StartSelector = StartSelector(True, False)
        start_Y: StartSelector = StartSelector(False, True)
        start_Z: StartSelector = StartSelector(True, True)

        self.__X: CoordinatesGenerator = CoordinatesGenerator(startRight_X, 0.0, stepsize= 0.0006006006006006006, num=50, linspace=True)
        self.__Y: CoordinatesGenerator = CoordinatesGenerator(start_Y, 0.7, stepsize= 0.0006006006006006006, num=50, linspace=True)
        self.__Z:  CoordinatesGenerator = CoordinatesGenerator(start_Z, 0.4, stepsize= 0.0006006006006006006, num=50, linspace=True)


        # --------------------------------------------------------------------------------
        if isinstance(typObj, str):
            object: iCubSim_ctrl.WorldController = self._objects_iCubSim_ID(typObj)
            time.sleep(1.)
            self.location, self.collision = self._object_iCubSim(object)

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
        coordTrajectory: Tuple = self.__X, self.__Y, self.__Z
        contactSensors: Tuple = self.__collisionDect_hand, self.__collisionDect_foreamr, self.__collisionDect_arm
        return self._object_movement(object, self.__iCubSim_wrld_ctrl, coordTrajectory, contactSensors, self.__visualPerc, self.__sceneRecorder)

    def _objects_iCubSim_ID(self, typObj:str) -> iCubSim_ctrl.WorldController:
        """ create a object ID according to "sbox": [1, 0, 0], "scyl": [0, 0.5, 0.3], "ssphere" """
        if(typObj):
            sphere_id = self.__iCubSim_wrld_ctrl.create_object(typObj, [0.05], [0, 1., 0.3], [1, 1, 1])
            return sphere_id

    @staticmethod
    def _object_movement(objID: iCubSim_ctrl.WorldController, iCubSim_wrld_ctrl: iCubSim_ctrl.WorldController, coordTrajectory: Tuple, cont_tracker: Tuple,
                         visual_tracker: VisualPerception, scene_recorder: SceneRecorder) -> Tuple[List, Tuple]:
        """create random area corrected trajectory"""

        tracked_collision_hand: List[List,...] = []
        tracked_collision_forearm : List[List,...] = []
        tracked_collision_arm: List[List,...] = []
        tracked_coord: List[List, ...] = []

        # create the directory wherein the imgs will be saved
        path: str = os.path.dirname(os.path.abspath(__file__)) + "/img"
        if not os.path.exists(path):
            os.mkdir(path)
        #---------------------- Anzahl der Events: Objektbewegung, Augenbilder, Scenebilder ------------------------

        X: np.ndarray = coordTrajectory[0]
        Y: np.ndarray = coordTrajectory[1]
        Z: np.ndarray = coordTrajectory[2]
        start: int = 9
        STEPSIZE: int = 10
        PAUSE: List[int] = [x for x in range(start, len(X), STEPSIZE)]

        # ---------------------initial phase: appending location data -------------
        tracked_coord.append(iCubSim_wrld_ctrl.get_object_location(objID).tolist())

        # ---------------------initial phase: appending the visual perception------
        visual_tracker.read_camera_images()

        # --------------------initial phase: appending the Scene ------------------
        scene_recorder.read_scene()

        # ---------------------initial phase: appending the collision data----------

        tracked_collision_hand.append(cont_tracker[0].skin_sensor_reader())
        #time.sleep(0.5)
        tracked_collision_forearm.append(cont_tracker[1].skin_sensor_reader())
        #time.sleep(0.5)
        tracked_collision_arm.append(cont_tracker[2].skin_sensor_reader())

        i: int = 0

        while True:

            try:

                # -------------- move the object -----------------------------
                iCubSim_wrld_ctrl.move_object(objID, [X[i], Y[i], Z[i]])
                #print(iCubSim_wrld_ctrl.get_object_location(objID).tolist())
                # --------------append visual perception data ---------------
                visual_tracker.read_camera_images(i)
                time.sleep(0.5)

                # --------------append scene data ----------------------------
                scene_recorder.read_scene(i)
                time.sleep(0.5)

                # ------------- append collision data of the hand -----------------------
                tracked_collision_hand.append(cont_tracker[0].skin_sensor_reader())
                time.sleep(0.5)
                tracked_collision_forearm.append(cont_tracker[1].skin_sensor_reader())
                time.sleep(0.5)
                tracked_collision_arm.append(cont_tracker[2].skin_sensor_reader())
                time.sleep(0.5)

                #------------- append location data of object ------------------------
                tracked_coord.append(iCubSim_wrld_ctrl.get_object_location(objID).tolist())
                time.sleep(0.5)

                if(i in PAUSE):
                    time.sleep(2)
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

        cont_tracker[0].closing_programm()
        cont_tracker[1].closing_programm()
        cont_tracker[2].closing_programm()

        iCubSim_wrld_ctrl.del_all()
        del iCubSim_wrld_ctrl

        return tracked_coord, (tracked_collision_hand, tracked_collision_forearm, tracked_collision_arm)

if __name__ == '__main__':

    obj_1: ObjectTrajectory = ObjectTrajectory()
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


