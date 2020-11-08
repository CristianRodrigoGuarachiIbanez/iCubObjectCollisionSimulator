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
import sys
sys.path.insert(0, "~/PycharmProjects/projectPy3.8/iclub_simulator_tools/python_scripts/")
from random_trajectory_generator_3d import corrections
from hand_trajectory_tracker import *
from tabulate import tabulate
from typing import List, Dict, Tuple, TypeVar
import numpy as np
import yarp
from visual_perception_tracker import VisualPerception
from tactil_perception_tracker import TactilCollisionDetector
from export_collision_data import Writer, Writer2

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
        #--------------------------------------------------------------------------------
        if isinstance(typObj, str):
            object: iCubSim_ctrl.WorldController = self._objects_iCubSim_ID(typObj)
            time.sleep(1.)
            self.location, self.collision = self._object_iCubSim(object)


        # elif isinstance(typObj, list):
        #     objects: List = self._objects_iCubSim_multi_IDs(typObj)
        #     self.__location: List[List, ...] = []
        #     for obj in objects: # oder objects
        #         print(obj)
        #         time.sleep(1.)
        #         joint_loc: List = self._object_iCubSim(obj)
        #         self.__location.append(joint_loc)


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
        contactSensors: Tuple = self.__collisionDect_hand, self.__collisionDect_foreamr, self.__collisionDect_arm
        return self._object_movement(object, self.__iCubSim_wrld_ctrl, contactSensors, self.__visualPerc)

    def _objects_iCubSim_ID(self, typObj:str) -> iCubSim_ctrl.WorldController:
        """ create a object ID according to "sbox": [1, 0, 0], "scyl": [0, 0.5, 0.3], "ssphere" """
        if(typObj):
            sphere_id = self.__iCubSim_wrld_ctrl.create_object(typObj, [0.1], [0, 1., 0.3], [1, 1, 1])
            return sphere_id

    @staticmethod
    def _object_movement(objID: iCubSim_ctrl.WorldController, iCubSim_wrld_ctrl: iCubSim_ctrl.WorldController, cont_tracker: Tuple, visual_tracker: VisualPerception) -> Tuple[List, Tuple]:
        """create random area corrected trajectory"""

        tracked_collision_hand: List[List,...] = []
        tracked_collision_forearm : List[List,...] = []
        tracked_collision_arm: List[List,...] = []
        tracked_coord: List[List, ...] = []
        # create the directory wherein the imgs will be saved
        path: str = os.path.dirname(os.path.abspath(__file__)) + "/img"
        if not os.path.exists(path):
            os.mkdir(path)
        data_list: List = corrections(20, 0.1)
        #---------------------initial phase for appending the collision data----------
        tracked_collision_hand.append(cont_tracker[0].skin_sensor_reader())
        time.sleep(0.5)
        tracked_collision_forearm.append(cont_tracker[1].skin_sensor_reader())
        time.sleep(0.5)
        tracked_collision_arm.append(cont_tracker[2].skin_sensor_reader())
        #---------------------initial phase for appending the visual perception------
        visual_tracker.read_camera_images()
        #---------------------initial phase for appending location data -------------
        tracked_coord.append(iCubSim_wrld_ctrl.get_object_location(objID).tolist())

        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                if(j>0):
                    break
                else:
                    #-------------- move the object -----------------------------
                    iCubSim_wrld_ctrl.move_object(objID, [data_list[i][j], data_list[i][j+1], data_list[i][j+2]])
                    #------------- append collision data of the hand -----------------------
                    tracked_collision_hand.append(cont_tracker[0].skin_sensor_reader())
                    time.sleep(0.5)
                    tracked_collision_forearm.append(cont_tracker[1].skin_sensor_reader())
                    time.sleep(0.5)
                    tracked_collision_arm.append(cont_tracker[2].skin_sensor_reader())
                    #--------------append visual perception data ---------------
                    visual_tracker.read_camera_images(i)
                    #------------- append location data of object ------------------------
                    tracked_coord.append(iCubSim_wrld_ctrl.get_object_location(objID).tolist())
                    time.sleep(2.)

        cont_tracker[0].closing_programm()
        cont_tracker[1].closing_programm()
        cont_tracker[2].closing_programm()
        visual_tracker.closing_program()

        iCubSim_wrld_ctrl.del_all()
        del iCubSim_wrld_ctrl

        return tracked_coord, (tracked_collision_hand, tracked_collision_forearm, tracked_collision_arm)

if __name__ == '__main__':
    #time_to_contact: TimeToContact = TimeToContact()
    obj_1: ObjectTrajectory = ObjectTrajectory()
    dfLoc: Writer2 = Writer2(obj_1.location)
    print("------------Exporting Location Data---------------")
    dfLoc.export_dataframe("location.csv", "csv")
    dfColli_hand: Writer = Writer(obj_1.collision[0])
    dfColli_forearm: Writer = Writer(obj_1.collision[1])
    dfColli_arm: Writer = Writer(obj_1.collision[2])
    print("------------Exporting Collition Data---------------")
    dfColli_hand.export_dataframe("collision_right_hand.csv", "collision_left_hand.csv", "csv")
    dfColli_forearm.export_dataframe("collision_right_forearm.csv", "collision_left_forearm.csv", "csv")
    dfColli_arm.export_dataframe("collision_right_arm.csv", "collision_left_arm.csv", "csv")


