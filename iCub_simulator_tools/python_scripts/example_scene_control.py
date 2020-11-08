"""
Created on Tue July 07 2020

@author: Torsten Follak

Visual perception example

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
#from random_trajectory_generator_3d import *
#from joint_trajectory_movement import *

#import matplotlib.pylab as plt
import numpy as np
import yarp

################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, GAZEBO_SIM, ROBOT_PREFIX

############# Import control classes for both simulators #############
if GAZEBO_SIM:
    import Python_libraries.gazebo_world_controller as gzbo_ctrl
else:
    import Python_libraries.iCubSim_world_controller as iCubSim_ctrl
    import Python_libraries.iCubSim_model_groups_definition as mdl_define
    import Python_libraries.iCubSim_model_groups_control as mdl_ctrl


def objects_gzbo():
    gzbo_wrld_ctrl = gzbo_ctrl.WorldController()

    # create simple objects
    box_id = gzbo_wrld_ctrl.create_object("box", [0.1, 0.1, 0.1], [1, 0.5, 1], [0, 0, 0], [1, 1, 1] )
    cyl_id = gzbo_wrld_ctrl.create_object("cylinder", [0.1, 0.1], [1, 0.5, 1], [0, 0, 0], [1, 1, 1] )
    sphere_id = gzbo_wrld_ctrl.create_object("sphere", [0.1], [1, 0.5, 1], [0, 0, 0], [1, 1, 1] )

    gzbo_wrld_ctrl.set_pose(box_id, [1., 1., 1.], [1, 1, 1])
    print(gzbo_wrld_ctrl.get_list())

    # create complex models
    # two ways:
    # 1. add the model path in the .bashrc to the GAZEBO_MODEL_PATH variable
    # car = gzbo_wrld_ctrl.create_model('/car/car.sdf', [0.5, 1, 0.5], [0, 0, 0])

    # 2. use full path to the model
    model_path = os.path.abspath("../gazebo_environment/object_models")
    car = gzbo_wrld_ctrl.create_model(model_path + '/car/car.sdf', [0.5, 1, 0.5], [0, 0, 0])

    del gzbo_wrld_ctrl

def objects_iCubSim():
    # create simple objects
    iCubSim_wrld_ctrl = iCubSim_ctrl.WorldController()
    #cyl_id = iCubSim_wrld_ctrl.create_object("scyl", [0.1, 0.1], [0, 0.5, 0.3], [1, 1, 1])
    #box_id = iCubSim_wrld_ctrl.create_object("sbox", [0.1, 0.1, 0.1], [1, 0, 0], [1, 1, 1])
    sphere_id = iCubSim_wrld_ctrl.create_object("ssphere", [0.1], [0., 0.7, 0.3], [1, 1, 1])
    time.sleep(15.)
    iCubSim_wrld_ctrl.move_object(sphere_id, [0., 1., 0.3])
    time.sleep(15.)

    #####
    """
    #right_hand_movement()
    #left_hand_movement()
    data_list: list = corrections(20, 0.1)
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            if (j>0):
                break
            else:
                iCubSim_wrld_ctrl.move_object(sphere_id, [data_list[i][j], data_list[i][j+1], data_list[i][j+2]])
                time.sleep(2.)
    """

    #iCubSim_wrld_ctrl.move_object(cyl_id, [0,.5,.4]) # Positivbereich: erst: links/rechts von roboter, zweit: über/unter den Roboter, Dritt: vor/hinter dem Roboter
    #time.sleep(2.)

    ####rotation
    #iCubSim_wrld_ctrl.rotate_object(cyl_id, [0, 0, 0])
    #time.sleep(2.)start

    iCubSim_wrld_ctrl.del_all()

    # create complex models
    #model_path = os.path.abspath("../iCubSim_environment/new_models")

    #bear_dict = mdl_define.dictionary_bear
    #bear = mdl_ctrl.ModelGroup(iCubSim_wrld_ctrl, bear_dict['model_list'], bear_dict['model_type'], [0, 1, 1], bear_dict['start_orient'], model_path + "/bear")
    #time.sleep(1.)

    iCubSim_wrld_ctrl.del_all()
    del iCubSim_wrld_ctrl


if __name__ == '__main__':
    if GAZEBO_SIM:
        objects_gzbo()
    else:
        objects_iCubSim()
