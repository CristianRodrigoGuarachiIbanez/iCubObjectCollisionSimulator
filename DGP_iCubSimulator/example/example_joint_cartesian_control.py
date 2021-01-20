"""
Created on Mon Apr 28 2020

@author: tofo

joint cartesian control example

"""

######################################################################
########################## Import modules  ###########################
######################################################################

import sys
import time

import numpy as np
import yarp
from hand_trajectory import ford_and_backwards
############ Import modules with specific functionalities ############
import Python_libraries.YARP_motor_control as mot

################ Import parameter from parameter file ################
from example_parameter import (Transfermat_robot2world, Transfermat_world2robot,
                                orientation_robot_hand, pos_hand_world_coord, 
                                CLIENT_PREFIX, ROBOT_PREFIX)

######################################################################
######################### Init YARP network ##########################
######################################################################

yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')


######################################################################
################# init variables from parameter-file #################
print('----- Init variables -----')

T_r2w: Transfermat_robot2world = Transfermat_robot2world
T_w2r: Transfermat_world2robot = Transfermat_world2robot

orient_hand = mot.npvec_2_yarpvec(orientation_robot_hand)

pos = yarp.Vector(3)
orient = yarp.Vector(4)

######################################################################
############## Init cartesian controller for right arm ###############

##################### Prepare a property object ######################
props = yarp.Property()
props.put("device", "cartesiancontrollerclient")
props.put("remote", "/" + ROBOT_PREFIX + "/cartesianController/right_arm")
props.put("local", "/" + CLIENT_PREFIX + "/right_arm")

######################## Create remote driver ########################
Driver_rarm: yarp.PolyDriver = yarp.PolyDriver(props)

################### Query motor control interfaces ###################
iCart = Driver_rarm.viewICartesianControl()
print(Driver_rarm, iCart)
iCart.setPosePriority("position")
time.sleep(1)

############ Move hand to inital position and orientation ############
print('----- Move hand to initial pose -----')
welt_pos = pos_hand_world_coord
init_hand_pos_r_np = np.dot(T_w2r, welt_pos.reshape((4, 1))).reshape((4,))
init_hand_pos_r_yarp = mot.npvec_2_yarpvec(init_hand_pos_r_np[0:3])

iCart.goToPoseSync(init_hand_pos_r_yarp, orient_hand)
iCart.waitMotionDone(timeout=5.0)
time.sleep(1)
iCart.getPose(pos, orient)
print('Hand position:', pos.toString())
print('Hand orientation:', orient.toString())

############ Move hand to new position and orientation ############
print('----- Move hand to new pose -----')

mov_range:list= ford_and_backwards(-2.,2.,20)

for row in mov_range:

    welt_pos_n = np.array([row[1], row[2], row[3], 1.]) #erster: links/rechts, zweiter: oben/unten, dritter: vorn/hinten
    new_hand_pos_r_np: np.dot = np.dot(T_w2r, welt_pos_n.reshape((4, 1))).reshape((4,))
    new_hand_pos_r_yarp: mot.npvec_2_yarpvec = mot.npvec_2_yarpvec(new_hand_pos_r_np[0:3])

    iCart.goToPoseSync(new_hand_pos_r_yarp, orient_hand)
    iCart.waitMotionDone(timeout=5.0)
    time.sleep(0.5)
    print('Hand preorientation:', iCart.getPose(pos, orient))
    iCart.getPose(pos, orient)
    print('Hand position:', pos.toString())
    print('Hand orientation:', orient.toString())
    time.sleep(1)

######################################################################
######################## Closing the program: ########################
#### Delete objects/models and close ports, network, motor cotrol ####
print('----- Close control devices and opened ports -----')

Driver_rarm.close()
yarp.Network.fini()


