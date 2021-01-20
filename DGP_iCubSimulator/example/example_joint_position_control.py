"""
Created on Mon Apr 28 2020

@author: Torsten Follak

joint position control example

"""

######################################################################
########################## Import modules  ###########################
######################################################################

import sys

import numpy as np
import yarp

############ Import modules with specific functionalities ############
import Python_libraries.YARP_motor_control as mot

################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, ROBOT_PREFIX

######################################################################
######################### Init YARP network ##########################
######################################################################
#mot.motor_init("head", "position", ROBOT_PREFIX, CLIENT_PREFIX)
yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')

######################################################################
################## Init motor control for the head ###################
print('----- Init head motor control -----')

props = yarp.Property()
props.put("device", "remote_controlboard")
props.put("local", "/" + CLIENT_PREFIX + "/head")
props.put("remote", "/" + ROBOT_PREFIX + "/head")

# create remote driver
Driver_head = yarp.PolyDriver(props)

#Driver_arms = yarp.PolyDriver(props)

# query motor control interfaces
iPos_head = Driver_head.viewIPositionControl()
iEnc_head = Driver_head.viewIEncoders()
#
#iPos_arms = Driver_arms.viewIPositionControl()



# retrieve number of joints
jnts_head = iPos_head.getAxes()
#jnts_arms

###################### Go to head zero position ######################
mot.goto_zero_head_pos(iPos_head, iEnc_head, jnts_head)

################ Move the head to predefined position ################
new_pos = yarp.Vector(jnts_head)
pos = np.array([0., 0., 0., 0., 0., 10.]) # x:upDown, y: rightLeft \|/ z: rotation RightLeft, a: eyedirection (same direction) leftRight, ag: eyedirection (opposite directio)  inside/outside
for i in range(jnts_head):
    new_pos.set(i, pos[i])
iPos_head.positionMove(new_pos.data())

# optional, for blocking while moving the joints
motion = False
while not motion:
    motion = iPos_head.checkMotionDone()

######################################################################
##################### Print head joints position #####################
encs = yarp.Vector(jnts_head)
iEnc_head.getEncoders(encs.data())
vector = np.zeros(encs.length(), dtype=np.float64)
for i in range(encs.length()):
    vector[i] = encs.get(i)

print(vector)

######################################################################
######################## Closing the program: ########################
#### Delete objects/models and close ports, network, motor cotrol ####
print('----- Close control devices and opened ports -----')

Driver_head.close()
yarp.Network.fini()
