"""
Created on Tue July 07 2020

@author: Torsten Follak

Visual perception example

"""

######################################################################
########################## Import modules  ###########################
######################################################################

import sys

import numpy as np
import os
import yarp
import matplotlib.pylab as plt
from example_scene_control import objects_iCubSim
################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, ROBOT_PREFIX


######################################################################
######################### Init YARP network ##########################
######################################################################

print('----- Init network -----')
# network initialization and check
yarp.Network.init()
if not yarp.Network.checkNetwork():
    print('[ERROR] Please try running yarp server')

print('----- Opened ports -----')
# Initialization of all needed ports
# Port for right eye image
input_port_right_eye = yarp.Port()
if not input_port_right_eye.open("/" + CLIENT_PREFIX + "/eyes/right"):
    print("[ERROR] Could not open right eye port")
if not yarp.Network.connect("/" + ROBOT_PREFIX + "/cam/right", "/" + CLIENT_PREFIX + "/eyes/right"):
    print("[ERROR] Could not connect input_port_right_eye")

# Port for left eye image
input_port_left_eye = yarp.Port()
if not input_port_left_eye.open("/" + CLIENT_PREFIX + "/eyes/left"):
    print("[ERROR] Could not open left eye port")
if not yarp.Network.connect("/" + ROBOT_PREFIX + "/cam/left", "/" + CLIENT_PREFIX + "/eyes/left"):
    print("[ERROR] Could not connect input_port_left_eye")


######################################################################
################## Initialization of both eye images #################

print('----- Init image array structures -----')
# Create numpy array to receive the image and the YARP image wrapped around it
left_eye_img_array = np.ones((240, 320, 3), np.uint8)
left_eye_yarp_image = yarp.ImageRgb()
left_eye_yarp_image.resize(320, 240)

right_eye_img_array = np.ones((240, 320, 3), np.uint8)
right_eye_yarp_image = yarp.ImageRgb()
right_eye_yarp_image.resize(320, 240)

left_eye_yarp_image.setExternal(
    left_eye_img_array.data, left_eye_img_array.shape[1], left_eye_img_array.shape[0])
right_eye_yarp_image.setExternal(
    right_eye_img_array.data, right_eye_img_array.shape[1], right_eye_img_array.shape[0])

#objects_iCubSim()
######################################################################
################### Read camera images from robot ####################
print('----- Read images from robot cameras -----')
for i in range(10):
    print("Image:", i)
    # Read the images from the robot cameras
    input_port_left_eye.read(left_eye_yarp_image)
    input_port_left_eye.read(left_eye_yarp_image)
    input_port_right_eye.read(right_eye_yarp_image)
    input_port_right_eye.read(right_eye_yarp_image)

    if left_eye_yarp_image.getRawImage().__int__() != left_eye_img_array.__array_interface__['data'][0]:
        print("read() reallocated my left_eye_yarp_image!")
    if right_eye_yarp_image.getRawImage().__int__() != right_eye_img_array.__array_interface__['data'][0]:
        print("read() reallocated my right_eye_yarp_image!")
    # create the directory wherein the imgs will be saved
    path: str = os.path.dirname(os.path.abspath(__file__)) + "/img"
    if not os.path.exists(path):
        os.mkdir(path)
    # show images
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.subplot(121)
    plt.title("Left camera image")
    plt.imshow(left_eye_img_array)
    #plt.savefig(path + "/eye_left_" + "{index}".format(index=i) + ".png")

    plt.subplot(122)
    plt.title("Right camera image")
    plt.imshow(right_eye_img_array)
    #plt.show()#

    plt.savefig(path + "/eye_right_" + "{index}".format(index=i) + ".png")

######################################################################
######################## Closing the program: ########################
#### Delete objects/models and close ports, network, motor cotrol ####
print('----- Close opened ports -----')

# disconnect the ports
if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/cam/right", input_port_right_eye.getName()):
    print("[ERROR] Could not disconnect input_port_right_eye")

if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/cam/left", input_port_left_eye.getName()):
    print("[ERROR] Could not disconnect input_port_left_eye")

# close the ports
input_port_right_eye.close()
input_port_left_eye.close()

# close the yarp network
yarp.Network.fini()