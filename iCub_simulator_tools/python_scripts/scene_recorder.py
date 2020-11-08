"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
Scene recorder
"""
######################################################################
########################## Import modules  ###########################
######################################################################
import os
import sys
import time
import matplotlib.pylab as plt
import numpy as np
from typing import List, Dict, Tuple, Any, TypeVar
import yarp
################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, GAZEBO_SIM, ROBOT_PREFIX
############# Import control classes for both simulators #############
class SceneRecorder:
    def __init__(self) -> None:
        yarp.Network.init()
        self.__input_port_scene: yarp.Port = self.__init_port()
        self.__scene_img_arr, self.__scene_yarp_image = self.__init_scene_camera() # type: np.ndarray, yarp.ImageRgb


    def __init_port(self) -> yarp.Port:
        """Open and connect ports"""
        input_port_scene: yarp.Port = yarp.Port()
        if not input_port_scene.open("/" + CLIENT_PREFIX + "/scene"):
            print("[ERROR] could not open scene port ")
        if not yarp.Network.connect("/" + ROBOT_PREFIX + "/cam", "/" + CLIENT_PREFIX + "/scene"):
            print("[ERROR] Could not connect to the scene port")

        return input_port_scene

    def __init_scene_camera(self) -> Tuple:
        """ return the img array and yarp image"""
        scene_img_array: np.ndarray = np.ones((240, 320, 3), np.uint8)
        scene_yarp_image: yarp.ImageRgb = yarp.ImageRgb()
        scene_yarp_image.resize(320, 240)
        # YARP image will wrap the arr
        scene_yarp_image.setExternal(
            scene_img_array.data, scene_img_array.shape[1], scene_img_array.shape[0])
        return scene_img_array, scene_yarp_image

    def read_scene(self) -> None:

        self.__input_port_scene.read(self.__scene_yarp_image)
        self.__input_port_scene.read(self.__scene_yarp_image)


        if self.__scene_yarp_image.getRawImage().__int__() != self.__scene_img_arr.__array_interface__['data'][0]:
             print("read() reallocated my scene_yarp_image!")


        # # create the directory wherein the imgs will be saved
        # path: str = os.path.dirname(os.path.abspath(__file__)) + "/img"
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # show images
        plt.figure(figsize=(10, 5))
        plt.tight_layout()
        #plt.subplot(121)
        plt.title("scene camera image")
        plt.imshow(self.__scene_img_arr)
        # plt.savefig(path + "/eye_left_" + "{index}".format(index=i) + ".png")
        plt.savefig("img.png")

        #plt.savefig(path + "/eye_right_" + "{index}".format(index=i) + ".png")


    def close_program(self) -> None:

        print('----- Close opened ports -----')

        # disconnect the ports
        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/cam", self.__input_port_scene.getName()):
            print("[ERROR] Could not disconnect input_port_scene")

        # close the ports
        self.__input_port_scene.close()


        # close the yarp network
        yarp.Network.fini()




if __name__ == "__main__":

    scene: SceneRecorder = SceneRecorder()
    scene.read_scene()
    scene.close_program()