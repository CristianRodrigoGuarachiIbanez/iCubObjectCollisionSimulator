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
from typing import List, Dict, Tuple, Any, TypeVar, Callable
import yarp
################ Import parameter from parameter file ################
from examples.example_parameter import CLIENT_PREFIX, GAZEBO_SIM, ROBOT_PREFIX
############# Import control classes for both simulators #############
class SceneRecorder:
    def __init__(self) -> None:
        yarp.Network.init()
        self.__input_port_scene: yarp.Port = self.__init_port()
        self.__scene_img_arr, self.__scene_yarp_image = self.__init_scene_camera() # type: np.ndarray, yarp.ImageRgb

        self.__sceneImgArray: List[List] = []
    def getSaveimgArraysInsideTheClass(self, index: int) -> np.ndarray:
        """ calls the function get save image arrays  from Outside
        @returns a numpy array with the scene data from OUTSIDE CLASS while saving it into the constructor INSIDE THE CLASS"""
        return self.__getSaveimgArraysInsideTheClass()

    @staticmethod
    def __getSaveimgArraysInsideTheClass(saveSceneArrays: Callable, readImgArray: Callable) -> np.ndarray:
        """ @returns a numpy array with the scene data OUTSIDE THE CLASS while saving it into the constructor INSIDE THE CLASS"""
        saveSceneArrays()
        return readImgArray()

    def __saveEyeImgArrays(self, index: int) -> None:
        """ saves the eyes image arrays into the class's constructor """
        self.__sceneImgArray.append(self.readSceneCamera(index))

    def __init_port(self) -> yarp.Port:
        """Open and connect ports"""
        input_port_scene: yarp.Port = yarp.Port()
        if not input_port_scene.open("/" + CLIENT_PREFIX + "/scene"):
            print("[ERROR] could not open scene port ")
        if not yarp.Network.connect("/" + ROBOT_PREFIX + "/cam", "/" + CLIENT_PREFIX + "/scene"):
            print("[ERROR] Could not connect to the scene port")

        return input_port_scene

    def __init_scene_camera(self) -> Tuple:
        """ @returns the img array and yarp image"""
        scene_img_array: np.ndarray = np.ones((240, 320, 3), np.uint8)
        scene_yarp_image: yarp.ImageRgb = yarp.ImageRgb()
        scene_yarp_image.resize(320, 240)
        # YARP image will wrap the arr
        scene_yarp_image.setExternal(
            scene_img_array.data, scene_img_array.shape[1], scene_img_array.shape[0])
        return scene_img_array, scene_yarp_image
    @staticmethod
    def __pathCreator(foldername: str = "trials/scene_outputs" ) -> os.path: #"scene_outputs"
        return os.path.dirname(os.path.abspath(__file__)) + "/" + foldername

    @staticmethod
    def __plottingSceneCamera( sceneImg: np.ndarray, pathCreator: os.path, index: int = None) -> None:

        # # create the directory wherein the imgs will be saved
        path: os.path = pathCreator
        if not os.path.exists(path):
            os.mkdir(path)

        # show images
        plt.figure(figsize=(10, 5))
        plt.tight_layout()
        # plt.subplot(121)
        plt.title("scene camera image")
        plt.imshow(sceneImg)
        if not (index):
            plt.savefig(path + "/scene.png")

        plt.savefig(path + "/scene_" + "{index}".format(index=index) + ".png")

    def readSceneCamera(self, index: int = None, plotting: bool = True) -> np.ndarray:
        """get the scena camera data
        @parameter the index and boolean value for the plotting
        @returns a numpy array with the bilder data"""
        self.__input_port_scene.read(self.__scene_yarp_image)
        self.__input_port_scene.read(self.__scene_yarp_image)

        if self.__scene_yarp_image.getRawImage().__int__() != self.__scene_img_arr.__array_interface__['data'][0]:
             print("read() reallocated my scene_yarp_image!")
        if (plotting):
            self.__plottingSceneCamera( self.__scene_img_arr, self.__pathCreator(), index)

        return self.__scene_img_arr

    def closing_program(self) -> None:

        print('----- Close opened ports -----')

        # disconnect the ports
        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/cam", self.__input_port_scene.getName()):
            print("[ERROR] Could not disconnect input_port_scene")

        # close the ports
        self.__input_port_scene.close()


        # close the yarp network
        yarp.Network.fini()




if __name__ == "__main__":

    # scene: SceneRecorder = SceneRecorder()
    # for i in range(5):
    #    print(scene.readSceneCamera())
    # scene.closing_program()
    pass