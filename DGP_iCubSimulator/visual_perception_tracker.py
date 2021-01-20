"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
Visual perception tracker
"""


######################################################################
########################## Import modules  ###########################
######################################################################
import os
import numpy as np
import yarp
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

#from example_scene_control import objects_iCubSim
from typing import List, Tuple, Any, TypeVar, Callable
from numpy import ndarray

################ Import parameter from parameter file ################
from examples.example_parameter import CLIENT_PREFIX, ROBOT_PREFIX


class VisualPerceptionTracker:
    def __init__(self) -> None:
        yarp.Network.init()
        # print("---------------Init YARP network-----------------")
        #--------------------open eye ports and connect input port eyes ------------
        self.__init_port_right_eye, self.__init_port_left_eye = self.__init_YARP_ports() # type: yarp.Port, yarp.Port

        #------------------- Init both eye images-----------------------------
        self.__right_eye_yarp_image, self.__right_eye_img_array, self.__left_eye_yarp_image, self.__left_eye_img_array = self.__initilizeBinocularImages() # type: arp.ImageRgb

        self.__leftEyeImgArray: List[ndarray] = [self.__left_eye_img_array]
        self.__rightEyeImgArray: List[ndarray] = [self.__right_eye_img_array]

    def getFinalEyeImgArrays(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        calls the function get save eye image arrays  from Outside the class
        @:return: a tuple of numpy arrays with the image information """
        return self.__rightEyeImgArray, self.__leftEyeImgArray

    def getSaveEyeImgArraysFromOutside(self, index: int = None) -> Tuple[ndarray, ndarray]:
        """
        @:param index; a integer value
        @:return: tuple of numpy arrays from OUTSIDE THE CLASS while saving the position coordinates into the class's constructor """
        return self.__getSaveEyeimgArraysInsideTheClass(self.readCameraImagesFromOutside(index), self.__saveEyeImgArrays())

    @staticmethod
    def __getSaveEyeimgArraysInsideTheClass(readImgArray: Tuple[ndarray, ndarray], saveEyeArrays: Any) -> Tuple[ndarray, ndarray]:
        """ @returns the eyes image array as a numpy array INSIDE THE CLASS while saving the position coordinates into the class's constructor """
        saveEyeArrays
        return readImgArray


    def __saveEyeImgArrays(self) -> None:
        """ saves the eyes image arrays into the class's constructor """
        self.__rightEyeImgArray.append(self.readCameraImagesFromOutside()[0])
        self.__leftEyeImgArray.append(self.readCameraImagesFromOutside()[1])

    def __init_YARP_ports(self) -> Tuple[Any, Any]:

        # print('------------- Opened ports for eyes ---------------')
        # Initialization of all needed portsFRAMES
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

        return input_port_right_eye, input_port_left_eye

    @staticmethod
    def __createAPathToSaveImages(foldername: str = "trials/image_outputs" ) -> os.path: #"image_outputs"
        # create the directory wherein the imgs will be saved
        return os.path.dirname(os.path.abspath(__file__)) + "/" + foldername

    @staticmethod
    def __plottingBinocularPerception(pathCreator: Any, ArrayRightEye: np.ndarray, ArrayLeftEye: np.ndarray, index: int = None) -> None:
        """plot the arrays containing the information of the eye cameras"""
        path: os.path = pathCreator
        if not os.path.exists(path):
            os.mkdir(path)

        #print("right eye", ArrayRightEye.shape)
        #print("left eye:", ArrayLeftEye.shape)
        # show images
        plt.figure(figsize=(10, 5))
        plt.tight_layout()
        plt.subplot(121)
        plt.title("Left camera image")
        plt.imshow(ArrayLeftEye)

        plt.subplot(122)
        plt.title("Right camera image")
        plt.imshow(ArrayRightEye)
        # plt.show()

        if not (index):
            plt.savefig(path + "/binocular_view.png")

        plt.savefig(path + "/binocular_view_" + "{index}".format(index=index) + ".png")

    def __initilizeBinocularImages(self) -> Tuple[yarp.ImageRgb, np.ndarray, yarp.ImageRgb, np.ndarray]:
        """  Create numpy array to receive the image and the YARP image wrapped around it """
        # print('------------- Init image-array structures -----------------')
        # YARP images for both eyes and the arr
        left_eye_img_array: np.ndarray = np.ones((240, 320, 3), np.uint8)
        left_eye_yarp_image: yarp.ImageRgb = yarp.ImageRgb()
        left_eye_yarp_image.resize(320, 240)

        right_eye_img_array: np.ndarray = np.ones((240, 320, 3), np.uint8)
        right_eye_yarp_image: yarp.ImageRgb = yarp.ImageRgb()
        right_eye_yarp_image.resize(320, 240)
        # YARP image will wrap the arr
        left_eye_yarp_image.setExternal(
            left_eye_img_array.data, left_eye_img_array.shape[1], left_eye_img_array.shape[0])
        right_eye_yarp_image.setExternal(
            right_eye_img_array.data, right_eye_img_array.shape[1], right_eye_img_array.shape[0])

        return right_eye_yarp_image, right_eye_img_array, left_eye_yarp_image, left_eye_img_array

    def readCameraImagesFromOutside(self, index: int = None, plotting: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Read the images from the robot cameras"""

        # print('--------- reading data from both robot camaras---------')
        self.__init_port_left_eye.read(self.__left_eye_yarp_image)
        self.__init_port_left_eye.read(self.__left_eye_yarp_image)
        self.__init_port_right_eye.read(self.__right_eye_yarp_image)
        self.__init_port_right_eye.read(self.__right_eye_yarp_image)

        if self.__left_eye_yarp_image.getRawImage().__int__() != self.__left_eye_img_array.__array_interface__['data'][0]:
            print("read() reallocated my left_eye_yarp_image!")
        if self.__right_eye_yarp_image.getRawImage().__int__() != self.__right_eye_img_array.__array_interface__['data'][0]:
            print("read() reallocated my right_eye_yarp_image!")

        if (plotting):
            self.__plottingBinocularPerception(self.__createAPathToSaveImages(), self.__right_eye_img_array, self.__left_eye_img_array, index)

        return self.__right_eye_img_array, self.__left_eye_img_array

    def closing_program(self):
        """Closing the program: Delete objects/models and close ports, network, motor cotrol """

        print('----------------- Close opened ports -------------------')
        # disconnect the ports
        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/cam/right", self.__init_port_right_eye.getName()):
            print("[ERROR] Could not disconnect input_port_right_eye")

        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/cam/left", self.__init_port_left_eye.getName()):
            print("[ERROR] Could not disconnect input_port_left_eye")

        # close the ports
        self.__init_port_right_eye.close()
        self.__init_port_left_eye.close()

        # close the yarp network
        yarp.Network.fini()

if __name__ == '__main__':
    pass

    # percep: VisualPerception = VisualPerception()
    # with PdfPages("binocular.pdf") as pdf:
    #     for i in range(10):
    #         print(i)
    #         #percep.readCameraImagesFromOutside(i)
    #         percep.getSaveEyeImgArraysFromOutside(i)
    #         pdf.savefig()
    #     #pdf.close()
    # print("final:", percep.getFinalEyeImgArrays())
    # percep.closing_program()