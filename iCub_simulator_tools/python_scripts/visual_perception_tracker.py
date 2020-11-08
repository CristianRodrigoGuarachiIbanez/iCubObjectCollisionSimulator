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

from example_scene_control import objects_iCubSim
from typing import List, Tuple, Any, TypeVar
################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, ROBOT_PREFIX


class VisualPerception:
    def __init__(self):
        yarp.Network.init()
        print("---------------Init YARP network-----------------")
        #--------------------open eye ports and connect input port eyes ------------
        self.__init_port_right_eye, self.__init_port_left_eye = self.__init_YARP_ports()
        #------------------- Init both eye images-----------------------------
        self.__right_eye_yarp_image, self.__right_eye_img_array, self.__left_eye_yarp_image, self.__left_eye_img_array = self.__init_both_eye_images()
    def __init_YARP_ports(self) -> Tuple[Any, Any]:

        print('----- Opened ports for eyes -----')
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

        return input_port_right_eye, input_port_left_eye

    def __init_both_eye_images(self) -> Tuple[yarp.ImageRgb, np.ndarray, yarp.ImageRgb, np.ndarray]:
        """  Create numpy array to receive the image and the YARP image wrapped around it """
        print('----- Init image-array structures -----')
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

    def read_camera_images(self, index: int):
        """Read the images from the robot cameras"""

        print('--------- reading data from both robot camaras---------')
        self.__init_port_left_eye.read(self.__left_eye_yarp_image)
        self.__init_port_left_eye.read(self.__left_eye_yarp_image)
        self.__init_port_right_eye.read(self.__right_eye_yarp_image)
        self.__init_port_right_eye.read(self.__right_eye_yarp_image)

        if self.__left_eye_yarp_image.getRawImage().__int__() != self.__left_eye_img_array.__array_interface__['data'][0]:
            print("read() reallocated my left_eye_yarp_image!")
        if self.__right_eye_yarp_image.getRawImage().__int__() != self.__right_eye_img_array.__array_interface__['data'][0]:
            print("read() reallocated my right_eye_yarp_image!")

        # create the directory wherein the imgs will be saved
        path: os.path = os.path.dirname(os.path.abspath(__file__)) + "/img"
        if not os.path.exists(path):
            os.mkdir(path)

        # show images
        plt.figure(figsize=(10, 5))
        plt.tight_layout()
        plt.subplot(121)
        plt.title("Left camera image")
        plt.imshow(self.__left_eye_img_array)

        plt.subplot(122)
        plt.title("Right camera image")
        plt.imshow(self.__right_eye_img_array)
        #plt.show()
        if not (index):
            plt.savefig(path + "/eye_right.png")
        plt.savefig(path + "/eye_right_" + "{index}".format(index=index) + ".png")

    def closing_program(self):
        """Closing the program: Delete objects/models and close ports, network, motor cotrol """

        print('----- Close opened ports -----')
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

# if __name__ == '__main__':
#
#     percep: VisualPerception = VisualPerception()
#     with PdfPages("binocular.pdf") as pdf:
#         for i in range(10):
#             percep.read_camera_images(i)
#             pdf.savefig()
#         pdf.close()
#     percep.closing_program()