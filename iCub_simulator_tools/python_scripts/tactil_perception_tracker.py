"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
Tactil perception tracker
"""


import sys
import time
from typing import List, Tuple, Any, TypeVar
import numpy as np
import yarp

############ Import modules with specific functionalities ############

################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, ROBOT_PREFIX, skin_idx_r_hand, skin_idx_l_hand, skin_idx_r_forearm, skin_idx_l_forearm, skin_idx_r_arm, skin_idx_l_arm

######################################################################
######################### Init YARP network ##########################
######################################################################
#print('----- Init network -----')
yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')

class TactilCollisionDetector:
    A = TypeVar("A", bool, List)
    def __init__(self, armSec:int):
        yarp.Network.init()

        self.__armSeccion, self.__armSide_right, self.__armSide_left, self.flag = self.__keyConnection(armSec)
        self.__idx_skin_right, self.__idx_skin_left, self.numSensors = self.__idx_skin_arm(armSec)

        print('----- Init network -----')
        self.__input_port_skin_right, self.__input_port_skin_left = self.__init_YARP_port()

    def __idx_skin_arm(self, part: int) -> Tuple[List, List, int]:
        if (part == 0):
            return skin_idx_r_hand, skin_idx_l_hand, 192

        elif (part == 1):
            return skin_idx_r_forearm, skin_idx_l_forearm, 384

        elif (part == 2):
            return skin_idx_r_arm, skin_idx_l_arm, 768

    def __keyConnection(self, part: int) -> Tuple:

        if (part == 0):
            flag: str = "hand"
            armSeccion: str = "/skin_read/hand"  # /skin_read/forearm"

            armSide_right: str = "/skin/right_hand_comp"  # "/skin/right_hand_comp",
            armSide_left: str = "/skin/left_hand_comp"

        elif (part == 1):
            flag: str = "forearm"
            armSeccion: str = "/skin_read/forearm"

            armSide_right: str = "/skin/right_forearm_comp"
            armSide_left: str = "/skin/left_forearm_comp"

        elif (part == 2):
            flag: str = "arm"
            armSeccion: str = "/skin_read/arm"

            armSide_right: str = "/skin/right_arm_comp"
            armSide_left: str = "/skin/left_arm_comp"

        return armSeccion, armSide_right, armSide_left, flag


    def __init_YARP_port(self) -> Tuple[Any, Any]:

        """ Open and connect YARP-Port to read right arm skin sensor data"""

        input_port_skin_right:yarp.Port = yarp.Port()
        if not input_port_skin_right.open("/" + CLIENT_PREFIX + self.__armSeccion + "_right"):
            print("[ERROR] Could not open skin {} port!".format(self.__armSeccion))
        if not yarp.Network.connect("/" + ROBOT_PREFIX + self.__armSide_right, "/" + CLIENT_PREFIX + self.__armSeccion + "_right"):
            print("[ERROR] Could not connect skin {} port!".format(self.__armSide_right))

        input_port_skin_left: yarp.Port = yarp.Port()
        if not input_port_skin_left.open("/" + CLIENT_PREFIX + self.__armSeccion + "_left"):
            print("[ERROR] Could not open skin {} port!".format(self.__armSeccion))
        if not yarp.Network.connect("/" + ROBOT_PREFIX + self.__armSide_left, "/" + CLIENT_PREFIX + self.__armSeccion + "_left"):
            print("[ERROR] Could not connect skin {} port!".format(self.__armSide_left))

        return input_port_skin_right, input_port_skin_left

    def skin_sensor_reader(self) -> Tuple[List, List]:
        """ return the data from both skin sensors"""
        print('--------------reading data sensors of the {}-------------'.format(self.flag) )
        return self.__skin_sensor_data_reader_right(), self.__skin_sensor_data_reader_left()

    def __skin_sensor_data_reader_right(self) -> A:
        """Read skin sensor data from the right hand"""
        print('---------------right {} with {} sensors-------------------'.format(self.flag, self.numSensors))
        tactile_arm: yarp.Vector = yarp.Vector(self.numSensors)
        while(not self.__input_port_skin_right.read(tactile_arm)):
            print("none right conection!")
            yarp.delay(0.001)
            self.__input_port_skin_right.read(tactile_arm)
        #self.__input_port_skin_right.read(tactile_arm)

        data_hand: List = []
        for j in range(len(self.__idx_skin_right)):
            if self.__idx_skin_right[j] > 0:
                #print("read Data Right: {}".format(j),tactile_hand.get(j))
                data_hand.append(tactile_arm.get(j))

        print("Data right {}:".format(self.flag))
        print(data_hand)
        #time.sleep(0.5)
        return data_hand

    def __skin_sensor_data_reader_left(self) -> A:
        """Read skin sensor data from the left hand """
        print('---------------left {} with {} sensors-------------------'.format(self.flag, self.numSensors))
        tactile_arm_l: yarp.Vector = yarp.Vector(self.numSensors)
        while(not self.__input_port_skin_left.read(tactile_arm_l)):
            print("none left conection!")
            yarp.delay(0.001)
            self.__input_port_skin_left.read(tactile_arm_l)
        #self.__input_port_skin_left.read(tactile_arm_l)

        data_hand_l: List = []
        for j in range(len(self.__idx_skin_left)):
            if self.__idx_skin_left[j] > 0:
                #print("read Data Left: {}".format(j), tactile_hand_l.get(j))
                data_hand_l.append(tactile_arm_l.get(j))

        print("Data left {}:".format(self.flag))
        print(data_hand_l)
        #time.sleep(0.5)
        return data_hand_l

    def closing_programm(self):
        """Delete objects/models and close ports, network, motor cotrol """

        print('----- Close opened ports -----')
        # disconnect the ports

        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + self.__armSide_right, self.__input_port_skin_right.getName()):
            print("[ERROR] Could not disconnect skin {} port!".format(self.__armSide_right))

        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + self.__armSide_left, self.__input_port_skin_left.getName()):
            print("[ERROR] Could not disconnect skin {} port!".format(self.__armSide_left))

        #close the ports
        self.__input_port_skin_right.close()
        self.__input_port_skin_left.close()

        yarp.Network.fini()


if __name__ == "__main__":
    contact: TactilCollisionDetector = TactilCollisionDetector(armSec=1)
    contact.skin_sensor_reader()
    #contact.skin_sensor_reader()
    time.sleep(0.5)
    contact.closing_programm()


