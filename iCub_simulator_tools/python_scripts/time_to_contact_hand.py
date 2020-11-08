import sys
import time
from typing import List, Tuple, Any
import numpy as np
import yarp

############ Import modules with specific functionalities ############

################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, ROBOT_PREFIX, skin_idx_r_hand, skin_idx_l_hand
#from object_trajectory_tracker import ObjectTrajectory
######################################################################
######################### Init YARP network ##########################
######################################################################
print('----- Init network -----')
yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')

class TimeToContact:
    def __init__(self):
        yarp.Network.init()
        ######################### Init YARP network ##########################
        print('----- Init network -----')
        self.input_port_skin_hand, self.input_port_skin_hand_l = self._open_connect_YARP_Port()

        ####################### reader of skin sensor data ###################
        #self._skin_sensor_data_reader()

        ######################## Closing the program: ########################
        #print('----- Close opened ports -----')
        #self._closinf_programm()
        #yarp.Network.fini()


    def _open_connect_YARP_Port(self) -> Tuple[Any, Any]:

        # Open and connect YARP-Port to read right hand skin sensor data
        input_port_skin_hand = yarp.Port()
        if not input_port_skin_hand.open("/" + CLIENT_PREFIX + "/skin_read/hand"):
            print("[ERROR] Could not open skin hand port!")
        if not yarp.Network.connect("/" + ROBOT_PREFIX + "/skin/right_hand_comp", "/" + CLIENT_PREFIX + "/skin_read/hand"):
            print("[ERROR] Could not connect skin hand port!")

        input_port_skin_hand_l = yarp.Port()
        if not input_port_skin_hand_l.open("/" + CLIENT_PREFIX + "/skin_read/hand"):
            print("[ERROR] Could not open skin hand port!")
        if not yarp.Network.connect("/" + ROBOT_PREFIX + "/skin/left_hand_comp", "/" + CLIENT_PREFIX + "/skin_read/hand"):
            print("[ERROR] Could not connect skin hand port!")

        return input_port_skin_hand, input_port_skin_hand_l


    def skin_sensor_data_reader(self) -> Tuple[List, List]:
        """Read skin sensor data """

        tactile_hand: yarp.Vector = yarp.Vector(192)
        self.input_port_skin_hand.read(tactile_hand)

        tactile_hand_l = yarp.Vector(192)
        self.input_port_skin_hand_l.read(tactile_hand_l)


        data_hand: List = []
        data_hand_l: List = []

        for j in range(len(skin_idx_r_hand)):
            #print(skin_idx_hand[j])
            if skin_idx_r_hand[j] > 0:
                print("read Data:",tactile_hand.get(j))
                data_hand.append(tactile_hand.get(j))

        for j in range(len(skin_idx_l_hand)):
            if skin_idx_l_hand[j] > 0:
                print("read Data left:", tactile_hand_l.get(j))
                data_hand_l.append(tactile_hand_l.get(j))

        print("Data hand:")
        print(data_hand)
        print(data_hand_l)
        time.sleep(0.5)

        return data_hand, data_hand_l

    def closinf_programm(self):
        """Delete objects/models and close ports, network, motor cotrol """

        print('----- Close opened ports -----')
        # disconnect the ports

        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/skin/right_hand_comp", self.input_port_skin_hand.getName()):
            print("[ERROR] Could not disconnect skin hand port!")

        if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/skin/left_hand_comp", self.input_port_skin_hand_l.getName()):
            print("[ERROR] Could not disconnect skin hand port!")

        #close the ports
        self.input_port_skin_hand.close()
        self.input_port_skin_hand_l.close()

        yarp.Network.fini()

if __name__ == "__main__":
    contact: TimeToContact = TimeToContact()
    contact.skin_sensor_data_reader()
    contact.closinf_programm()
