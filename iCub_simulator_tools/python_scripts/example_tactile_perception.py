"""
Created on Tue July 07 2020
@author: Torsten Follak
Tactile perception example
"""
######################################################################
########################## Import modules  ###########################
######################################################################
import sys
import time
import numpy as np
import yarp
############ Import modules with specific functionalities ############
################ Import parameter from parameter file ################
from example_parameter import CLIENT_PREFIX, ROBOT_PREFIX, MODE, PATH_TO_INTERFACE_BUILD, INTERFACE_INI_PATH, skin_idx_r_arm, skin_idx_r_forearm, skin_idx_r_hand, skin_idx_l_arm, skin_idx_l_forearm, skin_idx_l_hand
######################################################################
######################### Init YARP network ##########################
######################################################################
print('----- Init network -----')
yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')
# tactile perception with yarp commands
def tactile_prcptn_yarp():
    # Open and connect YARP-Port to read right upper arm skin sensor data
    input_port_skin_rarm = yarp.Port()
    if not input_port_skin_rarm.open("/" + CLIENT_PREFIX + "/skin_read/rarm"):
        print("[ERROR] Could not open skin arm port")
    if not yarp.Network.connect("/" + ROBOT_PREFIX + "/skin/right_arm_comp", input_port_skin_rarm.getName()):
        print("[ERROR] Could not connect skin arm port!")
    # Open and connect YARP-Port to read right forearm skin sensor data
    input_port_skin_rforearm = yarp.Port()
    if not input_port_skin_rforearm.open("/" + CLIENT_PREFIX + "/skin_read/rforearm"):
        print("[ERROR] Could not open skin forearm port!")
    if not yarp.Network.connect("/" + ROBOT_PREFIX + "/skin/right_forearm_comp", input_port_skin_rforearm.getName()):
        print("[ERROR] Could not connect skin forearm port!")
    # Open and connect YARP-Port to read right hand skin sensor data
    input_port_skin_rhand = yarp.Port()
    if not input_port_skin_rhand.open("/" + CLIENT_PREFIX + "/skin_read/rhand"):
        print("[ERROR] Could not open skin hand port!")
    if not yarp.Network.connect("/" + ROBOT_PREFIX + "/skin/right_hand_comp", input_port_skin_rhand.getName()):
        print("[ERROR] Could not connect skin hand port!")
    ######################################################################
    ####################### Read skin sensor data ########################
    ######################################################################
    for i in range(10):
        tactile_rarm = yarp.Vector(768)
        while(not input_port_skin_rarm.read(tactile_rarm)):
            yarp.delay(0.001)
        tactile_rforearm = yarp.Vector(384)
        while(not input_port_skin_rforearm.read(tactile_rforearm)):
            yarp.delay(0.001)

        tactile_rhand = yarp.Vector(192)
        while(not input_port_skin_rhand.read(tactile_rhand)):
            yarp.delay(0.001)

        data_rarm = []
        data_rforearm = []
        data_rhand = []
        for j in range(len(skin_idx_r_arm)):
            if skin_idx_r_arm[j] > 0:
                data_rarm.append(tactile_rarm.get(j))
        for j in range(len(skin_idx_r_forearm)):
            if skin_idx_r_forearm[j] > 0:
                data_rforearm.append(tactile_rforearm.get(j))
        for j in range(len(skin_idx_r_hand)):
            if skin_idx_r_hand[j] > 0:
                data_rhand.append(tactile_rhand.get(j))
        print("Data arm:")
        print(data_rarm)
        print("Data forearm:")
        print(data_rforearm)
        print("Data hand:")
        print(data_rhand)
        time.sleep(2.0)
    ######################################################################
    ######################## Closing the program: ########################
    #### Delete objects/models and close ports, network, motor cotrol ####
    print('----- Close opened ports -----')
    # disconnect the ports
    if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/skin/right_arm_comp", input_port_skin_rarm.getName()):
        print("[ERROR] Could not disconnect skin arm port!")
    if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/skin/right_forearm_comp", input_port_skin_rforearm.getName()):
        print("[ERROR] Could not disconnect skin forearm port!")
    if not yarp.Network.disconnect("/" + ROBOT_PREFIX + "/skin/right_hand_comp", input_port_skin_rhand.getName()):
        print("[ERROR] Could not disconnect skin hand port!")
    # close the ports
    input_port_skin_rarm.close()
    input_port_skin_rforearm.close()
    input_port_skin_rhand.close()
    yarp.Network.fini()
# tactile perception with iCub-ANNarchy-Interface
def tactile_prcptn_iCub_ANN():
    if len(PATH_TO_INTERFACE_BUILD) == 0:
        import iCub_Interface
    else:
        sys.path.append(PATH_TO_INTERFACE_BUILD)
        import iCub_Interface
    # interface wrapper
    iCub = iCub_Interface.iCubANN_wrapper()
    print(sys.path)
    # add skin reader instance
    iCub.add_skin_reader("skin_right")
    # init skin reader module
    iCub.tactile_reader["skin_right"].init("r", True, ini_path=INTERFACE_INI_PATH)
    for i in range(10):
        # read tactile data
        iCub.tactile_reader["skin_right"].read_tactile()
        # print tactile data
        print("Data arm:")
        print(iCub.tactile_reader["skin_right"].get_tactile_arm())
        print("Data forearm:")
        print(iCub.tactile_reader["skin_right"].get_tactile_forearm())
        print("Data hand:")
        print(iCub.tactile_reader["skin_right"].get_tactile_hand())
    iCub.tactile_reader["skin_right"].close()
    del iCub
if __name__ == '__main__':
    if MODE == "yarp":
        tactile_prcptn_yarp()
    elif MODE == "iCub_ANN":
        tactile_prcptn_iCub_ANN()
    else:
        print("No valid MODE given! Check the example_parameter file!")