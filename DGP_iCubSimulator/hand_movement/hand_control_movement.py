import sys
import time
import numpy as np
import yarp
############ Import modules with specific functionalities ############
import Python_libraries.YARP_motor_control as mot
################ Import parameter from parameter file ################
from example_parameter import (Transfermat_robot2world, Transfermat_world2robot,
                               orientation_robot_hand, pos_hand_world_coord, pos_hand_world_coord_new,
                               CLIENT_PREFIX, ROBOT_PREFIX, MODE)
T_r2w = Transfermat_robot2world
T_w2r = Transfermat_world2robot
orient_hand = mot.npvec_2_yarpvec(orientation_robot_hand)
######################################################################
######################### Init YARP network ##########################
yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')
def cartesian_ctrl_yarp():
    ######################################################################
    ################# init variables from parameter-file #################
    print('----- Init variables -----')
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
    driver_rarm = yarp.PolyDriver(props)
    if not driver_rarm:
        sys.exit("Motor initialization failed!")
    ################### Query motor control interfaces ###################
    iCart = driver_rarm.viewICartesianControl()
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
    print('Hand rientation:', orient.toString())
    ############ Move hand to new position and orientation ############
    print('----- Move hand to new pose -----')
    pos_hand_world_coord_new: np.ndarray = np.array([-0.111577, 0.27158, 0.501089, 0.1])
    welt_pos_n = pos_hand_world_coord_new
    new_hand_pos_r_np = np.dot(T_w2r, welt_pos_n.reshape((4, 1))).reshape((4,))
    new_hand_pos_r_yarp = mot.npvec_2_yarpvec(new_hand_pos_r_np[0:3])
    iCart.goToPoseSync(new_hand_pos_r_yarp, orient_hand)
    iCart.waitMotionDone(timeout=5.0)
    time.sleep(1)
    iCart.getPose(pos, orient)
    print('Hand position:', pos.toString())
    print('Hand orientation:', orient.toString())
    time.sleep(5)
    ######################################################################
    ################### Close network and motor cotrol ###################
    print('----- Close control devices and opened ports -----')
    driver_rarm.close()
    yarp.Network.fini()

if __name__ == "__main__":
    cartesian_ctrl_yarp()