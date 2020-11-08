#############################run previously ##########################
#bash start_environment.sh
#bash start_cartesian_control_modules.sh left_arm right_arm

########################## Import modules  ###########################
######################################################################

import sys
import time

import numpy as np
import yarp
from hand_trajectory import ford_and_backwards
from typing import List, Dict, Tuple
from export_arm_movement_data import Writer
from tabulate import tabulate
from joints_trajectory_tracker import jointTrajectory
############ Import modules with specific functionalities ############
import Python_libraries.YARP_motor_control as mot
from Python_libraries.YARP_motor_control import motor_init, get_joint_position, motor_init_cartesian

################ Import parameter from parameter file ################
from example_parameter import (Transfermat_robot2world, Transfermat_world2robot,
                                orientation_robot_hand, pos_hand_world_coord,
                                CLIENT_PREFIX, ROBOT_PREFIX)

######################################################################
######################### Init YARP network ##########################
######################################################################

yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')

class move_hands:
    def __init__(self, Output:str = "posi" ):
        self.output: str = Output
        ################# init variables from parameter-file #################
        print('----- Init variables -----')
        yarp.Network.init()
        self.T_r2w: Transfermat_robot2world = Transfermat_robot2world
        self.T_w2r: Transfermat_world2robot = Transfermat_world2robot

        self.orient_hand = mot.npvec_2_yarpvec(orientation_robot_hand)

        self.pos = yarp.Vector(3)
        self.orient = yarp.Vector(4)

        ##################### Prepare a property object ######################
        self.iCart_r, Driver_rarm = motor_init_cartesian("right_arm")
        self.iCart_l, Driver_larm = motor_init_cartesian("left_arm")
        ################### Query motor control interfaces ###################
        print("right_hand:", Driver_rarm, self.iCart_r)
        print("left_hand:", Driver_larm, self.iCart_l)
        ################### init encoder for get_join_position ###################
        print( '------ Init arm joint position______')

        self.iCtrl_r, self.iEnc_r, self.jnts_r, driver_r = motor_init("right_arm")
        #
        self.iCtrl_l, self.iEnc_l, self.jnts_l, driver_l = motor_init("left_arm")
        #
        print("iEnc:",self.iEnc_r, self.jnts_r)
        print("iEnc_2:",self.iEnc_r, self.jnts_l)

        #----------------- track joints ---------------------------------

        self.joints: jointTrajectory = jointTrajectory()

        ###########################################################################
        self.hand_pos_ori: List = self._data_cleaning(self._hand_movement_pos_ori()[0])
        self.lenght: int = len(self.hand_pos_ori)
        self.joints_trajectories: List = self._hand_movement_pos_ori()[1]
        print('----- Close control devices and opened ports -----')
        Driver_rarm.close()
        Driver_larm.close()
        yarp.Network.fini()

    def __repr__(self) -> tabulate:

        if(self.output == "posi"):
            both_concat: np.ndarray = self.joints.concat_data(self.hand_pos_ori[0],self.hand_pos_ori[1])
            return tabulate(both_concat, headers="firstrow", tablefmt="github")
        elif(self.output == "orient"):
            both_concat: np.ndarray = self.joints.concat_data(self.hand_pos_ori[2], self.hand_pos_ori[2])
            return tabulate(both_concat, headers="firstrow", tablefmt="github")
        elif(self.output == "joints"):
            return tabulate(self.joints_trajectories, headers="firstrow", tablefmt="github")

    def __len__(self) -> int:
        return self.lenght

    def __setitem__(self, key:int, value: List) -> None:
        self.hand_pos_ori[key] = value

    def __getitem__(self, index: int) -> List:
        return self.hand_pos_ori[index]


    def _get_pos_ori_right(self) -> Dict:

        self.iCart_r.getPose(self.pos, self.orient)
        print('Right hand position:', self.pos.toString())
        print('Right hand orientation:', self.orient.toString())
        rightPos, rightOri = self.pos.toString(), self.orient.toString()
        return {"RightPosi": rightPos, "RightOrient": rightOri}

    def _get_pos_ori_left(self) -> Dict:
        self.iCart_l.getPose(self.pos, self.orient)
        print('Left hand position:', self.pos.toString())
        print('Left hand orientation:', self.orient.toString())
        return {"LeftPosi": self.pos.toString(), "LeftOrient": self.orient.toString()}

    def _hand_init_pos_ori (self) -> List:

        """" Move hand to inital position and orientation """
        print('----- Move hand to initial pose -----')
        output = []

        welt_pos = pos_hand_world_coord
        init_hand_pos_np = np.dot(self.T_w2r, welt_pos.reshape((4, 1))).reshape((4,))
        init_hand_pos_yarp = mot.npvec_2_yarpvec(init_hand_pos_np[0:3])

        self.iCart_r.goToPoseSync(init_hand_pos_yarp, self.orient_hand)
        self.iCart_r.waitMotionDone(timeout=5.0)
        time.sleep(0.5)
        self._get_pos_ori_right()
        output.append(self._get_pos_ori_right())
        #----------- joint position----------------------
        self.joints.return_data_right(get_joint_position(self.iEnc_r, self.jnts_r, as_np=True))


        self.iCart_l.goToPoseSync(init_hand_pos_yarp, self.orient_hand)
        self.iCart_l.waitMotionDone(timeout=5.0)
        time.sleep(0.5)
        self._get_pos_ori_left()
        output.append(self._get_pos_ori_left())
        #-------- joint position ------------------------
        self.joints.return_data_left(get_joint_position(self.iEnc_l, self.jnts_l, as_np=True))

        return output

    def _hand_movement_pos_ori (self) -> Tuple[List, List]:
        """Move hand to new position and orientation """

        new_pos_ori: List[List,...] = self._hand_init_pos_ori()

        print('----- Move hand to new pose -----')
        mov_range: list = ford_and_backwards(-2., 2., 5)
        print(new_pos_ori)


        for row in mov_range:
            welt_pos_n: np.ndarray = np.array([row[1], row[2], row[3], 1.])  # erster: links/rechts, zweiter: oben/unten, dritter: vorn/hinten
            new_hand_pos_r_np: np.dot = np.dot(self.T_w2r, welt_pos_n.reshape((4, 1))).reshape((4,))
            new_hand_pos_r_yarp: mot.npvec_2_yarpvec = mot.npvec_2_yarpvec(new_hand_pos_r_np[0:3])

            #------------ Right Hand Position/Orientation -------------
            self.iCart_r.goToPoseSync(new_hand_pos_r_yarp, self.orient_hand)
            self.iCart_r.waitMotionDone(timeout=5.0)
            time.sleep(0.5)
            self._get_pos_ori_right()

            # ----- Outputs Position Orientation speichern
            new_pos_ori.append(self._get_pos_ori_right())
            time.sleep(1)

            #----joint_position
            right_joints: np.ndarray = get_joint_position(self.iEnc_r, self.jnts_r, as_np=True )
            self.joints.return_data_right(right_joints)
            #print(self.joints.joints_right)

            # ------------- Left Hand Position/Orientation -----------------
            self.iCart_l.goToPoseSync(new_hand_pos_r_yarp, self.orient_hand)
            self.iCart_l.waitMotionDone(timeout=5.0)
            time.sleep(0.5)
            self._get_pos_ori_left()

            # -------  Outputs Position Orientation speichern
            new_pos_ori.append(self._get_pos_ori_left())
            time.sleep(1)

            #------joint_position
            left_joints: np.ndarray = get_joint_position(self.iEnc_l, self.jnts_l, as_np=True)
            self.joints.return_data_left(left_joints)
            #print(self.joints.joints_left)

        joints_position: List[List, ...] = self.joints.concat_data(self.joints.joints_right, self.joints.joints_left)#self._hand_init_pos_ori()[1]
        #print("joints:", joints_position)
        return new_pos_ori, joints_position
    @staticmethod
    def _data_cleaning(func:List) -> List:
        output_left_pose: List[List[any], ...] = [["left_pose_X", "left_pose_Y", "left_pose_Z"]]
        output_left_orient: List[List[any], ...] = [["left_orient_1", "left_orient_2", "left_orient_3", "left_orient_4"]]
        output_right_pose: List[List[any], ...] = [["right_pose_X", "right_pose_Y", "right_pose_Z"]]
        output_right_orient: List[List[any], ...] = [["right_orient_1", "right_orient_2", "right_orient_3", "right_orient_4"]]

        for dicts in func:
            for key, value in dicts.items():
                if key == "LeftPosi":
                    coords_posleft: List = [float(coord) for coord in value.split("\t")]
                    output_left_pose.append(coords_posleft)
                elif key == "LeftOrient":
                    coords_orientleft: List = [float(coord) for coord in value.split("\t")]
                    output_left_orient.append(coords_orientleft)
                elif key == "RightPosi":
                    coords_poseright: List = [float(coord) for coord in value.split("\t")]
                    output_right_pose.append(coords_poseright)
                elif key == "RightOrient":
                    coords_orientright: List = [float(coord) for coord in value.split("\t")]
                    output_right_orient.append(coords_orientright)

        return [output_right_pose, output_left_pose, output_right_orient, output_left_orient]


if __name__ == "__main__":
    hand_mov: move_hands = move_hands(Output="posi")
    print(hand_mov)
    output: Writer = Writer(hand_mov, output="position")
    #1.- liefert eine Tabelle der ausgewählten Daten Position/Orientation aus dem __repr__ zurück
    print(output)
    #2.- liefert die Flugbahn der Gelenke zurück
    print(hand_mov.joints_trajectories)
    #3.- liefert die DataFrame von Position oder Orientation züruck
    print(output.data)



