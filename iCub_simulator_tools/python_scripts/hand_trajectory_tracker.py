"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
hand trajectory tracker
"""

#############################run previously ##########################
#bash start_environment.sh
#bash start_cartesian_control_modules.sh left_arm right_arm

########################## Import modules  ###########################
######################################################################

import sys
import time

import numpy as np
import yarp
from hand_tracker.hand_trajectory import ford_and_backwards
from typing import List, Dict, Tuple, Any
from export_data.export_steady_coordinates_data import Writer
from tabulate import tabulate
from hand_tracker.joints_tracker.joints_trajectory_tracker import jointTrajectory

############ Import modules with specific functionalities ############
import Python_libraries.YARP_motor_control as mot
from Python_libraries.YARP_motor_control import motor_init, get_joint_position, motor_init_cartesian

################ Import parameter from parameter file ################
from examples.example_parameter import (Transfermat_robot2world, Transfermat_world2robot,
                                orientation_robot_hand, pos_hand_world_coord,
                                CLIENT_PREFIX, ROBOT_PREFIX)

######################################################################
######################### Init YARP network ##########################
######################################################################

yarp.Network.init()
if not yarp.Network.checkNetwork():
    sys.exit('[ERROR] Please try running yarp server')

class MoveHands:
    def __init__(self, output:str = "posi", newPosition: bool = False ):

        self.output: str = output
        #---------------------- init variables from parameter-file ---------------
        print('----------------- Init variables ------------------')

        yarp.Network.init()
        self._T_r2w: Transfermat_robot2world = Transfermat_robot2world
        self._T_w2r: Transfermat_world2robot = Transfermat_world2robot

        self._orient_hand: yarp = mot.npvec_2_yarpvec(orientation_robot_hand)

        self._pos: yarp = yarp.Vector(3)
        self._orient: yarp = yarp.Vector(4)

        print("-------------- Prepare a property object -----------")
        self.__iCart_r, __Driver_rarm = motor_init_cartesian("right_arm")
        self.__iCart_l, __Driver_larm = motor_init_cartesian("left_arm")

        # print("------------- Query motor control interfaces----------")
        # print("right_hand:", Driver_rarm, self.iCart_r)
        # print("left_hand:", Driver_larm, self.iCart_l)

        # ---------------------init encoder for get_join_position ---------
        print('---------------- Init arm joint position ----------------')

        self.__iCtrl_r, self.__iEnc_r, self.__jnts_r, __driver_r = motor_init("right_arm")
        self.__iCtrl_l, self.__iEnc_l, self.__jnts_l, __driver_l = motor_init("left_arm")
        #
        print("iEnc:",self.__iEnc_r, self.__jnts_r)
        print("iEnc_2:",self.__iEnc_r, self.__jnts_l)

        # ---------------------- track joints ---------------------------------

        self.__joints: jointTrajectory = jointTrajectory()

        ###########################################################################
        if (newPosition is False):

            self.__NewHandPosiOri: List = self._data_cleaning(self._ContinousHandMovement()[0])
            self.__lenght: int = len(self.__NewHandPosiOri)
            self.joints_trajectories: np.ndarray = self._ContinousHandMovement()[1]
        if (newPosition):
            self.__NewHandPosiOri: None = None

        print('--------------- Close control devices and opened ports ---------------')
        __Driver_rarm.close()
        __Driver_larm.close()

        yarp.Network.fini()

    def __repr__(self) -> Any:
        if not (self.__NewHandPosiOri):
            return "New Position"
        elif(self.__NewHandPosiOri) and (self.output == "posi"):
            both_concat: np.ndarray = self.__joints.concat_data(self.__NewHandPosiOri[0],self.__NewHandPosiOri[1])
            return tabulate(both_concat, headers="firstrow", tablefmt="github")
        elif (self.__NewHandPosiOri) and (self.output == "orient"):
            both_concat: np.ndarray = self.__joints.concat_data(self.__NewHandPosiOri[2], self.__NewHandPosiOri[3])
            return tabulate(both_concat, headers="firstrow", tablefmt="github")
        elif (self.__NewHandPosiOri) and (self.output == "joints"):
            return tabulate(self.joints_trajectories, headers="firstrow", tablefmt="github")

    def __len__(self) -> int:
        return self.__lenght

    def __setitem__(self, key:int, value: List) -> None:
        self.__NewHandPosiOri[key] = value

    def __getitem__(self, index: int) -> List:
        return self.__NewHandPosiOri[index]

    def _get_pos_ori_right(self) -> Dict:

        self.__iCart_r.getPose(self._pos, self._orient)
        print('Right hand position:', self._pos.toString())
        print('Right hand orientation:', self._orient.toString())
        rightPos, rightOri = self._pos.toString(), self._orient.toString()
        return {"RightPosi": rightPos, "RightOrient": rightOri}

    def _get_pos_ori_left(self) -> Dict:

        self.__iCart_l.getPose(self._pos, self._orient)
        print('Left hand position:', self._pos.toString())
        print('Left hand orientation:', self._orient.toString())
        return {"LeftPosi": self._pos.toString(), "LeftOrient": self._orient.toString()}

    def _InitHandPositionOrientation (self) -> List:

        """" 1.- Move hand to inital position and orientation """
        print('----- Move hand to initial pose -----')
        output = []

        welt_pos = pos_hand_world_coord
        init_hand_pos_np = np.dot(self._T_w2r, welt_pos.reshape((4, 1))).reshape((4,))
        init_hand_pos_yarp = mot.npvec_2_yarpvec(init_hand_pos_np[0:3])
        # ------------------------------------------------------------------------
        # ------------------Right Hand Position-----------------------------------
        self.__iCart_r.goToPoseSync(init_hand_pos_yarp, self._orient_hand)
        self.__iCart_r.waitMotionDone(timeout=5.0)
        time.sleep(0.5)
        print(self._get_pos_ori_right())
        output.append(self._get_pos_ori_right())

        # ----------- joint position----------------------
        self.__joints.return_data_right(get_joint_position(self.__iEnc_r, self.__jnts_r, as_np=True))

        # ------------------------------------------------------------------------
        # ----------------------- Left Hand Position ----------------------------#
        self.__iCart_l.goToPoseSync(init_hand_pos_yarp, self._orient_hand)
        self.__iCart_l.waitMotionDone(timeout=5.0)
        time.sleep(0.5)
        print(self._get_pos_ori_left())
        output.append(self._get_pos_ori_left())

        # ---------------- joint position ------------------------
        self.__joints.return_data_left(get_joint_position(self.__iEnc_l, self.__jnts_l, as_np=True))
        # ----------------------------------------------------
        return output

    def _ContinousHandMovement(self) -> Tuple[List, np.ndarray]: #_Position and Orientation
        """Move hand to new position and orientation """

        new_pos_ori: List[List,...] = self._InitHandPositionOrientation()

        print('----- Move hand to new pose -----')
        mov_range: list = ford_and_backwards(-2., 2., 5)
        print(new_pos_ori)

        for row in mov_range:
            welt_pos_n: np.ndarray = np.array([row[1], row[2], row[3], 1.])  # erster: links/rechts, zweiter: oben/unten, dritter: vorn/hinten
            new_hand_pos_r_np: np.dot = np.dot(self._T_w2r, welt_pos_n.reshape((4, 1))).reshape((4,))
            new_hand_pos_r_yarp: mot.npvec_2_yarpvec = mot.npvec_2_yarpvec(new_hand_pos_r_np[0:3])

            # ----------------------------------------------------------
            # ------------ Right Hand Position/Orientation -------------
            self.__iCart_r.goToPoseSync(new_hand_pos_r_yarp, self._orient_hand)
            self.__iCart_r.waitMotionDone(timeout=5.0)
            #time.sleep(0.5)
            print(self._get_pos_ori_right())

            # ------------ Append Outputs Position Orientation -----------
            new_pos_ori.append(self._get_pos_ori_right())
            time.sleep(0.5)

            # ------------------ joint_position -------------------
            right_joints: np.ndarray = get_joint_position(self.__iEnc_r, self.__jnts_r, as_np=True )
            self.__joints.return_data_right(right_joints)
            #print(self.joints.joints_right)

            # -------------------------------------------------------------
            # ------------- Left Hand Position/Orientation -----------------
            self.__iCart_l.goToPoseSync(new_hand_pos_r_yarp, self._orient_hand)
            self.__iCart_l.waitMotionDone(timeout=5.0)
            time.sleep(0.5)
            print(self._get_pos_ori_left())

            # -------------- Outputs Position Orientation -------------------
            new_pos_ori.append(self._get_pos_ori_left())
            time.sleep(0.5)

            # ------------------ joint_position -------------------
            left_joints: np.ndarray = get_joint_position(self.__iEnc_l, self.__jnts_l, as_np=True)
            self.__joints.return_data_left(left_joints)
            #print(self.joints.joints_left)

        joints_position: np.ndarray = self.__joints.concat_data(self.__joints.joints_right, self.__joints.joints_left, axis=1)#self._hand_init_pos_ori()[1]
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

    def MoveToNewPosition(self) -> List:

        list = self._InitHandPositionOrientation()

        welt_pos_n = np.array([-0.3, 0.4, 0.7, 1.])  # erster: links/rechts, zweiter: oben/unten, dritter: vorn/hinten
        new_hand_pos_r_np: np.dot = np.dot(self._T_w2r, welt_pos_n.reshape((4, 1))).reshape((4,))
        new_hand_pos_r_yarp: mot.npvec_2_yarpvec = mot.npvec_2_yarpvec(new_hand_pos_r_np[0:3])

        # -------------------------------------------------------------
        # ------------- Right Hand Position/Orientation -----------------
        self.__iCart_r.goToPoseSync(new_hand_pos_r_yarp, self._orient_hand)
        self.__iCart_r.waitMotionDone(timeout=5.0)
        self._get_pos_ori_right()

        # -------------------------------------------------------------
        # ------------- Left Hand Position/Orientation -----------------
        self.__iCart_l.goToPoseSync(new_hand_pos_r_yarp, self._orient_hand)
        self.__iCart_l.waitMotionDone(timeout=5.0)
        self._get_pos_ori_left()

        return list




if __name__ == "__main__":
    handMov: MoveHands = MoveHands(output="posi", newPosition=False)

    #position: handMov = handMov.MoveToNewPosition()
    #print(handMov)
    output: Writer = Writer(handMov, output="position")
    #1.- liefert eine Tabelle der ausgewählten Daten Position/Orientation aus dem __repr__ zurück
    print(output)
    #2.- liefert die Flugbahn der Gelenke zurück
    print(handMov.joints_trajectories)
    #3.- liefert die DataFrame von Position oder Orientation züruck
    print(output.data)





