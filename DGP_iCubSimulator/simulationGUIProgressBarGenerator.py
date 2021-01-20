"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
Main Loop
"""


######################################################################
########################## Import modules  ###########################
######################################################################

import logging
import time
import numpy as np
import os
import sys
sys.path.insert(0, "~/PycharmProjects/projectPy3.8/iclub_simulator_tools/DGP_iCubSimulator/DGP_iCubSimulator")

from hand_trajectory_tracker import *
from numpy import ndarray
from typing import List, Dict, Tuple, TypeVar, Callable



from simulationEventModulator import SimulationEventModulator
from groundTruthGenerator import GroundTruthGenerator
from trajectory_generator import TrajectoryGenerator, __regulatingValuesEqualToLimit
from PyQt5.QtWidgets import (QApplication, QDialog, QProgressBar, QPushButton)
from export_data.export_static_coordinates_data import HandCoordinatesWriter, HeadCoordinatesWriter, SensorCoordinatesWriter, ObjectCoordinatesWriter, camerasCoordinatesWriter, GroundTruthWriter


logging.basicConfig(filename="simulationGUIGenerator.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


class SimulationGUIProgressBarGenerator(SimulationEventModulator, GroundTruthGenerator, QDialog):
    A: TypeVar = TypeVar("A", List, str)

    def __init__(self, object: str, armSide: str, newHandCoord: List[float], newHeadCoord: List[float],  trajCoord: List[ndarray], path: str):
        super(SimulationGUIProgressBarGenerator, self).__init__(object, armSide, newHandCoord, newHeadCoord)

        GroundTruthGenerator.__init__(self)
        QDialog.__init__(self)

        self.__X_trajectory, self.__Y_trajectory, self.__Z_trajectory = self.__trajectories(trajCoord)
        self.__path: str = path
        # ----------------------------- initial coordinates values --------------------------------
        self.__trackedCollisionHandRight: List[List[float]] = [self.startCollisionSensorDataFromOutside()[0][0]]
        self.__trackedCollisionHandLeft: List[List[float]] = [self.startCollisionSensorDataFromOutside()[0][1]]

        self.__trackedCollisionForearmRight: List[List[float]] = [self.startCollisionSensorDataFromOutside()[1][0]]
        self.__trackedCollisionForearmLeft: List[List[float]] = [self.startCollisionSensorDataFromOutside()[1][1]]

        self.__trackedCollisionArmRight: List[List[float]] = [self.startCollisionSensorDataFromOutside()[2][0]]
        self.__trackedCollisionArmLeft: List[List[float]] = [self.startCollisionSensorDataFromOutside()[2][1]]

        self.__objectCoordinates: List[List] = [self.startObjectCoordinatesFromOutside()]

        self.__jointCoordinates: List[List[float]] = [self.startJointsCoordinatesFromOutside()]
        self.__handCoordinates: List[Dict] = [self.startHandJointsCoordinatesFromOutside()]

        self.__visualPerceptionRight: List[ndarray] = [self.startVisualPerceptionCoordFromOutside()[0]]
        self.__visualPerceptionLeft: List[ndarray] = [self.startVisualPerceptionCoordFromOutside()[1]]
        self.__visualBinocularPerception: List[Tuple[ndarray, ndarray]] = [self.startVisualPerceptionCoordFromOutside()]
        self.__sceneRecords: List[ndarray] = [self.startSceneRecorderCoordsFromOutside()]

        self.__headEyesCoordinates: List[List] = [self.startHeadEyeCoordinatesFromOutside()]

        # ----------------------- Create the Ground Truth -------------------------------------

        #GroundTruth: GroundTruth = GroundTruth()
        self.__GroundTruthForRightHand: List[Tuple] = []
        self.__GroundTruthForLeftHand: List[Tuple] = []

        self.__GroundTruthForRightForearm: List[Tuple] = []
        self.__GroundTruthForLeftForearm: List[Tuple] = []

        self.__GroundTruthForRightArm: List[Tuple] = []
        self.__GroundTruthForLeftArm: List[Tuple] = []

        self.__initUI()

    def __initUI(self) -> None:
        """
        create a GUI which shows the loading process in a process bar
        """
        self.setWindowTitle('Progress Bar')
        self.__progress: QProgressBar = QProgressBar(self)
        self.__progress.setGeometry(0, 0, 300, 25)
        self.__progress.setMaximum(len(self.__X_trajectory))
        self.__button: QPushButton = QPushButton('Start', self)
        self.__button.move(0, 30)
        self.__button.clicked.connect(self.__run)
        self.show()

    def __trajectories(self,  trajectories: List[ndarray]) ->Tuple[ndarray, ndarray, ndarray]:
        return trajectories[0], trajectories[1], trajectories[2]

    def __run(self) -> None:
        """
        initialises the data collection instance variables, generates a trajectory and executes the movement the object according to it
        """

        # ---------------------- Parameters for the while loop ---------------------------------
        start: int = 9
        STEPSIZE: int = 10
        numPause: int = len(self.__X_trajectory)
        PAUSE: List[int] = [x for x in range(start, numPause, STEPSIZE)]
        print(PAUSE)
        i: int = 0;
        trialsCounter: int = 0;
        while True:

            try:
                logging.info(
                      f"index:{i},left hand:{len(self.__trackedCollisionHandLeft)}, left forearm:{len(self.__trackedCollisionForearmLeft)}, left arm:{len(self.__trackedCollisionArmLeft)},"
                      f" right hand:{len(self.__trackedCollisionHandRight)}, right forearm: {len(self.__trackedCollisionForearmRight)}, right arm: {len(self.__trackedCollisionArmRight)} ",
                      f" object coord: {len(self.__objectCoordinates)}, hand coord:{len(self.__handCoordinates)}, joints coord:{len(self.__jointCoordinates)}, head coord:{len(self.__headEyesCoordinates)}")

                # -------------------- move the object -----------------------------
                self.translationObjectCoordinatesFromOutside(self.__X_trajectory[i], self.__Y_trajectory[i], self.__Z_trajectory[i])
                # iCubSim_wrld_ctrl.move_object(self.__object, [X[i], Y[i], Z[i]])
                # print(iCubSim_wrld_ctrl.get_object_location(objID).tolist())

                # -------------------- Head, Hand and Eyes Coordinates ------------
                self.__handCoordinates.append(self.startHandJointsCoordinatesFromOutside())
                self.__jointCoordinates.append(self.startJointsCoordinatesFromOutside())
                self.__headEyesCoordinates.append(self.startHeadEyeCoordinatesFromOutside())

                # -------------------- append visual perception data ---------------
                self.__visualPerceptionRight.append(self.startVisualPerceptionCoordFromOutside(i)[0])
                time.sleep(0.5)
                self.__visualPerceptionLeft.append(self.startVisualPerceptionCoordFromOutside(i)[1])
                time.sleep(0.5)

                # ------------------------ append scene data ----------------------------
                self.__sceneRecords.append(self.startSceneRecorderCoordsFromOutside(i))

                # -------------------- append collision data of the arm skin -----------------------
                self.__trackedCollisionHandRight.append(self.startCollisionSensorDataFromOutside()[0][0])
                time.sleep(0.5)
                self.__trackedCollisionHandLeft.append(self.startCollisionSensorDataFromOutside()[0][1])
                time.sleep(0.5)

                self.__trackedCollisionForearmRight.append(self.startCollisionSensorDataFromOutside()[1][0])
                time.sleep(0.5)
                self.__trackedCollisionForearmLeft.append(self.startCollisionSensorDataFromOutside()[1][1])
                time.sleep(0.5)

                self.__trackedCollisionArmRight.append(self.startCollisionSensorDataFromOutside()[2][0])
                time.sleep(0.5)
                self.__trackedCollisionArmLeft.append(self.startCollisionSensorDataFromOutside()[2][1])
                time.sleep(0.5)

                # ---------------------- append location data of object ------------------------
                self.__objectCoordinates.append(self.startObjectCoordinatesFromOutside())
                time.sleep(0.5)

                if (i in PAUSE):
                    time.sleep(2);
                    # print(tracked_collision_hand_right)
                    self.__GroundTruthForRightHand.append(self.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, self.__trackedCollisionHandRight))
                    self.__GroundTruthForLeftHand.append(self.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, self.__trackedCollisionHandLeft))
                    self.__GroundTruthForRightForearm.append(self.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, self.__trackedCollisionForearmRight))
                    self.__GroundTruthForLeftForearm.append(self.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, self.__trackedCollisionForearmLeft))
                    self.__GroundTruthForRightArm.append(self.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, self.__trackedCollisionArmRight))
                    self.__GroundTruthForLeftArm.append(self.createFileGroundTruthForEachTrial(trialsCounter, i, STEPSIZE, self.__trackedCollisionArmLeft))

                    time.sleep(2)
                    trialsCounter += 1;
                else:
                    pass

            except Exception as e:
                print(e)
                break
            except KeyboardInterrupt:
                break

            i += 1
            self.__progress.setValue(i);
        self.finishProgrammConectionFromOutside();
        print("right hand:", len(self.__trackedCollisionHandRight))
        print("left hand:", len(self.__trackedCollisionHandLeft))
        print("right forearm:", len(self.__trackedCollisionForearmRight))
        print("left forearm", len(self.__trackedCollisionForearmLeft))
        print("right arm", len(self.__trackedCollisionArmRight))
        print("left arm", len(self.__trackedCollisionArmLeft))

        print("joint's coordinates", len(self.__jointCoordinates))
        print("hand coordinates", len(self.__handCoordinates))

        print("visual right perception:", len(self.__visualPerceptionRight))
        print("visual left perception:", len(self.__visualPerceptionLeft))

        print("scene records:", len(self.__sceneRecords))

        print("head eye coordinates:", len(self.__headEyesCoordinates))
        print("object coordinates:", len(self.__objectCoordinates))
        #print('JOINT COORDINATES', joint_coordinates)

        print("x axis", self.__X_trajectory)
        print("y axis", self.__Y_trajectory)
        print("z axis", self.__Z_trajectory)

        print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(self.__GroundTruthForLeftHand))
        print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(self.__GroundTruthForRightHand))
        print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(self.__GroundTruthForLeftForearm))
        print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(self.__GroundTruthForRightForearm))
        print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(self.__GroundTruthForLeftArm))
        print('Number of trials:', len(PAUSE), 'Number of Ground Truth:', len(self.__GroundTruthForRightArm))
        self.__saveData()

    def __saveData(self) -> None:
        """
        save the collected data from the different instance variables into a CSV file
        """
        __DATAOBJECT: ObjectCoordinatesWriter = ObjectCoordinatesWriter()
        __DATAOBJECT.outputDataToCSV(self.__objectCoordinates, filename=self.__path + '/' + 'object_data.csv')

        __DATAHAND: HandCoordinatesWriter = HandCoordinatesWriter()
        #print("HAND COORD",self.__handCoordinates)
        #print("JOINTS COORD", self.__jointCoordinates)
        __DATAHAND.HandCoordinatesToCSV(self.__handCoordinates, filename=self.__path + '/' + 'hand_coordinates.csv')
        __DATAHAND.JointsCoodinatesToCSV(self.__jointCoordinates, filename=path + '/' + 'joints_coordinates.csv')

        __DATAHEAD: HeadCoordinatesWriter = HeadCoordinatesWriter()
        __DATAHEAD.outputToCSV(self.__headEyesCoordinates, filename=self.__path + '/' + 'head_coordinates.csv')

        __DATASENSORS: SensorCoordinatesWriter = SensorCoordinatesWriter()
        __DATASENSORS.outputDataToCSV(self.__trackedCollisionArmRight, filename=self.__path + '/' + 'arm_right.csv')
        __DATASENSORS.outputDataToCSV(self.__trackedCollisionArmLeft, filename=self.__path + '/' + 'arm_left.csv')
        __DATASENSORS.outputDataToCSV(self.__trackedCollisionForearmRight, filename=self.__path + '/' + 'forearm_right.csv')
        __DATASENSORS.outputDataToCSV(self.__trackedCollisionForearmLeft, filename=self.__path + '/' + 'forearm_left.csv')
        __DATASENSORS.outputDataToCSV(self.__trackedCollisionHandRight, filename=self.__path + '/' + 'hand_right.csv')
        __DATASENSORS.outputDataToCSV(self.__trackedCollisionHandLeft, filename=self.__path + '/' + 'hand_left.csv')

        __DATACAMERAS: camerasCoordinatesWriter = camerasCoordinatesWriter()
        __DATACAMERAS.saveImgDataIntoDataSet(self.__visualBinocularPerception, datasetName='binocularPerception', filename=self.__path + '/' + 'binocular_perception.h5')

        __DATACAMERAS.saveImgDataIntoDataSet(self.__sceneRecords, datasetName='sceneRecords', filename=self.__path + '/' + 'scene_records.h5')

        __GROUNDTRUTH: GroundTruthWriter = GroundTruthWriter()

        __GROUNDTRUTH.saveGroundTruthtoCVS(self.__GroundTruthForRightHand, filename=self.__path + '/' + 'gt_right_hand.csv')
        __GROUNDTRUTH.saveGroundTruthtoCVS(self.__GroundTruthForLeftHand, filename=self.__path + '/' + 'gt_left_hand.csv')

        __GROUNDTRUTH.saveGroundTruthtoCVS(self.__GroundTruthForRightForearm, filename=self.__path + '/' + 'gt_right_forearm.csv')
        __GROUNDTRUTH.saveGroundTruthtoCVS(self.__GroundTruthForLeftForearm, filename=self.__path + '/' + 'gt_left_forearm.csv')

        __GROUNDTRUTH.saveGroundTruthtoCVS(self.__GroundTruthForRightArm, filename=self.__path + '/' + 'gt_right_arm.csv')
        __GROUNDTRUTH.saveGroundTruthtoCVS(self.__GroundTruthForLeftArm, filename=self.__path + '/' + 'gt_left_arm.csv')


def __initializeArmSide(ArmSide: bool) -> Tuple[float, float, str, List[float], List[float]]:
    """
    @:param Armside: a boolean value
    @:return: a tuple with the start, end value for the X Trajectory of the object. Two lists of floating numbers for the hand and head coordinates
    """
    start_X, end_X, armSide, NewHandCoord, NewHeadCoord = None, None, None, None, None # type: float, float, str, List[float], List[float]
    if (ArmSide):
        start_X = __regulatingValuesEqualToLimit(0.0) #StartSelector(ArmSide, False)  # handSide == True -> right hand
        end_X = 0.0
        armSide = "right_arm"
        NewHandCoord = [-0.111577, 0.27158, 0.501089, 0.1]
        NewHeadCoord = [-35., -10., -30., 0., 0., 10.]

    elif not (ArmSide):
        start_X = __regulatingValuesEqualToLimit(0.3) #StartSelector(ArmSide, False)  # handSide == False -> left Hand
        end_X = 0.3
        armSide = "left_arm"
        NewHandCoord = [0.111577, 0.27158, 0.501089, 0.1]
        NewHeadCoord = [-35., 10., 30., 0., 0., 10.]

    return start_X, end_X, armSide, NewHandCoord, NewHeadCoord

if __name__ == '__main__':

    # ----------------------- create the directory wherein the imgs will be saved ---------------------------
    path: str = os.path.dirname(os.path.abspath(__file__)) + "/trials"
    if not os.path.exists(path):
        os.mkdir(path)
        pathBinocular: str = os.path.dirname(os.path.abspath(__file__)) + "/trials/image_outputs"
        pathScene: os.path = os.path.dirname(os.path.abspath(__file__)) + "/trials/scene_outputs"

        if not os.path.exists(pathBinocular):
            os.mkdir(pathBinocular)


        elif not os.path.exists(pathScene):
            os.mkdir(pathScene)

    # parameters for the trajectory
    ArmSide: bool = True
    steps: int = 5  ######

    # ---------------------- Anzahl der Events: Objektbewegung, Augenbilder, Scenebilder ------------------------
    start_X, end_X, armSide, NewHandCoord, NewHeadCoord = __initializeArmSide(ArmSide)  # type: float, float, str, List[float], List[float]
    start_Y: float = __regulatingValuesEqualToLimit(0.7, False, True)  # StartSelector(False, True)
    start_Z: float = __regulatingValuesEqualToLimit(0.3, True, True)  # StartSelector(True, True)

    X: TrajectoryGenerator = TrajectoryGenerator(start_X, end_X, stepsize=0.05, dtyp=np.float_, num=steps, linspace=True)
    Y: TrajectoryGenerator = TrajectoryGenerator(start_Y, 0.8, stepsize=0.05, dtyp=np.float_, num=steps, linspace=True)
    Z: TrajectoryGenerator = TrajectoryGenerator(start_Z, 0.4, stepsize=0.05, dtyp=np.float_, num=steps, linspace=True)
    X_trajectory: ndarray = X.ConcatenateTrajectoryArrays()
    Y_trajectory: ndarray = Y.ConcatenateTrajectoryArrays()
    Z_trajectory: ndarray = Z.ConcatenateTrajectoryArrays()
    trajectories: List[ndarray] = [X_trajectory, Y_trajectory, Z_trajectory]

    # ------------------------------ instantiate the main class ----------------------------------
    app: QApplication = QApplication(sys.argv)
    runMainLoop: SimulationGUIProgressBarGenerator = SimulationGUIProgressBarGenerator(object='sbox', armSide=armSide, newHandCoord=NewHandCoord, newHeadCoord=NewHeadCoord,trajCoord=trajectories, path=path)
    sys.exit(app.exec_())
