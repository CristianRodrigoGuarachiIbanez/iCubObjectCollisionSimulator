from typing import List, Dict, Tuple, TypeVar, Any, Callable, Union
import csv
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
class Writer:

    A = TypeVar("A", np.ndarray, List)
    def __init__(self, raw_data:A, output: str = "both") -> None:
        self.output: str = output
        self.position, self.orientation = self._concatenate(raw_data)
        print(self.position)

        #self._updater(self.output + ".csv", self.position)
        self.data: pd.DataFrame = self._dataFrame(self.output)
        if not os.path.exists(self.output + ".csv"):
            try:
                self._dataframe_to_csv(self.output + ".csv", self.data, export="csv", mode="w")
            except Exception as e:
                print(e)
        else:
            try:
                self._dataframe_to_csv(self.output + ".csv", self.data, export="csv", mode="a")
            except Exception as e:
                print(e)

    def __repr__(self) -> tabulate:
        if (self.output == "position"):
            return tabulate(self.position, headers="firstrow", tablefmt="github")
        elif (self.output == "orientation"):
            return tabulate(self.orientation, headers="firstrow", tablefmt="github")
        elif (self.output == "both"):
            both_concat: np.ndarray = self._joint_pose_orient(self.position, self.orientation)
            return tabulate(both_concat, headers="firstrow", tablefmt="github")

    def _dataFrame(self, output="both") -> pd.DataFrame:

        if (output == "position"):
            return pd.DataFrame(self.position[1:,:], index=[x for x in range(1, len(self.position))], columns=self.position[0])
        elif(output == "orientation"):
            return pd.DataFrame(self.orientation[1:,:], index=[x for x in range(1, len(self.position))], columns=self.position[0])
        elif(output == "both"):
            both_concat: np.ndarray = self._joint_pose_orient(self.position, self.orientation)
            return pd.DataFrame(both_concat[1:,:], index=[x for x in range(1, len(both_concat))], columns=both_concat[0])


    @staticmethod
    def _dataframe_to_csv(filename:str, data: pd.DataFrame, export:str = None, mode:str ="w") -> None:
        if(export == "csv"  ):
            data.to_csv(filename, mode=mode, float_format='%.2f')
        elif(export == "xlsx"):
            data.to_excel(filename)

    def _concatenate(self, raw_data) -> tuple:
        position: np.ndarray = np.zeros((len(raw_data[0]) + len(raw_data[1])-1, len(raw_data[0][0])))
        orientation: np.ndarray = np.zeros((len(raw_data[2]) + len(raw_data[3])-1, len(raw_data[0][0])))
        if (len(raw_data) ==4):

            pose_right: np.ndarray = np.array(raw_data[0])
            pose_left: np.ndarray = np.array(raw_data[1])
            orient_right: np.ndarray = np.array(raw_data[2])
            orient_left: np.ndarray = np.array(raw_data[3])
            try:
                position = self._joint_pose_orient(pose_right, pose_left)
                orientation = self._joint_pose_orient(orient_right, orient_left)

            except Exception as e:
                print("Exception reached:", e)
        else:
            self.raw_data: List[List[float],...] = raw_data
            self.num_var: int = len(raw_data)

        return position, orientation

    @staticmethod
    def _joint_pose_orient(arr1:np.ndarray, arr2: np.ndarray) -> np.array:
        output:np.ndarray = np.empty((2, len(arr1)))

        if(len(arr1) == len(arr2)):
            output: np.ndarray = np.concatenate((arr1, arr2), axis=1)
        else:
            print("the lenght of the arrays is not similar")
        return output

    def _writer(self, filename, header: list, data:A, option) -> None:
        with open(filename, "w", newline="") as csvfile:
            if option == "write":
                lines: csv.writer = csv.writer(csvfile)
                lines.writerow(header)
                for x in data:
                    print(x)
                    lines.writerow(x)
            elif option == "update":
                writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                writer.writerows(data)
            else:
                print("Option is not known")

    def _updater(self, filename: str, new_data: np.ndarray) -> None:
        with open(filename, "r", newline="") as file:
            readData: List[Dict,...] = [row for row in csv.DictReader(file)]
            # print(readData)
            num_rows: int = len(new_data)

            for row in range(num_rows):
                for col in range(len(new_data[row])):
                    if(row != 0):
                        new_dict: Dict = {new_data[0][col]: new_data[row][col]}
                        print(new_dict)
                        readData.append(new_dict)
            # print(readData)
        header: List = readData[0].keys()
        self._writer(filename, header, readData, "update")
        #return readData




#if __name__ == "__main__":
    """lista:List = [[['right_pose_X', 'right_pose_Y', 'right_pose_Z'], [-0.227873, 0.147853, 0.202495], [-0.285013, -0.01859, 0.278243], [-0.289472, -0.017227, 0.275252], [-0.294914, -0.013953, 0.271476], [-0.301658, -0.009349, 0.266319], [-0.311748, -0.006329, 0.242138], [-0.32168, -0.001416, 0.211529], [-0.331333, 0.008486, 0.167857], [-0.335888, 0.024509, 0.116697], [-0.330917, 0.049971, 0.056212], [-0.311087, 0.085144, -0.006193], [-0.274959, 0.1252, -0.061042], [-0.229994, 0.161125, -0.099909], [-0.183353, 0.204284, -0.120928], [-0.141887, 0.236714, -0.129954], [-0.108228, 0.259444, -0.132269], [-0.080252, 0.27629, -0.131231], [-0.057658, 0.288751, -0.128568], [-0.03991, 0.297741, -0.125534], [-0.027625, 0.303987, -0.122421], [-0.025894, 0.30586, -0.120595], [-0.026742, 0.304115, -0.121483], [-0.038616, 0.297632, -0.12481], [-0.056852, 0.289019, -0.128573], [-0.079754, 0.276654, -0.131358], [-0.104997, 0.261423, -0.132349], [-0.14065, 0.237486, -0.130162], [-0.180623, 0.206518, -0.121785], [-0.228248, 0.162479, -0.101052], [-0.274465, 0.125678, -0.061585], [-0.31, 0.086529, -0.008444], [-0.330479, 0.051316, 0.053702], [-0.335898, 0.024952, 0.115443], [-0.331436, 0.008655, 0.167277], [-0.322398, -0.000956, 0.208722], [-0.312881, -0.005879, 0.239109], [-0.303033, -0.008908, 0.263583], [-0.296127, -0.013238, 0.270391], [-0.290556, -0.016622, 0.274514], [-0.284962, -0.019596, 0.278316]],
                  [['left_pose_X', 'left_pose_Y', 'left_pose_Z'], [-0.210773, 0.069752, 0.199114], [-0.258229, -0.336, 0.28318], [-0.26602, -0.332061, 0.276207], [-0.275623, -0.32566, 0.268235], [-0.288472, -0.315733, 0.25627], [-0.301636, -0.302883, 0.24335], [-0.249289, -0.338638, 0.219917], [-0.258308, -0.320023, 0.190452], [-0.342583, -0.22011, 0.110353], [-0.337918, -0.167393, 0.04834], [-0.313212, -0.122035, -0.010555], [-0.274442, -0.090645, -0.055832], [-0.232126, -0.066796, -0.08543], [-0.193335, -0.051315, -0.103135], [-0.161955, -0.042229, -0.113009], [-0.13262, -0.035988, -0.119309], [-0.110323, -0.032664, -0.122458], [-0.091745, -0.030992, -0.124172], [-0.076807, -0.029399, -0.124638], [-0.064176, -0.02837, -0.12494], [-0.054235, -0.028308, -0.125215], [-0.061166, -0.028721, -0.125429], [-0.076142, -0.030227, -0.125251], [-0.091243, -0.030959, -0.124216], [-0.109773, -0.032288, -0.122337], [-0.132189, -0.035905, -0.119394], [-0.159177, -0.041485, -0.11368], [-0.190771, -0.05055, -0.104122], [-0.231737, -0.066605, -0.085664], [-0.274041, -0.090226, -0.05616], [-0.312626, -0.12138, -0.011512], [-0.337663, -0.16719, 0.047598], [-0.343276, -0.218385, 0.110925], [-0.335214, -0.255199, 0.164702], [-0.319254, -0.283203, 0.211385], [-0.303811, -0.300182, 0.241756], [-0.289541, -0.314793, 0.255259], [-0.277287, -0.324426, 0.266848], [-0.268248, -0.330692, 0.274347], [-0.257896, -0.337081, 0.28219]],
                  [['right_orient_1', 'right_orient_2', 'right_orient_3', 'right_orient_4'], [-0.363861, -0.642207, 0.674667, 2.297906], [0.355201, -0.934775, 0.005268, 2.94552], [0.36101, -0.931368, 0.047171, 3.020706], [0.34702, -0.936679, 0.047015, 3.066631], [0.31415, -0.949372, 0.001603, 3.102766], [-0.292958, 0.956063, -0.010912, 3.090917], [-0.234072, 0.259522, -0.936941, 2.666718], [-0.190353, 0.568865, -0.800099, 2.687856], [-0.144534, 0.947837, 0.284104, 2.809155], [-0.079925, 0.293551, 0.952596, 3.115596], [-0.121136, 0.566067, 0.81541, 2.652715], [0.24289, 0.736481, -0.631348, 2.51664], [-0.274253, 0.141869, 0.951135, 2.551199], [-0.175687, 0.479054, 0.860024, 2.030225], [-0.191194, 0.453866, 0.870317, 1.918319], [-0.385113, 0.078168, 0.919553, 2.233526], [-0.499809, -0.174928, 0.848287, 2.609844], [0.595462, 0.672223, -0.439934, 2.373552], [-0.587431, -0.34837, 0.730454, 2.967321], [-0.308536, 0.276499, 0.910139, 1.722197], [-0.290634, 0.306207, 0.906515, 1.656686], [-0.283314, 0.316965, 0.905133, 1.670487], [-0.490989, -0.062398, 0.868928, 2.291312], [0.596675, 0.665971, -0.44773, 2.391145], [-0.296822, 0.280291, 0.912871, 1.883447], [0.54191, 0.543795, -0.640797, 2.786491], [0.494394, 0.701903, -0.512745, 2.440952], [0.438441, 0.709094, -0.552227, 2.474127], [-0.15629, 0.514272, 0.843266, 2.181382], [-0.246435, -0.061057, 0.967234, 2.975767], [0.10674, 0.768939, -0.630348, 2.515958], [-0.079679, 0.272495, 0.958852, 3.128205], [-0.143211, 0.952072, 0.270278, 2.802394], [-0.184731, 0.961126, 0.20521, 2.947371], [-0.310156, 0.755936, -0.576511, 2.721552], [-0.362735, 0.709099, -0.60465, 2.749854], [-0.357801, 0.914233, -0.190147, 3.04991], [-0.306562, -0.062171, -0.949818, 2.494282], [-0.327373, -0.071166, -0.942211, 2.469502], [-0.393278, 0.049102, -0.918108, 2.477821]],
                  [['left_orient_1', 'left_orient_2', 'left_orient_3', 'left_orient_4'], [-0.363861, -0.642207, 0.674667, 2.297906], [0.355201, -0.934775, 0.005268, 2.94552], [0.36101, -0.931368, 0.047171, 3.020706], [0.34702, -0.936679, 0.047015, 3.066631], [0.31415, -0.949372, 0.001603, 3.102766], [-0.292958, 0.956063, -0.010912, 3.090917], [-0.234072, 0.259522, -0.936941, 2.666718], [-0.190353, 0.568865, -0.800099, 2.687856], [-0.144534, 0.947837, 0.284104, 2.809155], [-0.079925, 0.293551, 0.952596, 3.115596], [-0.121136, 0.566067, 0.81541, 2.652715], [0.24289, 0.736481, -0.631348, 2.51664], [-0.274253, 0.141869, 0.951135, 2.551199], [-0.175687, 0.479054, 0.860024, 2.030225], [-0.191194, 0.453866, 0.870317, 1.918319], [-0.385113, 0.078168, 0.919553, 2.233526], [-0.499809, -0.174928, 0.848287, 2.609844], [0.595462, 0.672223, -0.439934, 2.373552], [-0.587431, -0.34837, 0.730454, 2.967321], [-0.308536, 0.276499, 0.910139, 1.722197], [-0.290634, 0.306207, 0.906515, 1.656686], [-0.283314, 0.316965, 0.905133, 1.670487], [-0.490989, -0.062398, 0.868928, 2.291312], [0.596675, 0.665971, -0.44773, 2.391145], [-0.296822, 0.280291, 0.912871, 1.883447], [0.54191, 0.543795, -0.640797, 2.786491], [0.494394, 0.701903, -0.512745, 2.440952], [0.438441, 0.709094, -0.552227, 2.474127], [-0.15629, 0.514272, 0.843266, 2.181382], [-0.246435, -0.061057, 0.967234, 2.975767], [0.10674, 0.768939, -0.630348, 2.515958], [-0.079679, 0.272495, 0.958852, 3.128205], [-0.143211, 0.952072, 0.270278, 2.802394], [-0.184731, 0.961126, 0.20521, 2.947371], [-0.310156, 0.755936, -0.576511, 2.721552], [-0.362735, 0.709099, -0.60465, 2.749854], [-0.357801, 0.914233, -0.190147, 3.04991], [-0.306562, -0.062171, -0.949818, 2.494282], [-0.327373, -0.071166, -0.942211, 2.469502], [-0.393278, 0.049102, -0.918108, 2.477821]]]
    #output: Writer = Writer(lista, output="position")
    #print(output.data)
    #print(output.dataFrame(output="both"))"""

