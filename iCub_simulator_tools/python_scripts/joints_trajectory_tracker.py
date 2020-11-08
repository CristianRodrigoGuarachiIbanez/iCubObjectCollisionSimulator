import numpy as np
from typing import Dict, Tuple, List, Any
import pandas as pd

class jointTrajectory:
    def __init__(self, num_cols: int = 17):
        self.joints_right: List[List,...] = [["joints_right_" + str(i) for i in range(1,num_cols)]]
        self.joints_left: List[List,...] = [["joints_left_" + str(i) for i in range(1,num_cols)]]
        self.joints_loc: pd.DataFrame = self._to_dataframe()

    def return_data_right(self, func: np.ndarray) -> List:
        self._save_data_right(func)
        return self.joints_right

    def _save_data_right(self, func: np.ndarray) -> None:
        self.joints_right.append(func)

    def return_data_left(self, func:np.ndarray) -> List:
        self._save_data_left(func)
        return self.joints_left

    def _save_data_left(self, func: np.ndarray) -> None:
        self.joints_left.append(func)

    def _to_dataframe(self):
        concat: np.ndarray = self.concat_data(self.joints_right, self.joints_left)
        return pd.DataFrame(concat[1:, :], index=[x for x in range(1, len(concat))],columns=concat[0])

    @staticmethod
    def concat_data(right: List, left: List) -> np.ndarray:

        if (len(right) == len(left)):

            arrRight: np.ndarray = np.array(right)
            arrLeft: np.ndarray = np.array(left)

            concat: np.ndarray = np.concatenate((arrRight, arrLeft), axis=1)
        return concat




#
# if __name__== "__main__":
#     joints: jointTrajectory= jointTrajectory(5)
#     arr: List= [[1,2,3,0], [3,4,5,6], [2,1,3,34], [3,2,4,5], [2,3,4,5], [98,76,32,3]]
#
#     for i in arr:
#         joints.return_data_left(i)
#         print(joints.joints_left)
#         joints.return_data_right(arr[arr.index(i)-1])
#         print(joints.joints_right)
#     print("x" *20)
#     print("right:")
#     print(joints.joints_right)
#     print(joints.joints_left)
#     print("concatenated:",joints.export_data(joints.joints_right, joints.joints_left))