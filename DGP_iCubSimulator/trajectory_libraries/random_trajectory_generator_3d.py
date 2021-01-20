"""
Created on Fr Nov 6 2020
@author: Cristian Rodrigo Guarachi Ibanez
random trajectory generator 3d
"""

import numpy as np
from random import choice#
from typing import List, Dict, TypeVar, Any, TypeVar
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
# import matplotlib
# #matplotlib.use('TKAgg', force=True)
# path: np.ndarray = np.zeros((3, 4))
# print(path)
# for i in range(3):
#     x = np.array([6., 7., 8.])
#     path[:, i] = path[:, i - 1] + x
#     print(path)
# print("letzter Stand:", path)

def range_selector(beginn:float, end:float) -> float:
    range_num:np.linspace = np.linspace(beginn, end, endpoint=False, num=5, retstep=True)

    return choice(range_num[0])



def generator(steps: int, step:float) -> np.array:
    path: np.ndarray = np.zeros((3, steps))
    # print(path)
    for i in range(steps):
        x, y, z = np.random.rand(3)
        sgnX: float = (x - 0.5) / abs(x - 0.5)
        sgnY: float = (y - 0.5) / abs(y - 0.5)
        sgnZ: float = (z - 0.5) / abs(z - 0.5)
        a = np.array([step*sgnX, step*sgnY, step*sgnZ])
        path[:, i] = path[:, i - 1] + a
        # print(path)
    return path


def update(i) -> None:
    # particles: List[List[float]] = list()
    global points, trajectories
    for trajectory, point in zip(trajectories, points):
        trajectory.set_data(point[0:2, :i])
        trajectory.set_3d_properties(point[2, :i])

A = TypeVar("A", List, np.ndarray)
def rev(l: A):
    if len(l) == 0: return []
    return [l[-1]] + rev(l[:-1])

"""
#print("letzter Stand:", generator(3))

fig = plt.figure()
ax = p3.Axes3D(fig)

N: int = 100
steps: int = 100
step: int = 2
# data = np.array(list(gen(N))).T
# line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

points: np.ndarray = [generator(steps, step) for _ in range(3)]





points_reverse: np.ndarray = rev(points)
# points_reverse: np.ndarray=  [[1-j for j in point[::-1]] for point in points]
print(points)
print("Reverse", points_reverse)
points: np.ndarray = np.concatenate((points, points_reverse), axis=1)
print(points)
trajectories: np.array = [ax.plot(point[0, 0:3], point[1, 0:3], point[2, 0:3])[0] for point in points]

print("#####")
print(trajectories)
print('####')

# Setting the axes properties
ax.set_xlim3d([-100, 100])
ax.set_xlabel('X')
ax.set_ylim3d([-100, 100])
ax.set_ylabel('Y')
ax.set_zlim3d([-100, 100])
ax.set_zlabel('Z')

ani: animation.FuncAnimation = animation.FuncAnimation(fig, update, N, interval=10000 / N, blit=False)
ani.save('matplot003.gif', writer='imagemagick')
#plt.show()
"""

def corrections( steps:int, step: float) -> List:

    points: List = [generator(steps, step) for _ in range(1)]
    points_reverse: np.ndarray = rev(points)
    points: np.ndarray = np.concatenate((points, points_reverse), axis=1)
    print("POINTS:", points)
    print(points.shape) # (1, 6, 20) first array, arrows, columns

    trajectories: List = [(point[0, 0:steps], point[1, 0:steps], point[2, 0:steps]) for point in points]
    print(trajectories)
    print("LEN DER ERSTEN LIST[ARRAYS]:", len(trajectories[0]), len(trajectories[0][0]))
    #output: np.ndarray = np.zeros((steps, 3), dtype = np.float32)
    output:List = list()
    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            for z in range(len(trajectories[i][j])):
                print('index_i:', i, 'index_j:',j, 'index_z:', z)
                if (j > 0):
                    break
                elif (-0.3 < trajectories[i][j][z] < 0.3):
                    LIN_RE: float = trajectories[i][j][z]


                    if (0.45 < trajectories[i][j+1][z] < 0.7):
                        UP_DOWN = trajectories[i][j + 1][z]


                        if (0.1 < trajectories[i][j + 2][z] < 0.4):
                            FORW_BEH = trajectories[i][j + 2][z]
                            output.append([LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)


                        elif (trajectories[i][j + 2][z] <= 0.1) or (trajectories[i][j + 2][z] >= 0.4):
                            FORW_BEH = range_selector(0.15, 0.4)
                            output.append( [LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)


                    elif (trajectories[i][j + 1][z] <= 0.4) or (trajectories[i][j + 1][z] >= 0.7):
                        #trajectories[i][j + 1][z] = range_selector(0.45, 0.7)
                        UP_DOWN = range_selector(0.45, 0.7)

                        if (0.2 < trajectories[i][j + 2][z] < 0.5):
                            FORW_BEH = trajectories[i][j + 2][z]
                            output.append( [LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)


                        elif (trajectories[i][j + 2][z] <= 0.1) or (trajectories[i][j + 2][z] >= 0.4):
                            FORW_BEH = range_selector(0.15, 0.4)
                            output.append([LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)


                elif (trajectories[i][j][z] <= -0.3) or (trajectories[i][j][z] >= 0.3):
                    LIN_RE: float = range_selector(-0.35, 0.3)

                    if (0.45 < trajectories[i][j + 1][z] < 0.7):
                        UP_DOWN = trajectories[i][j + 1][z]

                        if (0.1 < trajectories[i][j + 2][z] < 0.45):
                            FORW_BEH = trajectories[i][j + 2][z]
                            output.append([LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)


                        elif (trajectories[i][j + 2][z] <= 0.1) or (trajectories[i][j + 2][z] >= 0.4):
                            FORW_BEH = range_selector(0.15, 0.4)
                            output.append([LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)



                    elif (trajectories[i][j + 1][z] <= 0.4) or (trajectories[i][j + 1][z] >= 0.7):
                        #print("erstes Elif 2")
                        trajectories[i][j + 1][z] = range_selector(0.45, 0.7)
                        UP_DOWN = trajectories[i][j + 1][z]

                        if (0.2 < trajectories[i][j + 2][z] < 0.5):
                            FORW_BEH = trajectories[i][j + 2][z]
                            output.append([LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)


                        elif (trajectories[i][j + 2][z] <= 0.1) or (trajectories[i][j + 2][z] >= 0.4):
                            FORW_BEH = range_selector(0.15, 0.4)
                            output.append([LIN_RE, UP_DOWN, FORW_BEH])
                            print(output)

                else:
                    print("erstes Else")
                    output.append([trajectories[i][j][z], trajectories[i][j + 1][z], trajectories[i][j + 2][z]])

    return output

def replaceArraysItems(arr: np.ndarray, minValue: float, maxValue: float)-> np.ndarray:

    foundIndex: np.ndarray = np.where((arr <= minValue) & (arr >= maxValue))

    for index in foundIndex:
        arr[index] = range_selector(minValue + 0.2, maxValue)

    return arr



def trajectoryCorrections(steps:int, step: float) -> np.ndarray:

    points: List = [generator(steps, step) for _ in range(1)]
    points_reverse: np.ndarray = rev(points)
    points: np.ndarray = np.concatenate((points, points_reverse), axis=1)
    print("POINTS:", points)
    print(points.shape) # (1, 6, 20) first array, arrows, columns

    trajectories: List = [(point[0, 0:steps], point[1, 0:steps], point[2, 0:steps]) for point in points]
    print(trajectories)
    print("LEN np.ndarray[ARRAYS]:", len(trajectories[0]), len(trajectories[0][1]))

    output: np.ndarray = np.zeros((steps, len(trajectories[0])), dtype = np.float32)
    LinkRechts: np.ndarray = replaceArraysItems(trajectories[0][0], -0.3, 0.3)
    UpDown: np.ndarray = replaceArraysItems(trajectories[0][1], 0.4, 0.7)
    ForeBack: np.ndarray = replaceArraysItems(trajectories[0][2], 0.1, 0.4)
    for i in range(len(LinkRechts)):
        output[i: i+1] = [LinkRechts[i], UpDown[i], ForeBack[i]]

    return output



if __name__== '__main__':
    from tabulate import tabulate
    corr: Any = corrections(20,0.1)
    print("LEN DER ZWEITEN LIST[LISTS]",len(corr))
    print(tabulate(corr, headers=["X_Coordinates", "Y_Coordinates", "Z_Coordinates"], showindex="always", tablefmt="github"))

    # corrNew: np.ndarray = trajectoryCorrections(20, 0.1)
    # print(corrNew)
    # print(len(corrNew))

