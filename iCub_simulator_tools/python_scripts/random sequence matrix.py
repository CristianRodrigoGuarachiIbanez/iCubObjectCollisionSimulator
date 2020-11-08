import numpy as np
from random import choice

#sequence = [i for i in range(10)]
z_sequence: np.array= np.linspace(-10, 10, num =10)
print(z_sequence)
y_sequence: np.array= np.linspace(-10,10,num=10)
print(y_sequence)
x_sequence: np.array= np.linspace(-10,10, num=10)
print(x_sequence)
matrix: np.ndarray = np.zeros((5, 3))

# print(matrix)
#
for trial in range(len(matrix)):
    for achse in range(len(matrix[trial])):
        if (achse == 0):
            matrix[trial][achse] = choice(x_sequence)
        elif(achse==1):
            matrix[trial][achse] = choice(y_sequence)
        elif(achse==2):
            matrix[trial][achse] = choice(z_sequence)

# for trial in range(len(matrix)):
#     for achse in range(len(matrix[trial])):
#         if (achse == 0):
#             matrix[trial][achse] = x_sequence[trial]
#         elif(achse==1):
#             matrix[trial][achse] = y_sequence[trial]
#         elif(achse==2):
#             matrix[trial][achse] = z_sequence[trial]
print(matrix)
# def vonLinksnachRechts(matrix:np.array)-> None:
for row in range(len(matrix)):
    for col in range(len(matrix[row])):
        if (col == 2):
            print(1, 1, 1 * matrix[row][col])
        elif (col == 1):
            print(1, 1 * matrix[row][col], 1)
        elif (col == 0):
            print(1 * matrix[row][col], 1, 1)