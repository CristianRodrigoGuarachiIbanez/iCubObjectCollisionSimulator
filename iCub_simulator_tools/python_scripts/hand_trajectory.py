import numpy as np





def ford_and_backwards(start:float, end:float, steps:int)-> list:
    X_coord: np.linspace = np.linspace(start, end, steps)  # l/r 1. bis -1.
    Y_coord: np.linspace = np.linspace(start, end, steps)  # o/u  0.7 bis -0.7
    Z_coord: np.linspace = np.linspace(abs(start)+start, end, steps)  # v/h 0.3 bis 0.8
    output:list = []
    length:int= len(X_coord)-1
    while length > 0:
        print(length, X_coord[length], Y_coord[length], Z_coord[length])
        output.append([int(length), float(X_coord[length]),float(Y_coord[length]), float(Z_coord[length])])
        length -=1
        if(length == 0):
            #print(len, X_coord[len])
            while length <=steps-1:
                print(length, X_coord[length],Y_coord[length], Z_coord[length])
                output.append([int(length), float(X_coord[length]),float(Y_coord[length]), float(Z_coord[length])])
                length+=1
            length= 0
    return output

#if __name__=="__main__":


    # trajectory: list= ford_and_backwards(-2.,2.,5)
    # #print(trajectory)
    # for row in trajectory:
    #     print("row", row[0], [row[1], row[2], row[3]])
