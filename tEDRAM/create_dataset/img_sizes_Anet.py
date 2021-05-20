"""
    Shows Image Size Distribution of Training Data

"""

from __future__ import print_function
import os
import h5py
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt

def img_sizes(data_path=r'D:/Documents/Akademia/Aktuell/MA_IGS/data/AffectNet/'):

    hist_x = []
    hist_y = []

    for data in ['training', 'validation']:
        print("[INFO] opening "+data+" data...")
        with open(os.path.normpath(data_path + data + '.csv')) as csvfile:

            # read the .csv and skip header
            reader = csv.reader(csvfile, delimiter=',', quotechar="'")
            next(reader)

            # process the images
            for i, row in enumerate(reader):

                # emotion label belongs to one of the first 7 categories
                if (int(row[6]) < 7):

                    # read the image
                    image = cv2.imread(os.path.normpath(data_path + str(row[0])))
                    if image is not None:
                        hist_x.append(image.shape[0])
                        hist_y.append(image.shape[1])
                    else:
                        print("I'm working:", i)

    hist_x = np.array(hist_x)
    hist_y = np.array(hist_y)

    print("\n",np.min(hist_x))
    print(np.min(hist_y))
    print("\n",np.mean(hist_x))
    print(np.mean(hist_y))
    print("\n",np.median(hist_x))
    print(np.median(hist_y))
    print("\n",np.histogram(hist_x[hist_x<550]))
    print("\n",np.histogram(hist_y[hist_y<550]))


if __name__ == "__main__":


    # path to AffectNet image data
    data_path = "/scratch/facs_data/AffectNet/Manually_Annotated_Images/"

    # generate the training set
    img_sizes(data_path)
