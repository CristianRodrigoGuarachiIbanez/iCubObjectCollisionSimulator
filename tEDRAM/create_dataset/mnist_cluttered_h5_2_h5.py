"""

    Generate the MNIST Cluttered Dataset from Julien's Data for the EDRAM

    Digits in this dataset are contained in randomly placed 28x28 patches

"""

import os
import cv2
import h5py
import numpy as np
from argparse import ArgumentParser
from fuel.converters.base import fill_hdf5_file


def main(path, mode):

    # provide empty datasets
    train_features = []
    train_locations = []
    train_labels = []
    test_features = []
    test_locations = []
    test_labels = []

    # open source h5 file as 'f'
    if (os.path.isfile(path)):
        print("\n[INFO] Opening", path, "\n")
    else:
        print("[ERROR]", path, "does not exist\n")
        exit()

    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        print(e)
        exit()

    # access the data
    X = f["X"]
    Y = f["Y"]
    px = f["px"]
    py = f["py"]

    # change format
    print("[INFO] Start processing data for "+mode+"...\n")
    for i in range(70000):

        # centered location of the digit patch in the image
        location = np.array((0.28, 0, (int(px[i]) + 14.0 - 50.0) / 50.0, 0, 0.28, (int(py[i]) + 14.0 - 50.0) / 50.0), ndmin=1, dtype=np.float32)
        # image and down-scaled (coarse) image
        if mode == 'theano':
            # channel first
            image = np.array(X[i, ...], ndmin=3, dtype=np.uint8)
            image_coarse = np.array(cv2.resize(X[i, ...], (12,12)), ndmin=3, dtype=np.uint8)
        else:
            # chanel last
            image = np.array(X[i, ...], ndmin=2, dtype=np.uint8)
            image.shape = image.shape + (1,)
            image_coarse = np.array(cv2.resize(X[i, ...], (12,12)), ndmin=2, dtype=np.uint8)
            image_coarse.shape = image_coarse.shape + (1,)

        # target output
        if mode == 'theano':
            # one-hot to digit label
            j = 0
            while Y[i,j] == 0 and j<9:
                j += 1
            label = int(j)
        else:
            # one-hot
            label = Y[i,:]

        # first 60.000 examples are training data
        if int(i) < 60000:
            train_features.append(image)
            train_locations.append(location)
            train_labels.append(label)
        else:
            test_features.append(image)
            test_locations.append(location)
            test_labels.append(label)

        if (i+1) % 1000 == 0:
            print("[INFO] Appended", i+1, "rows of data")

    # save data
    if mode == 'theano':
        save_path = '/scratch/forch/EDRAM/datasets/mnist_cluttered_test.hdf5'
    elif mode == 'keras':
        save_path = '/scratch/forch/EDRAM/datasets/mnist_cluttered_keras.hdf5'
    else:
        save_path = '/scratch/forch/EDRAM/datasets/mnist_cluttered_'+mode+'.hdf5'

    h5file = h5py.File(save_path, mode='w')

    data = (
            ('train', 'features', np.array(train_features)),
            ('test', 'features', np.array(test_features)),
            ('train', 'locations', np.array(train_locations)),
            ('test', 'locations', np.array(test_locations)),
            ('train', 'labels', np.array(train_labels, dtype=np.uint8)),
            ('test', 'labels', np.array(test_labels, dtype=np.uint8)),
    )
    fill_hdf5_file(h5file, data)
    for i, label in enumerate(('batch', 'channel', 'height', 'width')):
        h5file['features'].dims[i].label = label
    for i, label in enumerate(('batch', 'index')):
        h5file['locations'].dims[i].label = label
    for i, label in enumerate(('batch',)):
        h5file['labels'].dims[i].label = label

    h5file.flush()
    h5file.close()

    print("\n[INFO] Saved data to", save_path,"\n")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, dest="path",
                        default='/scratch/facs_data/mnist_cluttered/mnist_cluttered.h5', help="Path to dataset file")
    parser.add_argument("--mode", type=str, dest="mode",
                        default='theano', help="Defines channel ordering of image data")
    args = parser.parse_args()

main(**vars(args))