"""

    Generate the Training Data for EDRAM from the compiled AffectNet DB removing unnecessary information

    TODO: Y_loc needs to be figured out, as there are multiple ROIs in the face

"""

import os
import numpy as np
import h5py
import cv2
from fuel.converters.base import fill_hdf5_file
from argparse import ArgumentParser


def main(path, mode):

    # empty datasets
    train_features = []
    train_features_100 = []
    train_locations = []
    train_labels = []
    train_dims = []
    test_features = []
    test_features_100 = []
    test_locations = []
    test_labels = []
    test_dims = []

    # open h5 file as 'f'
    if (os.path.isfile(path)):
        print("Opening", path, "\n")
    else:
        print(path, "does not exist\n")
        exit()

    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        print(e)
        exit()

    # access the data
    X = f['X']
    X_100 = f['X_100']
    Y = f['Y_lab']
    Y_val = f['Y_val']
    Y_ars = f['Y_ars']
    Y_loc = f['Y_loc']
    data = f['Train']

    hist=np.zeros(7)

    # change format
    print("[INFO] Start processing data...")
    for i in range(0, X.shape[0]):

        # target output
        if mode == 'theano':
            # one-hot to emotion category
            j = 0
            while Y[i,j] == 0 and j<6:
                j += 1
            label = int(j)
        else:
            # one-hot
            label = Y[i,:]

        j = 0
        while Y[i,j] == 0 and j<6:
            j += 1
        hist[j] = hist[j]+1

        # select focal point (www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg)
        # for a first try, simply the mouth region (point 62)
        px, py = Y_loc[i,61,:]
        # zoom, skew, x, skew, zoom, y
        location = np.array((0.28, 0, (int(px) - 50.0) / 50.0, 0, 0.28, (int(py) - 50.0) / 50.0), ndmin=1, dtype=np.float32)

        # image and down-scaled (coarse) image
        if mode == 'theano':
            # change image dim from (100, 100, 1) to (1, 100, 100)
            image = np.array(X[i,:,:,0], ndmin=3, dtype=np.float32)
            image_100 = np.array(X_100[i,:,:,0], ndmin=3, dtype=np.float32)
            # image_coarse = np.array(cv2.resize(X[i,:,:,0], (12,12)), ndmin=3, dtype=np.uint8)
        else:
            # keep image dim
            image = np.array(X[i, ...], ndmin=2, dtype=np.float32)
            image_100 = np.array(X_100[i, ...], ndmin=2, dtype=np.float32)
            # image_coarse = np.array(cv2.resize(X[i, ...], (12,12)), ndmin=2, dtype=np.uint8)

        # valence and arousal
        dims = np.array((Y_val[i], Y_ars[i]), ndmin=1, dtype=np.float32)

        # append data row
        if data[i] == b'train':
            train_features.append(image)
            train_features_100.append(image_100)
            train_locations.append(location)
            train_labels.append(label)
            train_dims.append(dims)
        else:
            test_features.append(image)
            test_features_100.append(image_100)
            test_locations.append(location)
            test_labels.append(label)
            test_dims.append(dims)

        # feedback
        if (i+1)%1000==0:
            print("[INFO] Appended", i+1, "rows of data")

    print('\n',hist,'\n')

    # save data
    save_path = '/scratch/forch/EDRAM/datasets/AffectNet_train_data_'+mode+'.hdf5'

    h5file = h5py.File(save_path, mode='w')

    dfata = (
            ('train', 'features', np.array(train_features, dtype=np.float32)),
            ('test', 'features', np.array(test_features, dtype=np.float32)),
            ('train', 'features_100', np.array(train_features_100, dtype=np.float32)),
            ('test', 'features_100', np.array(test_features_100, dtype=np.float32)),
            ('train', 'locations', np.array(train_locations, dtype=np.float32)),
            ('test', 'locations', np.array(test_locations, dtype=np.float32)),
            ('train', 'labels', np.array(train_labels, dtype=np.uint8)),
            ('test', 'labels', np.array(test_labels, dtype=np.uint8)),
            ('train', 'dimensions', np.array(train_dims, dtype=np.float32)),
            ('test', 'dimensions', np.array(test_dims, dtype=np.float32)),
    )

    fill_hdf5_file(h5file, data)
    for i, label in enumerate(('batch', 'channel', 'height', 'width')):
        h5file['features'].dims[i].label = label
    for i, label in enumerate(('batch', 'channel', 'height', 'width')):
        h5file['features_100'].dims[i].label = label
    for i, label in enumerate(('batch', 'index')):
        h5file['locations'].dims[i].label = label
    for i, label in enumerate(('batch',)):
        h5file['labels'].dims[i].label = label
    for i, label in enumerate(('batch', 'val|ars')):
        h5file['dimensions'].dims[i].label = label

    h5file.flush()
    h5file.close()

    print("[INFO] Saved data to", save_path,"\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, dest="path",
                        default='/scratch/facs_data/AffectNet/AffectNet_DB.h5', help="Path to dataset file")
    parser.add_argument("--mode", type=str, dest="mode",
                        default='keras', help="Defines channel ordering of image data")
    args = parser.parse_args()
main(**vars(args))