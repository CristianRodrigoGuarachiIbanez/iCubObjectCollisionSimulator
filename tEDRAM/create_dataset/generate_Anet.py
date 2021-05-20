"""
    Compiles a monolithic AffectNet Database including bounding box and other auxilliary information

      * generate_h5
      * main
"""

from __future__ import print_function
import os
import h5py
import numpy as np
import csv

from imgPreprocessingDepends import *


def generate_h5(filename='AffectNet_training_data_new.h5', data_path=r'D:/Documents/Akademia/Aktuell/MA_IGS/data/AffectNet/', overwrite=False, input_shape=(100,100,1)):
    """
        Generates a hdf5 file with the AffectNet data in the correct format.

        Parameters:

          * filename: name of the hdf5 target file
          * data_path: path to the AffectNet image data
          * overwrite: whether an existing hdf5 file will be overwritten
          * input_shape: target shape of the images for the neural network (e.g., (100, 100, 1))

    """

    # emotion expressions 0:10 are neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, non-face
    # we leave out contempt, none, uncertain, non-face, 7 labels remain
    n_cols = 7

    # HDF5 data file, write (non-overwrite)
    if overwrite:
        file = h5py.File(data_path+filename, "w")
    else:
        file = h5py.File(data_path+filename, "w-")

    # Create Datasets
    # input image for the CNN
    X = file.create_dataset("X",
                                  (0,) + input_shape,
                                  maxshape=(None,) + input_shape,
                                  dtype='i')
    # small input image for the CNN
    X_100 = file.create_dataset("X_100",
                                  (0, 100, 100, 1),
                                  maxshape=(None,) + input_shape,
                                  dtype='i')
    # emotion label as one-hot vector
    Y_lab = file.create_dataset("Y_lab",
                                      (0, n_cols),
                                      maxshape=(None, n_cols),
                                      dtype='i')
    # target value emotional valence
    Y_val = file.create_dataset("Y_val",
                                      (0,),
                                      maxshape=(None,),
                                      dtype='f')
    # target value arousal
    Y_ars = file.create_dataset("Y_ars",
                                      (0,),
                                      maxshape=(None,),
                                      dtype='f')
    # landmarks for localization
    Y_loc = file.create_dataset("Y_loc",
                                      (0, 68, 2),
                                      maxshape=(None, 68, 2),
                                      dtype='i')
    # training or validation data ("train" or "valid")
    Train = file.create_dataset("Train",
                                      (0,),
                                      maxshape=(None,),
                                      dtype='S5')
    # name of the image file
    Image = file.create_dataset("Images",
                                      (0,),
                                      maxshape=(None,),
                                      dtype='S66')
    # face detector for the image (Caffe = 1 or Dlib = 2)
    Detector = file.create_dataset("Detector",
                                         (0,),
                                         maxshape=(None,),
                                         dtype='i')
    # bounding box information
    BBox = file.create_dataset("BoundingBox",
                                     (0, 5),
                                     maxshape=(None, 5),
                                     dtype='i')

    # simple statisitcs for feedback
    n_processed = 0
    n_images = 0
    n_dlib = 0

    print("[INFO] loading face detectors...")
    # load face detectors
    detector_caffe = cv2.dnn.readNetFromCaffe('caffeModels/deploy.prototxt.txt', 'caffeModels/res10_300x300_ssd_iter_140000.caffemodel')
    detector_landmarks = dlib.shape_predictor('dlibModels/shape_predictor_68_face_landmarks.dat')
    detector_dlib = dlib.get_frontal_face_detector()

    # open .csv file containing filepaths and image information
    for data in ['training', 'validation']:
        print("[INFO] opening "+data+" data...")
        with open(os.path.normpath(data_path + data + '.csv')) as csvfile:

            # read the .csv and skip header
            reader = csv.reader(csvfile, delimiter=',', quotechar="'")
            next(reader)

            # process the images
            for i, row in enumerate(reader):

                # emotion label belongs to one of the first 7 categories
                if (int(row[6]) < n_cols):

                    # read the image
                    image = cv2.imread(os.path.normpath(data_path + str(row[0])))

                    # some images are missing
                    if isinstance(image, (type(None))):
                        print("[INFO] Could not open file",str(row[0]))
                        continue

                    if image.shape[0]+image.shape[1] >= 300:

                        n_processed += 1

                        # detect the face in the image
                        try:
                            detected, face, bb, conf, sl, lm = preprocess(image, detector_dlib, detector_caffe, detector_landmarks, input_shape, dtype='int')
                        except Exception as error:
                            print(error)

                        # face detected
                        if detected:

                            # new data row in hdf5 file
                            X.resize(X.shape[0]+1, axis=0)
                            X_100.resize(X_100.shape[0]+1, axis=0)
                            Y_lab.resize(Y_lab.shape[0]+1, axis=0)
                            Y_val.resize(Y_val.shape[0]+1, axis=0)
                            Y_ars.resize(Y_ars.shape[0]+1, axis=0)
                            Y_loc.resize(Y_ars.shape[0]+1, axis=0)
                            Train.resize(Train.shape[0]+1, axis=0)
                            Image.resize(Image.shape[0]+1, axis=0)
                            Detector.resize(Detector.shape[0]+1, axis=0)
                            BBox.resize(BBox.shape[0]+1, axis=0)

                            # store data
                            X[n_images, ...] = face
                            X_100[n_images, ...] = cv2.resize(face, (100,100))[None,:,:,None]
                            Y_lab[n_images, :] = np.zeros(n_cols)
                            Y_lab[n_images, int(row[6])] = 1
                            Y_val[n_images] = float(row[7])
                            Y_ars[n_images] = float(row[8])
                            Y_loc[n_images, ...] = lm
                            if data == 'training':
                                Train[n_images] = ('train').encode("ascii", "ignore")
                            else:
                                Train[n_images] = ('valid').encode("ascii", "ignore")
                            Image[n_images] = str(row[0]).encode("ascii", "ignore")
                            if conf:
                                Detector[n_images] = 1
                            else:
                                Detector[n_images] = 2
                            BBox[n_images] = bb + (conf,) + (sl,)


                            n_images += 1

                            # feedback
                            if (conf==0):
                                n_dlib += 1
                            if n_images % 5000 == 0:
                                print("[INFO] Processed", n_processed, "images, saved", n_images, "images ("+ str(n_dlib) +" from dlib)")
                                # save some examples
                                cv2.imwrite(os.path.normpath(data_path+'examples/'+str(i)+'.png'), np.clip(image,0,255))
                                cv2.imwrite(os.path.normpath(data_path+'examples/'+str(i)+'_bb.png'), np.clip(face,0,255))


    file.close()
    print("[INFO] Training data successfully generated!")


if __name__ == "__main__":

    # size of the input images
    input_shape = (100, 100, 1)

    # path to AffectNet image data
    data_path = "/scratch/facs_data/AffectNet/Manually_Annotated_Images/"

    # name of HDF5 file to generate
    output_file = "AffectNet_DB.h5"

    # generate the training set
    generate_h5(output_file, data_path, True, input_shape)
