"""
    Dependencies for Image Preprocessing of Images Containing Faces (or not)

        * adapt_bounding_box
        * histo_colour                               |
        * clahe_colour                               |   |
        * a comparison of the equalization methods  <|   |
        * a comparison of clahe parameters              <|
        * face_detect_dlib                                         |
        * face_detect_caffe                                        |
        * a test of the face detectors with local AffectNet data  <|
        * shape_to_np
        * facial landmarks dict
        * preprocess                     |
        * a test of the preprocessing   <|
        * preprocess_julien (DEPRECATED)
        * adapt_bounding_box_old (DEPRECATED)

"""


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import dlib
import cv2

from collections import OrderedDict


def adapt_bounding_box(x, y, w, h, image, rate, target_size=96):
    """
        Produces Quadratic Bounding Boxes within Image Boundaries

        Parameters:

            * x,y: coordinates of the upper right corner of the bounding box
            * w,h: width and height of the bounding box
            * image: the corresponding image
            * rate: factor for increasing the size of the bounding box
            * target_size: size of the image after preprocessing (default=96x96)

        Returns:

            * coordinates of the upper right corner and
              side length of the quadratic bounding box
    """

    # create a square that contains the original (rectangular) bounding box
    side_length = np.max([w, h])

    # possibly scaling up bounding box
    # face detectors differ in the degree of "tightness" of the face crop
    side_length_new = np.floor(side_length + rate*side_length)

    # limiting the minimal size of the bounding box to the final image size
    # scaling up face crops does not increase the information
    # small bounding boxes tend to be false positives or only parts of faces
    side_length_new = np.max([side_length_new, target_size])

    # bounding box size must not exceed image dimensions
    side_length_new = np.min([side_length_new, image.shape[0], image.shape[1]])

    # centering the bounding box
    y_new = np.floor(y-((side_length_new-h)/2))
    x_new = np.floor(x-((side_length_new-w)/2))

    # top left corner needs to be in the image
    if y_new < 0:
        y_new = 0
    if x_new < 0:
        x_new = 0

    # coordinates of the bottom right corner
    y_end = np.floor(y_new+side_length_new)
    x_end = np.floor(x_new+side_length_new)

    # bottom right corner needs to be in the image
    if(y_end > image.shape[0]):
        y_new = image.shape[0]-side_length_new
    if(x_end > image.shape[1]):
        x_new = image.shape[1]-side_length_new

    # produce correct output
    x_new = int(x_new)
    y_new = int(y_new)
    side_length_new = int(side_length_new)

    if(side_length_new <= 0):
        print("Error while computing bounding box")
        print(x, y, w, h, x_new, y_new)
        print(side_length_new)
        print(image.shape)

    return (x_new, y_new, side_length_new)


def histo_colour(img, c=True):
    """
        Histogram Equalization of RGB Images

        Parameter:

            * c: 'copy', if False or None img will be processed in place, otherwise a new object is returned

    """

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    if c:
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    else:
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)


def clahe_colour(img, c=True, tGS=7, cL=1):
    """
        Contrast Limited Adaptive Histogram Equalization of RGB Images (en.wikipedia.org/wiki/Adaptive_histogram_equalization)

        Parameter:

            c: 'copy', if False or None img will be processed in place, otherwise a new object is returned
            tGS: 'tileGridSize', size of the area where histogram equalization is applied
            cL: 'ClipLimit', pixel values exceeding this threshold in the histogram are equally redistributed, preventing noise amplification

    """

    clahe = cv2.createCLAHE(clipLimit=cL, tileGridSize=(tGS,tGS))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    channels[0] = clahe.apply(channels[0])
    cv2.merge(channels, ycrcb)
    if c:
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    else:
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)


# %% for testing
if False:
# %%

    import os

    # local images
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\output\ModelData\100')
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\output\ModelData\420')

    # comparison of clahe and histogram equalization
    # requires a folder with images and a subfolder named "equalized"
    for i, filename in enumerate(os.listdir()):

        if filename != "equalized" and filename != "cropped":

            i1 = cv2.imread(filename)

            e1 = cv2.cvtColor(cv2.cvtColor(cv2.resize(histo_colour(i1), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
            c1 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)



            i1 = cv2.resize(i1, (96,96), 0, 0)

            cv2.imwrite(os.path.normpath('equalized/'+str(i)+'.jpg'), np.hstack((e1, i1, c1)))

    # cdfs of the pixel intensity distributions of the original and the equalized images
    hist,bins = np.histogram(i1.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    hist,bins = np.histogram(e1.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized1 = cdf * hist.max() / cdf.max()

    hist,bins = np.histogram(c1.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized2 = cdf * hist.max() / cdf.max()

    cdf_normalized = cdf_normalized * (cdf_normalized2[255]/cdf_normalized[255])
    cdf_normalized1 = cdf_normalized1 * (cdf_normalized2[255]/cdf_normalized1[255])

    lin = np.linspace(0, cdf_normalized2[255], 256)

    plt.plot(np.abs(lin-cdf_normalized), color='g')
    plt.plot(np.abs(lin-cdf_normalized1), color='b')
    plt.plot(np.abs(lin-cdf_normalized2), color='r')
    plt.legend(('img', 'hist', 'clahe'), loc='upper left')
    plt.xlim([0,256])
    plt.show()

# %%

    # comparison of clahe parameters
    # requires a folder with images and a subfolder named equalized
    for i, filename in enumerate(os.listdir()):

        if filename != "equalized" and filename != "cropped":

            i1 = cv2.imread(filename)

            c1 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=0.5, tGS=4), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
            c2 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=0.5, tGS=6), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
            c3 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=0.5, tGS=7), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

            c4 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=1, tGS=4), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
            c5 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=1, tGS=6), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
            c6 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=1, tGS=7), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

            c7 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=1.5, tGS=4), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
            c8 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=1.5, tGS=6), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
            c9 = cv2.cvtColor(cv2.cvtColor(cv2.resize(clahe_colour(i1, cL=1.5, tGS=7), (96,96), 0, 0), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

            cv2.imwrite(os.path.normpath('equalized/'+str(i)+'.jpg'), np.hstack((np.vstack((c1, c2, c3)),np.vstack((c4, c5, c6)),np.vstack((c7, c8, c9)))))

# %%


def face_detect_dlib(img, detector, target_size=96):
    """
        Find Bounding Boxes for Faces with Dlib Frontal Face Detector

        Parameters:

            * detector: a Dlib face detector
            * target_size: size of the image after preprocessing (default=96x96)

        Returns:

            * whether a face was detected
            * the specification of the bounding box

    """

    # rescale big images to not slow down the face detection
    if img.shape[0] > 1000 and img.shape[0] == img.shape[1]:
        rescale = img.shape[0]/1000.
        img = cv2.resize(img, (1000,1000), 0, 0)
    else:
        rescale = 0

    # detect candidate faces
    face_coords = detector(img, 1)

    # find the best bounding box
    if len(face_coords) == 0:
        return False, (0,0,0), 0
    elif len(face_coords) == 1:
        coords = face_coords[0]
    elif len(face_coords) > 1:
        coords = face_coords[0]
        first_area = (coords.top() - coords.bottom())*(coords.left() - coords.right())
        # TODO: no update of 'first area'!?
        for c in face_coords[1:]:
            area = (c.top() - c.bottom())*(c.left() - c.right())
            if area > first_area:
                coords = c
                # DONE (but is this right?)
                first_area = (coords.top() - coords.bottom())*(coords.left() - coords.right())

    # adapt the bounding box
    x_new, y_new, sl = adapt_bounding_box(coords.left(), coords.top(), coords.right()-coords.left(), coords.bottom()-coords.top(), img, 0.10, target_size)
    if sl <= 0:
        return False, (0,0,0)
    sl_old = int(np.max([coords.right()-coords.left(), coords.bottom()-coords.top()]))

    # scale back
    if rescale:
        print("[INFO] Down-scaled an image")
        return True, (int(x_new*rescale), int(y_new*rescale), int(sl*rescale)), int(sl_old*rescale)
    else:
        return True, (x_new, y_new, sl), sl_old


def face_detect_caffe(img, detector, threshold, target_size=96):
    """
        Find Bounding Boxes for Faces with the Caffe Face Detector of CV3.3

        Parameters:

            * detector: a pretrained (caffe) DNN face detector
            * threshold: confidence cut-off for candidate faces
            * target_size: size of the image after preprocessing (default=96x96)

        Returns:

            * whether a face was detected
            * the specification of the bounding box
            * the confidence of the bounding box

    """

    # convert the image to a BLOB
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    # detect candidate faces
    detector.setInput(blob)
    detections = detector.forward()

    # find the best bounding box
    face_box = np.array([0,0,0,0])
    confidence_max = 0
    startX = startY = endX = endY = 0
    for i in range(0, detections.shape[2]):

        # confidence of the bounding box
        confidence = detections[0, 0, i, 2]
        if confidence < threshold:
            continue

        if confidence > confidence_max:

            # compute the (x, y)-coordinates of the bounding box
            face_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X1, Y1, X2, Y2) = face_box.astype('int')

            # at least 90% of each side of the BB inside of img
            if ((X2-X1)*0.9 > img.shape[1]-X1 or (Y2-Y1)*0.9 > img.shape[0]-Y1):
                continue

            confidence_max = confidence
            (startX, startY, endX, endY) = face_box.astype('int')

    # no adequate bounding box found
    if(confidence_max == 0):
        return False, (0,0,0), 0.0, 0

    # adapt the bounding box
    x_new, y_new, sl = adapt_bounding_box(startX, startY, endX-startX, endY-startY, img, 0.0, target_size)
    if sl <= 0:
        return False, (0,0,0), 0.0, 0
    sl_old = int(np.max([endX-startX, endY-startY]))

    return True, (x_new, y_new, sl), confidence_max, sl_old


# %% test the face detectors with histogram equalization
if False:
# %%

    import os
    from copy import copy

    # load face detectors
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\EmotiW\create_dataset')

    detector_dlib = dlib.get_frontal_face_detector()
    detector_landmarks = dlib.shape_predictor('dlibModels/shape_predictor_68_face_landmarks.dat')
    detector_caffe = cv2.dnn.readNetFromCaffe('caffeModels/deploy.prototxt.txt', 'caffeModels/res10_300x300_ssd_iter_140000.caffemodel')

    # local images
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\output\ModelData\100')
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\output\ModelData\420')

    # scribble
    i1 = cv2.imread('0.jpg')
    detected, bb, conf = face_detect_caffe(i1, detector_caffe, .20)
    detected, bb = face_detect_dlib(i1, detector_dlib)
    face_coords = detector_dlib(i1, 1)

    dlib.rectangle(bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[2])
    face_coords[0]

    crop1 = i1[bb[1]:bb[1]+bb[2], bb[0]:bb[0]+bb[2]]
    crop2 = i1[face_coords[0].top():face_coords[0].bottom(), face_coords[0].left():face_coords[0].right()]

    plt.imshow(i1)
    plt.imshow(crop1)
    plt.imshow(crop2)

    landmarks = detector_landmarks(i1, dlib.rectangle(bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[2]))
    landmarks1 = shape_to_np(landmarks)
    landmarks = detector_landmarks(i1, face_coords[0])
    landmarks2 = shape_to_np(landmarks)

    img1 = copy(i1)
    for (x, y) in landmarks1:
        cv2.circle(img1, (x, y), 1, (0, 0, 255), -1)

    img2 = copy(i1)
    for (x, y) in landmarks2:
        cv2.circle(img2, (x, y), 1, (0, 0, 255), -1)

    plt.imshow(i1)
    plt.imshow(img1)
    plt.imshow(img2)

    landmarks1-landmarks2

    # TODO: no more face crop applied within the face detection
    d = 0
    for i, filename in enumerate(os.listdir()):

        if i >= 1500:
            break

        if i < 0:
            continue

        if filename != "equalized" and filename != "cropped":

            i1 = cv2.imread(filename)
            c1 = claheEqulColor(i1)

            detected, di, bb, conf = face_detect_caffe(i1, (96,96,3), detector_caffe, .20)
            if not detected:
                detected, di, bb = face_detect_dlib(i1, (96,96,3), detector_dlib)
            cdi = claheEqulColor((di*255).astype('uint8'), tGS=3)

            detected, dc, bb, conf = face_detect_caffe(c1, (96,96,3), detector_caffe, .20)
            if not detected:
                detected, dc, bb = face_detect_dlib(c1, (96,96,3), detector_dlib)

            i1 = cv2.resize(i1, (96,96))

            cv2.imwrite(os.path.normpath('cropped/'+str(i)+'.jpg'), cv2.cvtColor(np.hstack((i1, dc*255, cdi)).astype('uint8'), cv2.COLOR_BGR2GRAY))

# %%


def shape_to_np(shape=None, dtype="int"):

    """
        Extract Coordinates from Dlib Facial Landmark Detector

        Parameters:

            * shape: shape object of dlib facial landmark detector
            * dtype: data type of the coordinates (default: int)

        Returns:

            * numpy array (68, 2) with facial landmark coordinates
    """

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # convert landmarks to a 2-tuple of (x, y)-coordinates
    if shape is not None:
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


# map indeces of facial landmarks to specific face regions
FACIAL_LANDMARKS = OrderedDict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])


def preprocess(image, detector_dlib=None, detector_caffe=None, detector_landmarks=None, target_size=(96, 96, 1), mode=0, dtype='float'):
    """
    Preprocess an Image for the CNN

    Parameters:

        * image: image as numpy array
        * detectorDlib and Caffe: the Dlib and Caffe face detectors
        * target_size: size of the cropped face to return (default: (96, 96, 1))
        * mode: determines which detector is used, 0: both, 1: Caffe, 2: Dlib

    Returns:

        * whether a face was detected
        * array (values 0.0 to 1.0) with detected grayscale face (nearly centered),
          or an empty matrix if an error occured (check read_image or cut_and_resize for reasons)
        * the bounding box specifications
        * the confidence of the prediction (0 if Dlib detected the face)
        * coordinates of the facial landmarks
    """

    detected = False

    # Caffe
    if mode==0 or mode==1:
        detected, bb, conf, sl = face_detect_caffe(image, detector_caffe, .20, target_size[0])
    if not detected:

        # CLAHE + Caffe
        if mode==0 or mode==1:
            detected, bb, conf, sl = face_detect_caffe(clahe_colour(image), detector_caffe, .20, target_size[0])
        if not detected:

            conf = 0
            landmarks = shape_to_np()

            # Dlib
            if mode==0 or mode==2:
                detected, bb, sl = face_detect_dlib(image, detector_dlib, target_size[0])
            if not detected:

                # Histogram Equalization + Dlib
                if mode==0 or mode==2:
                    detected, bb, sl = face_detect_dlib(histo_colour(image), detector_dlib, target_size[0])

    if not detected:

        return False, np.zeros(target_size), (0,0,0), conf, 0, landmarks

    # apply the bounding box and scale image
    cropped_face = image[bb[1]:bb[1]+bb[2], bb[0]:bb[0]+bb[2]]
    cropped_face = cv2.resize(cropped_face, target_size[:2], 0, 0)

    # detect the facial landmarks
    i = 0
    while True:

        if detector_landmarks is not None:
            landmarks = detector_landmarks(cropped_face, dlib.rectangle(0, 0, target_size[0], target_size[1]))
            landmarks = shape_to_np(landmarks)

        # chin landmarks out of frame?
        chin_y = int(np.max([landmarks[7][1],landmarks[8][1],landmarks[9][1]]))
        if chin_y > target_size[0]+1 and (bb[1]+bb[2]) < image.shape[0]:

            # resize bounding box
            bb = adapt_bounding_box(bb[0], bb[1], bb[2], bb[2]+np.round((chin_y-target_size[0])*(sl/target_size[0])), image, 0.0, target_size[0])

            # crop again
            cropped_face = image[bb[1]:bb[1]+bb[2], bb[0]:bb[0]+bb[2]]
            cropped_face = cv2.resize(cropped_face, target_size[:2], 0, 0)

            i = i+1

            if i > 10:
                break

        else:
            break

    # equalize and grayscale
    cropped_face = clahe_colour(cropped_face, tGS=3)
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

    # convert to float with range 0..1
    if dtype == 'float':
        cropped_face = cropped_face/255.

    # reshape
    cropped_face = np.reshape(cropped_face, target_size)

    return True, cropped_face, bb, conf, sl, landmarks


# %% test the preprocessing
if False:
# %%

    import os
    # load face detectors
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\EmotiW\create_dataset')

    detector_dlib = dlib.get_frontal_face_detector()
    detector_caffe = cv2.dnn.readNetFromCaffe('caffeModels/deploy.prototxt.txt', 'caffeModels/res10_300x300_ssd_iter_140000.caffemodel')
    detector_landmarks = dlib.shape_predictor('dlibModels/shape_predictor_68_face_landmarks.dat')

    # local images
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\data\AffectNet\preprocessing\100')
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\data\AffectNet\preprocessing\420')

    # scribble
    i1 = cv2.imread('0.jpg')
    detected, image, bb, conf, landmarks = preprocess(i1, detector_dlib, detector_caffe, detector_landmarks, mode=0)

    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    plt.imshow(i1)
    cv2.imshow('ImageWindow', image)
    cv2.waitKey()

    # test
    for i, filename in enumerate(os.listdir()):

        if filename != "equalized" and filename != "cropped":

            # print(i)

            img = cv2.imread(filename)
            detected, image, bb, conf, sl, lm = preprocess(img, detector_dlib, detector_caffe, detector_landmarks, target_size=(100,100,1), dtype='int')

            #for (x, y) in lm:
            #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            x,y = lm[61]

            coarse = cv2.resize(image, (12,12), 0, 0)

            cv2.rectangle(image, (x-14,y-14), (x+14,y+14), (0, 255, 0), 2)

            src = cv2.cvtColor(np.hstack([image[:,:,0], cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (100,100))]).astype('uint8'), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.normpath('cropped/'+str(int(conf*100))+'_'+str(sl)+'_'+str(i)+'.jpg'), src)
            cv2.imwrite(os.path.normpath('cropped/'+str(int(conf*100))+'_'+str(sl)+'_'+str(i)+'_coarse.jpg'), coarse)

            if i==20:
                break

    # find image filename
    for i, filename in enumerate(os.listdir()):

        if filename != "equalized" and filename != "cropped":

          image = cv2.imread(filename)
          if image is None:
            print(i)
          if i==10:
              break

    cv2.imshow('',image)
    cv2.waitKey()

    print(filename)

# %%


def preprocess_julien(image, target_size=(48, 48, 1), detector=None, debug=True):
    """
    Preprocess an Image for the CNN, DEPRECATED

    Parameters:

        * image: image as numpy array or as path
        * target_size: size of thecropped face to return (default: (96, 96, 1))
        * detector: the dlib face detector if multiple calls.

    Returns:

        * array with detected face (nearly centered and aligned), or None if
          an error occured (check read_image or cut_and_resize for reasons)
    """

    # Ratio to downsize the frames when detecting the faces
    RATIO = 2.0

    # Open the image
    if isinstance(image, str):  # a path is given, not the numpy array
        # read the colored image and transform to grayscale
        image = cv2.imread(image, 0)
    if not isinstance(image, (np.ndarray)):
        # image could not be loaded, abort processing
        if debug:
            print("Could not load the image as numpy array.")
        return False, np.zeros(target_size, dtype=np.uint8)
    shape = image.shape

    # Face detector
    if detector is None:
        detector = dlib.get_frontal_face_detector()

    if (shape[0] > 1000 and shape[1] > 1000):
        RATIO = 4.0
    else:
        RATIO = 2.0

    # Downsize the gray image
    gray_downsized = cv2.resize(image, (int(shape[1]/RATIO), int(shape[0]/RATIO)), 0, 0)

    # Detect the face
    face_coords = detector(gray_downsized, 1)

    if len(face_coords) == 0:
        if debug:
            print("Unable to extract a face.")
        return False, np.zeros(target_size, dtype=np.uint8)

    # Count the number of faces
    if len(face_coords) == 1:
        coords = face_coords[0]
    if len(face_coords) > 1:
        coords = face_coords[0]
        first_area = (coords.top() - coords.bottom())*(coords.left() - coords.right())
        for c in face_coords[1:]:
            area = (c.top() - c.bottom())*(c.left() - c.right())
            if area > first_area:
                coords = c

    # Compute the bounding box coordinates
    top = int(min(max(0, coords.top()), shape[0]/RATIO))
    bottom = int(min(max(0, coords.bottom()), shape[0]/RATIO))
    left = int(min(max(0, coords.left()), shape[1]/RATIO))
    right = int(min(max(0, coords.right()), shape[1]/RATIO))
    # print(top, bottom, left, right)

    # Extract the face region
    roi_face = image[int(RATIO*top):int(RATIO*bottom), int(RATIO*left):int(RATIO*right)]

    # Resize the cropped region
    cropped_face = cv2.resize(roi_face, target_size[:2], 0, 0)

    return True, cropped_face.reshape(target_size)/255.


def adapt_bounding_box_old(x, y, w, h, image, rate, target_size=96):
    """
        Produces Quadratic Bounding Boxes within Image Boundaries

        Parameters:

            * x,y: coordinates of the upper right corner of the bounding box
            * w,h: width and height of the bounding box
            * image: the corresponding image
            * rate: factor for increasing the size of the bounding box
            * target_size: size of the image after preprocessing (default=96x96)

        Returns:

            * coordinates of the upper right corner and
              side length of the quadratic bounding box
    """

    # create a square that contains the original (rectangular) bounding box
    side_length = np.max([w, h])

    # possibly scaling up bounding box
    # face detectors differ in the degree of "tightsness" of the face crop
    side_length_new = np.floor(side_length + rate*side_length)

    # limiting the minimal size of the bounding box to the final image size
    # scaling up face crops does not increase the information
    # small bounding boxes tend to be false positives or only parts of faces
    side_length_new = np.max([side_length_new, target_size])

    # centering the bounding box by moving its top left corner up and to the left
    y_new = np.floor(y-((side_length_new-h)/2))
    x_new = np.floor(x-((side_length_new-w)/2))

    # top left corner needs to be in the image
    if y_new < 0:
        y_new = 0
    if x_new < 0:
        x_new = 0

    # new coordinates of the bottom right corner
    y_end = np.floor(y_new+side_length_new)
    x_end = np.floor(x_new+side_length_new)

    # reduce side length if bounding box size exceeds image size
    side_length_new_y = side_length_new
    side_length_new_x = side_length_new

    if(y_end > image.shape[0]):
        side_length_new_y = np.floor(image.shape[0]-y_new)
    if(x_end > image.shape[1]):
        side_length_new_x = np.floor(image.shape[1]-x_new)

    side_length_new = np.min([side_length_new_y, side_length_new_x])

    # produce correct output
    x_new = int(x_new)
    y_new = int(y_new)
    side_length_new = int(side_length_new)

    if(side_length_new <= 0):
        print("Error while computing bounding box")
        print(x, y, w, h, x_new, y_new)
        print(side_length_new)
        print(image.shape)

    return (x_new, y_new, side_length_new)
