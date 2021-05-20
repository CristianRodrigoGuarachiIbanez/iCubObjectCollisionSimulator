from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

from imgPreprocessingDepends import *


def ExtractFace(frame, startX, startY, endX, endY, sideLength):
    target_size=(sideLength,sideLength)
    # grab the frame dimensions and convert it to a blob
    #(h, w) = frame.shape[:2]
    (x_new, y_new, sl) = AdaptBoundingBox((startX,startY,endX-startX,endY-startY),frame, 0.0)
    cropped_face = frame[y_new:y_new+sl, x_new:x_new+sl]
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
    cropped_face = cv2.equalizeHist(cropped_face)
    #plt.imshow(cropped_face, cmap='gray')
    #plt.show()
    try:
        cropped_face= cv2.resize(cropped_face, target_size, 0, 0)
    except Exception as e:
        print("error resizing the frame")
        return 0, np.zeros(target_size, dtype=np.uint8), (0,0,0), 0.0
    return cropped_face


def StartDemo(frame_rate=15):

    #confidence for face detection
    conf = 0.2

    # load face detectorsf
    os.chdir(r'D:\Documents\Akademia\Aktuell\MA_IGS\models\EmotiW\create_dataset')
    detector_caffe = cv2.dnn.readNetFromCaffe('caffeModels/deploy.prototxt.txt', 'caffeModels/res10_300x300_ssd_iter_140000.caffemodel')
    detector_landmarks = dlib.shape_predictor('dlibModels/shape_predictor_68_face_landmarks.dat')
    detector_dlib = dlib.get_frontal_face_detector()

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:

        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        detected, face, bb, conf, sl, lm = preprocess(frame, detector_dlib, detector_caffe, detector_landmarks, target_size=(100,100,1), dtype='int')

        cv2.rectangle(frame, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[2]), (0, 0, 255), 50)

        mask = np.zeros((100,100,1))+1
        for i, (x, y) in enumerate(lm):
            if i>16:
                cv2.circle(frame, (bb[0]+int(x*(bb[2]/100)), bb[1]+int(y*(bb[2]/100))), 1, (0, 0, 0), 2)


        #show output on screen
        #cv2.imshow("Face",face)
        cv2.imshow("Frame", frame)
        #cv2.imshow("Points", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
    vs.stop()

    del(vs)




if __name__== "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-fps", "--frames_per_second", type=float, default=15.0, help="frames per second for recording videos")

    args = vars(ap.parse_args())
    StartDemo(args["frames_per_second"])