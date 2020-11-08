import numpy as np
import cv2
from PIL import ImageGrab

fourcc: cv2.cv2 = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # you can use other codecs as well.
vid: cv2.VideoWriter = cv2.VideoWriter('record.avi', fourcc, 8, (1000, 900))
while (True):
    img = ImageGrab.grab(bbox=(100, 100, 1100, 1000))  # x, y, w, h
    img_np = np.array(img)
    # frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    vid.write(img_np)
    cv2.imshow("frame", img_np)
    key = cv2.waitKey(1)
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()

"""import numpy as np
import cv2
# for windows, mac users
# from PIL import ImageGrab
# for linux users
import pyscreenshot as ImageGrab

# four character code object for video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video writer object
out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1366, 768))

while True:
    # capture computer screen
    img = ImageGrab.grab()
    # convert image to numpy array
    img_np = np.array(img)
    # convert color space from BGR to RGB
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # show image on OpenCV frame
    cv2.imshow("Screen", frame)
    # write frame to video writer
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

out.release()
cv2.destroyAllWindows()"""