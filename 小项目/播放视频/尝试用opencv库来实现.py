import numpy as np
import matplotlib.pyplot as plt
import pylab
import imageio
import skimage.io
import numpy as np
import cv2
cap = cv2.VideoCapture('D:/素材.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('image', frame)
    k = cv2.waitKey(20)
    #q键退出
    if (k & 0xff == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()