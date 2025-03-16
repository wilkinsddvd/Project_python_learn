import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)#0是表示调用电脑自带的摄像头，1是表示调用外接摄像头
cap.set(cv.CAP_PROP_FPS, 30)

while True:
    # 读取一帧视频
    ret, frame = cap.read()
    if not ret:
        break

        # 将图像转换为灰度图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_detector = cv.CascadeClassifier(
        "D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")

    # 检测人脸
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 这里的face_cascade 变量未定义，未知原因

    # 在图像中标注人脸
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果
    cv.imshow('Real-time Face Tracking', frame)
    if cv.waitKey(1) == ord('q'):
        break


