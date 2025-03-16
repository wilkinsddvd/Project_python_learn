import cv2 as cv

def face_detect_demo():
    # 将图片转换为灰度图片
    grav=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 加载特征数值
    face_detector=cv.CascadeClassifier('D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
    faces=face_detector.detectMultiScale(grav)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        cv.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=2)
    cv.imshow('result', img)
# 加载图片
img=cv.imread('heying3.jpg')
face_detect_demo()


cv.imshow('img', img)



cv.waitKey(0)
cv.destroyAllWindow()
