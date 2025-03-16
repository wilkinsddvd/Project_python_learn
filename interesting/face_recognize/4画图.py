import cv2 as cv
img=cv.imread('manji1.jpg')

# 画矩形 左上角的坐标是(x,y) 矩形的宽度和高度是(w,h)
# x, y, w, h=100, 100 ,100, 100
x, y, r=200, 200, 100
# cv.rectangle(img, (x, y, x+w, y+h), color=(0, 255, 0), thickness=2)  # color=BGR
# cv.imshow('rectangle_img', img)
# cv.waitKey(0)
# cv.destroyWindow()

cv.circle(img, center=(x, y), radius=r, color=(0, 0, 255), thickness=2)
cv.imshow('result image', img)
cv.waitKey(0)
cv.destroyAllWindow()