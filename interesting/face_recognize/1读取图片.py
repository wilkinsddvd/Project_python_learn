import cv2 as cv

img=cv.imread('manji1.jpg')  # 路径中不能有中文，否则加载图片失败

# 显示图片
cv.imshow('read_img', img)

# 等待键盘输入 单位毫秒 传入0 则就是无限等待
cv.waitKey(3000)

# 释放内存  由于OpenCV底层是C++写的
cv.destroyWindow()