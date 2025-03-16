import cv2 as cv

src=cv.imread('manji1.jpg')
cv.imshow('input image', src)
# cv2读取图片的通道是BGR（blue green red）
# PIL 读取图片的通道是RGB
gray_img=cv.cvtColor(src, code=cv.COLOR_BGR2GRAY)
cv.imshow('gray_image', gray_img)

# 保存图片
cv.imwrite('gray_manji.jpg', gray_img)

cv.waitKey(0)
cv.destroyWindow()