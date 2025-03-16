import cv2
d=cv2.QRCodeDetector()
val,_,_ = d.detectAndDecode(cv2.imread('D:\ target.jpg'))   # 绝对路径也可
print('text is:',val)