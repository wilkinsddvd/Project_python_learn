import cv2 as cv

# 运动点坐标列表
point_array = []
# 运动点最大存储量
point_max = 15


def face_detect(image):
    global point_array
    global point_max

    # 灰度转换
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 加载库特征
    face_detector = cv.CascadeClassifier(
        "D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")

    # 调用内置检测函数
    faces = face_detector.detectMultiScale(gray_image)

    # 人脸检测
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        cv.putText(image, "face", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 添加新的运动点
        point_array.append([int((x + x + w) / 2), int((y + y + h) / 2)])
        # 存储运动点达到设定上限
        if (len(faces) * point_max) < len(point_array):
            # 删除前len(faces)个元素
            point_array = point_array[len(faces)::]

    # 运动追踪
    for x, y in point_array:
        cv.circle(image, (x, y), 2, (255, 0, 0), 3)


# 打开摄像头
cap = cv.VideoCapture(0)
# 设置宽度和高度
cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    # 每次读取一帧摄像头或者视频
    flag, image = cap.read()

    # 检测摄像头识别到的人脸
    face_detect(image)

    # 输出显示
    cv.imshow('result', image)

    # ESC关闭，ESC键值为27
    if (cv.waitKey(1) & 0xff) == 27:
        break

# 释放资源
cap.release()
cv.destroyAllWindows()