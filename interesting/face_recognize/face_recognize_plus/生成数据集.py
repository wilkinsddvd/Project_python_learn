import cv2

detector = cv2.CascadeClassifier('D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
sampleNum = 0
Id = input('enter your id: ')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User." + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])  #

        cv2.imshow('frame', img)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum > 20:
        break

cap.release()
cv2.destroyAllWindows()