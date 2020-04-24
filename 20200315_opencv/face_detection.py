import cv2
import sys

img_file = "Image/iu.jpg"
cascade_file = "C:/Users/user/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
# 정면 얼굴 인식
cascade_file2 = "C:/Users/user/Anaconda3/Lib/site-packages/cv2/data/haarcascade_lefteye_2splits.xml"

img = cv2.imread(img_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(70,70))

cascade2 = cv2.CascadeClassifier(cascade_file2)
eye_list = cascade2.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(10,10))

if len(face_list) > 0:
    print(face_list)
    color = [(0, 0, 255), (0, 255, 0)] #빨강 초록
    for face in face_list:
        x, y, w, h = face
        cv2.rectangle(img, (x,y), (x+w,y+h), color[0], thickness=5)

    if len(eye_list) > 0:
        print(eye_list)
        for eye in eye_list:
            x, y, w, h = eye
            cv2.rectangle(img, (x, y), (x + w, y + h), color[1], thickness=3)
    cv2.imshow("face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("no face")

