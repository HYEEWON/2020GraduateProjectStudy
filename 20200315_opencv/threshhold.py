import cv2
# threshold: 특징을 검출할 떄, 문턱값 이하은 0, 이상는 255(흰색)로 바꿈
# 이진화

def img_load():
    img = cv2.imread('Image/iu.jpg', cv2.IMREAD_COLOR)
    gray = cv2.imread('Image/iu.jpg', cv2.IMREAD_GRAYSCALE)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray3 = cv2.threshold(gray2, 20, 255, cv2.THRESH_BINARY)
    #127: 원하는 픽셀 문턱값, 255: 150보다 크면 255

    cv2.imshow('IU', img)
    cv2.imshow('gray_iu', gray)
    cv2.imshow('gray_iu2', gray2)
    cv2.imshow('gray_iu3', gray3)
    k = cv2.waitKey(0)&0xFF

    if k == 27: #esc
        cv2.destroyAllWindows()

def showcam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, gray3 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

        cv2.imshow('frame', frame)
        cv2.imshow('gray2', gray2)
        cv2.imshow('gray3', gray3)
        if cv2.waitKey(1) > 0: break
    cap.release()
    cv2.destroyAllWindows()


img_load()
#showcam()