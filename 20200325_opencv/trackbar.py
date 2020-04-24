import numpy as np
import cv2

def onChange(x):
    pass

def trackbar():
    fruit = cv2.imread("Image/fruit.jpg", cv2.IMREAD_COLOR)
    img = np.zeros((1024, 1024, 3), np.uint8)
    cv2.namedWindow("Bar")
    cv2.namedWindow("hsv")
    cv2.createTrackbar('B', "Bar", 0, 255, onChange)
    cv2.createTrackbar('G', "Bar", 0, 255, onChange)
    cv2.createTrackbar('R', "Bar", 0, 255, onChange)
    #(trackbar이름, 윈도우 이름, 시작, 끝, 콜백 함수)
    switch = '0:off\n1: on'
    cv2.createTrackbar(switch, "Bar", 0, 1, onChange)
    cv2.createTrackbar("gray", "Bar", 0, 255, onChange)

    cv2.createTrackbar("h_max", "hsv", 0, 179, onChange)
    cv2.setTrackbarPos('h_max', 'hsv', 179)
    cv2.createTrackbar("h_min", "hsv", 0, 255, onChange)
    cv2.createTrackbar("s_max", "hsv", 0, 255, onChange)
    cv2.createTrackbar("s_min", "hsv", 0, 255, onChange)
    cv2.createTrackbar("v_max", "hsv", 0, 255, onChange)
    cv2.createTrackbar("v_min", "hsv", 0, 255, onChange)

    while True:
        b = cv2.getTrackbarPos('B', "Bar")
        g = cv2.getTrackbarPos('G', "Bar")
        r = cv2.getTrackbarPos('R', "Bar")
        s = cv2.getTrackbarPos(switch, "Bar")
        #(트랙바 이름, 트랙바 생성 윈도우 이름름)
        gray = cv2.getTrackbarPos("gray", "Bar")

        gg = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(gg, gray, 255, cv2.THRESH_BINARY)

        H_max = cv2.getTrackbarPos("h_max", "hsv")
        H_min = cv2.getTrackbarPos("h_min", "hsv")
        S_max = cv2.getTrackbarPos("s_max", "hsv")
        S_min = cv2.getTrackbarPos("s_min", "hsv")
        V_max = cv2.getTrackbarPos("v_max", "hsv")
        V_min = cv2.getTrackbarPos("v_min", "hsv")

        lower = np.array([H_min, S_min, V_min])
        higher = np.array([H_max, S_max, V_max])
        hsv = cv2.cvtColor(fruit, cv2.COLOR_BGR2HSV)
        Gmask = cv2.inRange(hsv, lower, higher)
        G = cv2.bitwise_and(fruit, fruit, mask=Gmask)

        cv2.imshow("BGR", img)
        cv2.imshow("fruit", thr)
        cv2.imshow("hsvv", G)
        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


trackbar()