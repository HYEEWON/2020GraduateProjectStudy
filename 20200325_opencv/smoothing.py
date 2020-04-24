#영상의 노이즈를 줄이기 위해 사용
#흐릿하게해서 선명도 저하
# LPF(Low Pass Filter): 노이즈 제거, 이미지 흐릿하게 하기
# 노이즈나 모서리 등 고주파 부분 제거, edge 무뎌짐
# HPF(High Pass Filter): 이미지의 윤곽선을 찾음

import cv2
import numpy as np

def conv(): # 2D Convolution (Image Filtering)
    img = cv2.imread("Image/cat.png", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (400, 220))
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(img, -1, kernel) #픽셀들의 평균값
    cv2.imshow("original", img)
    cv2.imshow("blur_kernel", blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def onMouse(x):
    pass

def bluring():
    img = cv2.imread("Image/median.png", cv2.IMREAD_COLOR)
    #img = cv2.resize(img, (500, 250))
    cv2.namedWindow("blur_bar")
    cv2.createTrackbar("blur_mode", "blur_bar", 0, 3, onMouse)
    cv2.createTrackbar("kernel_size", "blur_bar", 0, 5, onMouse)
    cv2.setTrackbarPos("blur_mode", "blur_bar", 0)
    mode = cv2.getTrackbarPos("blur_mode", "blur_bar")
    val = cv2.getTrackbarPos("kernel_size", "blur_bar")
    while True:
        val = val*2 + 1
        try:
            if mode == 0: # averagin blur
                blur = cv2.blur(img, (val, val))
            elif mode == 1: # gausian filter: 가우스함수를 필터로 사용
                # 백색노이즈 제거에 좋음
                blur = cv2.GaussianBlur(img, (val, val), 0)
                # edge가 보존되지 않고 뭉개짐
            elif mode == 2: # median filter: 소금-후추 노이즈 제거
                blur = cv2.medianBlur(img, val) #중앙값 블러
                # 커널이 클수록 제거 잘됨
            elif mode == 3: #양방향 필터링
                blur = cv2.bilateralFilter(img, val, 75, 75)
                # 비슷한 intensity를 가진 픽셀까지 고려해 필터링
                # edge 보존됨
                #두 픽셀의 거리, 명암 차이 고려
            else:
                print("try")
                break
        except:
            break

        cv2.imshow("blur_bar", blur)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        mode = cv2.getTrackbarPos("blur_mode", "blur_bar")
        val = cv2.getTrackbarPos("kernel_size", "blur_bar")

    cv2.destroyAllWindows()

bluring()

# 가우시안
# 현재 픽셀값과 이웃 픽셀값들의 가중평균을 이용해서 현재 픽셀 값을 대체
# 현재 픽셀과 가까울스록 가중치가 큼
# 픽셀 사이의 거리만을 가중치에 반영

# 빌라테랄
# 픽셀 사이의 거리+픽셀값의 차이를 동시에 가중치에 반영
# pixel의 값의 차이에도 의존하게 하면, noise가 있는 부분에서는 넓은 종모양의
# 분포를 가지고 색상의 경계부분에서는 종 모양을 뾰족하게 만들 수 있을 것이다
