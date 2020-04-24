# morphological transformations
# 영상, 이미지를 형태학적 관점에서 접근
# 이미지를 segmentation하여 단순화, 제거, 보정을 통해 형태를 파악
# 입력: 원본이미지 + structuring element(kernel)

# Erosion 침식 이진화된 이미지
# 픽셀에 structuring element를 적용하여 0이 한개라도 있으면 대상 픽셀 제거
# 커널 영역 안의 모든 픽셀 값을 커널 내부의 극소값으로 대체
# 밝은 영역이 줄고 어두운 영역 증가
# 작은 오브젝트 제거에 효과적, 노이즈 제거에 사용

# Dilationm 팽창 이진화된 이미지
# 픽셀에 structuring element를 적용하여 OR연산, 겹치는 부분이 한개라도 있으면 이미지 확장
# 1이 적어도 1개이면 픽셀값을 모두 1로 변경
# 커널 영역 안의 모든 픽셀 값을 커널 내부의 극대값으로 대체
# 어두운 영역 줄고 밝은 영역 증가
# 노이즈 제거 후, 줄어든 크기 복구시 사용

# Opening
# erosion 적용 후 dilation 적용, 작은 object 제거에 적함

# Closing
# dilation 적용 후 erosion 적용, 전체적인 윤곽 파악에 적합

import numpy as np
import cv2

def dilate_erode():
    src = cv2.imread("Image/zebra.jpg")
    src = cv2.resize(src, (400, 300))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # (커널형태, 커널크기, 중심점) 형태: cross(십자가)

    dilate = cv2.dilate(src, kernel, anchor=(-1, -1), iterations=5)
    erode = cv2.erode(src, kernel, anchor=(-1, -1), iterations=5)
    # anchor=(-1, -1) : 커널의 중심부에 고정점 위치
    # iterations: 반복 회수, 더 가늘어지거나 두꺼워짐
    dst = np.concatenate((src, dilate, erode), axis=1)
    # axis=0 : 세로 방향으로 연결
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def opening_closing():
    img = cv2.imread("Image/ab.PNG", cv2.IMREAD_GRAYSCALE)
    alpha = cv2.imread("Image/alpha.PNG", cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #a
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) #b
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    cv2.imshow("opening", opening)
    cv2.imshow("closing", closing)
    cv2.imshow("gradient", gradient)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#dilate_erode()
opening_closing()

# 그레디언트
# dilate(src) - erode(src)
# 객체의 가장자리 반환, 그레이스케일 이미지가 가장 급격하게
# 변하는 곳에서 가장 높은 결과 반환