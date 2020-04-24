import cv2

src = cv2.imread("Image/fruit.jpg", cv2.IMREAD_COLOR)
dst = cv2.blur(src, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
# (원본 이미지, (커널x크기, 커널y크기), 앵커 포인트, 픽셀 외삽법)
# 앵커 포인트: 커널의 중심점

cv2.imshow("blur", dst)
cv2.waitKey(0)
cv2.destroyWindows()

