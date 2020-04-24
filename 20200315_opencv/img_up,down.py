import cv2

src = cv2.imread("Image/moon.jpg", cv2.IMREAD_COLOR)
# src = cv2.resize(src, (300, 300))
height, width, channel = src.shape
dst = cv2.pyrUp(src, dstsize = (width*2, height*2), borderType = cv2.BORDER_DEFAULT)
# pyrUp: 이미지 확대, pyrDown: 이미지 1/2배 축소
dst2 = cv2.pyrDown(src)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()