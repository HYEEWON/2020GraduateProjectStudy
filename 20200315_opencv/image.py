import cv2

image = cv2.imread("Image/moon.jpg", cv2.IMREAD_ANYCOLOR)
# cv2.IMREAD_ANYCOLOR: 가능한 3채널 색 이용
# cv2.IMREAD_UNCHAGED: 원본 사용

image = cv2.resize(image, (300, 300))
cv2.imshow("moon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()