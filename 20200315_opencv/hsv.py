import cv2
# h(hue): 색상 0=빨강, 0~180
# s(saturation): 채도 100=색진함, 0~255
# v(value): 명도 100=흰색, 0~255
img = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #hsv채널로 변결
h, s, v = cv2.split(hsv)
#분리된 채널은 단일 채널이므로 흑백의 색상으로만 표현
cv2.imshow("hsv", hsv)
cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)

h = cv2.inRange(h, 8, 20) #inRange(단일 채널 이미지, 최소값, 최대값)
#주황은 8~20의 범위를 가짐
orange = cv2.bitwise_and(hsv, hsv, mask = h) #마스크 덧 씌우기
orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)
cv2.imshow("orange", orange)

cv2.waitKey(0)
cv2.destroyAllWindows()
