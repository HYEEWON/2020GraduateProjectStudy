import cv2
# edge: 픽셀값이 급격히 변하는 지점, 1차 미분값이 커짐
# sobel: 1차 미분을 사용, 간결, 노이즈에 약함
# laplacian: 2차 미분을 사용, 두꺼운 검출, 노이즈에 민감
rice = cv2.imread("Image/rice.jpg", cv2.IMREAD_COLOR)
rice = cv2.resize(rice, (400, 300))
gray_rice = cv2.cvtColor(rice, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(rice, 100, 255)
# (이미지, 임계1, 임계2, 커널크기, L2그라디언트)
# 임계1 이하는 가장자리에서 제외
# 임계2 이상은 가장자리로 간주

sobel_x = cv2.Sobel(gray_rice, cv2.CV_64F, 1, 0, 3)
# (그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법)
# 1차 미분의 근사값 계산을 위해 커널과 이미지를 컨볼루션하여 에지를 검출
# x방향 에지 검출과 y방향 에지 검출을 위해 별도의 커널 사용
# 1, 0: 세로 / 0, 1: 가로

sobel_x = cv2.convertScaleAbs(sobel_x)
# sobel 결과에 절대값 적용후 값 범위를 8비트 unsigned int로 변경
sobel_y = cv2.Sobel(gray_rice, cv2.CV_64F, 0, 1, 3)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)

laplacian = cv2.Laplacian(gray_rice, cv2.CV_8U, ksize=3)


cv2.imshow("Canny", canny)
cv2.imshow("sobel", sobel)
cv2.imshow("laplacian", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()