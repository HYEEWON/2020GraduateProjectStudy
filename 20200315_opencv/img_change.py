import cv2
src = cv2.imread("Image/fruit.jpg", cv2.IMREAD_COLOR)
cut_img = src.copy()
cut_img = src[100:600, 200:700] # [높이, 너비]
gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow("src", src)
cv2.imshow("cut_img", cut_img)
cv2.imshow("gray_img", gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
