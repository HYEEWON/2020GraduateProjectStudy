import cv2

capture = cv2.VideoCapture(0) # 카메라 번호, 0=내장 카메라 번호
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    # ret: 카메라의 상태 저장, 정상=True
    # frame: 현재 프레임을 저장
    cv2.imshow("videoframe", frame)
    if cv2.waitKey(1) > 0: break # 키 입력이 있을 때까지 반복
    # if cv2.waitKey(1) == ord('q'): break # q입력 시 종료

capture.release() # 카메라 메모리 해제
cv2.destroyAllWindows() # 모든 윈도우창 닫음
# cv2.destroyWindow("윈도우 창 이름") # 특정 윈도우 창을 닫음