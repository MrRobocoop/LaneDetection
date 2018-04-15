import cv2

id = 1
for i in range(1,8):
	img = cv2.imread("%s.jpg"%str(i))
	dst = cv2.resize(img,(320,240), interpolation=cv2.INTER_CUBIC)
	cv2.imwrite("c%s.jpg"%str(i),dst)
