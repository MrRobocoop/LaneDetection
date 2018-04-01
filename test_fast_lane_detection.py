import time
import cv2
import cv2.cv as cv
import numpy as np

# initialize the camera and grab a reference to the raw camera capture

start = time.time()
end = time.time()

pts1 = np.float32([[118,60], [188,60], [287,180], [11, 183]]) 
pts2 = np.float32([[25,10], [160, 10], [160, 300], [25, 300]])
erode_k = np.uint8([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])
M = cv2.getPerspectiveTransform(pts1,pts2)
idx = 0

# capture frames from the camera
fushi = np.ones((320,200), np.uint8)*255

mask = np.ones((240,320), np.uint8)*255
mask = cv2.warpPerspective(mask, M,(200, 320))
mask = cv2.erode(mask, erode_k)
for i in range(1, 99):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    end = time.time()
    print 1/(end-start)
    start = time.time()
    idx = idx+1
    image = cv2.imread('%s.jpg' %str(idx),1)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cut out image
    #image = ii.image_cut_out(image, 0, 320, 0, 200)

    #test_image = cv2.imread('shit.jpg', 1)

    #rows, cols, ch = test_image.shape
    
    cv2.warpPerspective(img_gray, M,(200, 320),fushi,cv2.INTER_LINEAR)
    kernel = np.ones((7,7), np.float32) / 25
    dst = cv2.filter2D(fushi, -1, kernel)
    dst = cv2.Canny(dst, 60, 180)
    #dst = cv2.dilate(dst,erode_k)
    dst = cv2.bitwise_and(dst,dst, mask=mask)
    dst = cv2.dilate(dst, erode_k)
    dst = cv2.dilate(dst, erode_k)
    #dst = cv2.warpPerspective(dst, M, (200,320))
    lines = cv2.HoughLines(dst,1, np.pi/180, 1)
    print lines
    try:
        line = lines[:,0,:]
        for rho,theta in line[:]:
            try:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 500*(-b))
                y1 = int(y0 + 500*(a))
                x2 = int(x0 - 500*(-b))
                y2 = int(y0 - 500*(a))
                cv2.line(dst, (x1, y1),(x2,y2),255,2)
    # show the frame
            except:
                pass
    except:
        pass
    cv2.imshow("Frame", dst)
    key = cv2.waitKey(100) & 0xFF
    # clear the stream in preparation for the next frame
    #rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
