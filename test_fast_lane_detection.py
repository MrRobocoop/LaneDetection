import time
import cv2
#qimport cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
#import tkinter
from line_collect import find_lines, line_class, classify_line
from lane_fit import line_fit
# initialize the camera and grab a reference to the raw camera capture

def lane_pipline():
    start = time.time()
    end = time.time()

    #pts1 = np.float32([[90,1], [230,1], [320,203], [1, 203]])
    pts1 = np.float32([[115, 108], [190, 108], [264, 197], [61, 198]])
    pts2 = np.float32([[100,150], [200, 150], [200, 280], [100, 280]])
    erode_k = np.uint8([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    idx = 0
    mapx = np.load('caliberation/caliberation/mapx.npy')
    mapy = np.load('caliberation/caliberation/mapy.npy')

    # capture frames from the camera
    fushi = np.ones((320,300), np.uint8)*255
    mask = np.ones((240,320), np.uint8)*255
    mask = cv2.warpPerspective(mask, M,(300, 320))
    mask = cv2.erode(mask, erode_k)
    mask = cv2.erode(mask, erode_k)
    for i in range(1, 299):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        end = time.time()
        print 1/(end-start)
        start = time.time()
        idx = idx+1
        image = cv2.imread('data_all/%s.jpg' %str(idx),1)
        undis_image = cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST)
        img_gray = cv2.cvtColor(undis_image, cv2.COLOR_BGR2GRAY)
        # cut out image
        #image = ii.image_cut_out(image, 0, 320, 0, 200)

        #test_image = cv2.imread('shit.jpg', 1)

        #rows, cols, ch = test_image.shape

        cv2.warpPerspective(img_gray, M,(300, 320),fushi,cv2.INTER_LINEAR)
        kernel = np.ones((7,7), np.float32) / 25
        dst = cv2.filter2D(fushi, -1, kernel)
        r, countour_fushi = cv2.threshold(fushi, 0, 255, cv2.THRESH_OTSU)
        dst = cv2.Canny(dst, 40, 120)
        #dst = cv2.dilate(dst,erode_k)
        #dst = cv2.bitwise_and(dst,dst, mask=mask)
        #dst = cv2.dilate(dst, erode_k)
        dst = cv2.dilate(dst, erode_k)
        #dst = cv2.warpPerspective(dst, M, (200,320))
        #lines = cv2.HoughLines(dst,1, np.pi/180, 1)
        #print lines

        cv2.imshow("Frame", dst)
        cv2.imshow("origin", fushi)
        cv2.imshow("boundary", undis_image)
        ret = find_lines(dst, margin = 50 )
        plt.figure(1)
        plt.axis([0, 200, 320, 0])
        co = 0.1

        try:
            for l in ret:
                #rint l.points
                plt.plot(l.points[:, 0], l.points[:, 1], ".")
                co = co + 0.1
                #plt.plot(rlx, rly, "b.")
        except:
            pass



        #cv2.imshow("lane_find", ret['out_img'])



        key = cv2.waitKey(100) & 0xFF

        # clear the stream in preparation for the next frame
        #rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        plt.show()

if __name__ == '__main__':
    lane_pipline()
