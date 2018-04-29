import time
import cv2
#qimport cv2.cv as cv
import numpy as np
#import matplotlib.pyplot as plt
#import tkinter
from line_collect import find_lines, line_class, classify_line
from lane_fit import line_fit
# initialize the camera and grab a reference to the raw camera capture

def lane_pipline():
    start = time.time()
    end = time.time()

    #pts1 = np.float32([[90,1], [230,1], [320,203], [1, 203]])
    pts1 = np.float32([[115, 108], [190, 108], [264, 197], [61, 198]])
    pts2 = np.float32([[100,200], [200, 200], [200, 300], [100, 300]])
    erode_k = np.uint8([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    idx = 0
    mapx = np.load('caliberation/caliberation/mapx.npy')
    mapy = np.load('caliberation/caliberation/mapy.npy')

    # capture frames from the camera
    fushi = np.ones((320,300), np.uint8)*255
    mask = np.ones((240,320), np.uint8)*255
    mask = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)
    mask = cv2.warpPerspective(mask, M,(300, 320))
    mask = cv2.erode(mask, erode_k)
    mask = cv2.erode(mask, erode_k)
    for i in range(1, 299):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        end = time.time()
        print 1/(end-start)
        start = time.time()
        idx = i
        image = cv2.imread('data_all/%s.jpg' %str(idx),1)
        undis_image = cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST)
        img_gray = cv2.cvtColor(undis_image, cv2.COLOR_BGR2GRAY)
        img_cont = np.copy(img_gray)
        r, thresh = cv2.threshold(img_cont, 130, 255, cv2.THRESH_BINARY)
        img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = 0
        max_cnt = 0
        for contour in contours:
            #img_cont = cv2.drawContours(thresh, contour, -1, (0, 0, 125), 3)
            area = cv2.contourArea(contour)
            if(area > max_contour):
                max_contour = area
                max_cnt = contour
        zeros = np.zeros_like(image)
        #print max_cnt
        img_cont = cv2.drawContours(zeros, max_cnt, -1, (255,0,0),cv2.FILLED)


        # cut out image
        #image = ii.image_cut_out(image, 0, 320, 0, 200)

        #test_image = cv2.imread('shit.jpg', 1)

        #rows, cols, ch = test_image.shape

        cv2.warpPerspective(img_gray, M,(300, 320),fushi,cv2.INTER_CUBIC)
        kernel = np.ones((5,5), np.float32) / 25
        #fushi = cv2.equalizeHist(fushi)
        test = np.zeros((300,320,3),np.uint8)
        dst = cv2.filter2D(fushi, -1, kernel)
        #fushi = cv2.filter2D(fushi, -1, kernel)
        #print dst

        dst = cv2.Canny(dst, 40, 90)
        thresh = cv2.dilate(thresh, erode_k)
        thresh = cv2.dilate(thresh, erode_k)
        #dst = cv2.dilate(dst,erode_k)
        dst = cv2.bitwise_and(dst,dst, mask=mask)
        #dst = cv2.bitwise_and(dst,dst, mask=thresh)
        #dst = cv2.dilate(dst, erode_k)
        dst = cv2.dilate(dst, erode_k)
        #dst = cv2.warpPerspective(dst, M, (200,320))
        #lines = cv2.HoughLines(dst,1, np.pi/180, 1)
        #print lines


        ret = line_fit(dst, window_size=40, line_margin=200, vertical_margin=150)

        #plt.figure(1)
        #plt.axis([0, 300, 320, 0])
        co = 0.1
        left_fit = ret['left_fit']
        left_fit = np.poly1d(left_fit)
        right_fit = ret['right_fit']
        right_fit = np.poly1d(right_fit)
        left_y = left_fit(np.array(range(0,300)))
        left_x = np.array(range(0,300))
        right_y = right_fit(np.array(range(0, 300)))
        right_x = np.array(range(0, 300))
        #plt.plot(left_y, left_x,".")
        #plt.plot(ret['leftx'], ret['lefty'], "*")
        #plt.plot(right_y, right_x,".")
        #plt.plot(ret['rightx'], ret['righty'], "*")
        left_pts = np.stack((left_y, left_x), axis=1)
        right_pts = np.stack((right_y, right_x), axis=1)
        left_pts = left_pts.astype(np.int32)
        right_pts = right_pts.astype(np.int32)        #print left_pts
        fushi = cv2.polylines(fushi, [left_pts], False, (255, 0, 0),1)
        fushi = cv2.polylines(fushi, [right_pts], False, (0, 0, 255), 1)
        try:
            for l in ret:
                #rint l.points
                plt.plot(l.points[:, 0], l.points[:, 1], ".")
                co = co + 0.1
                #plt.plot(rlx, rly, "b.")
        except:
            pass

        cv2.imshow("Frame", dst)
        cv2.imshow("origin", fushi)
        #cv2.imshow("boundary", )
        cv2.imshow("lane_find", thresh)



        key = cv2.waitKey(100) & 0xFF

        # clear the stream in preparation for the next frame
        #rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        #plt.show()

if __name__ == '__main__':
    lane_pipline()
