import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(21,21),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(gray, (9,6), corners,ret)
        #cv2.imshow('img',gray)
        #cv2.waitKey(0)

cv2.destroyAllWindows()
objpoints = np.asarray([objpoints], dtype='float64').reshape(-1, 1, 54, 3)
imgpoints = np.asarray([imgpoints], dtype='float64').reshape(-1, 1, 54, 2)

mtx = np.eye(3)
dist = np.zeros(4)
calib_flags=cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
#rvecs=np.asarray([[[np.zeros(3).tolist() for i in xrange(img)]]],dtype='float64').reshape(-1,1,1,3)
#tvecs=np.asarray([[[np.zeros(3).tolist() for i in xrange(img)]]],dtype='float64').reshape(-1,1,1,3)
rvecs=np.asarray([[[np.zeros(3).tolist() for i in xrange(objpoints.shape[0])]]],dtype='float64').reshape(-1,1,1,3)
tvecs=np.asarray([[[np.zeros(3).tolist() for i in xrange(objpoints.shape[0])]]],dtype='float64').reshape(-1,1,1,3)
camera_matrix = np.eye(3)
dist_coeffs = np.zeros(4)

camera_matrix = np.eye(3)
camera_matrix[0][0]=1000
camera_matrix[1][1]=1000
camera_matrix[0][2]=(gray.shape[::-1][0]-1)/2
camera_matrix[1][2]=(gray.shape[::-1][1]-1)/2
mtx = camera_matrix
dist = dist_coeffs
test = cv2.imread('254.jpg')
#gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], mtx, dist, rvecs, tvecs, flags = calib_flags)
np.save('mtx.npy',mtx)
np.save('dist.npy',dist)
print gray.shape[::-1]
for i in range(1,8):
    img = cv2.imread('254.jpg')
    #img = cv2.resize(img, (2592, 1944), interpolation=cv2.INTER_NEAREST)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    nk = mtx.copy()

    nk = mtx/8.1
    nk[0, 0] = mtx[0, 0] / 8.1
    nk[1, 1] = mtx[1, 1] / 8.1
    nk[2,2] = mtx[2,2]

    new_nk = nk.copy();
    new_nk[0,0] = nk[0,0]/1.5
    new_nk[1,1] = new_nk[1,1]/1.5
    mapx,mapy = cv2.fisheye.initUndistortRectifyMap(nk,dist,np.eye(3),new_nk,(w,h),5)
    np.save('mapx.npy', mapx)
    np.save('mapy.npy', mapy)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_NEAREST)
    #dst = cv2.fisheye.undistortImage(img,mtx,dist,newcameramtx)
    #dst = cv2.resize(dst, (320, 240), interpolation=cv2.INTER_NEAREST)
    # crop the image
    x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]


    #dst = cv2.undistort(img, mtx, dist, None, None)
    #dst = dst[-100:-100+h, -100:-100+w]
    # crop the image
    x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult%s.bmp'%str(i),dst)
    cv2.imwrite('mapx.jpg', mapx)
    cv2.imwrite('mapy.jpg', mapy)
    cv2.imshow('img',dst)
    cv2.waitKey(500)
cv2.destroyAllWindows()
