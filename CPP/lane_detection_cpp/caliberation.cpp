#include "stdio.h"
#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


int caliberation(int argc, char** argv)
{
    string filePath = "/home/raosiyue/raspi_lane_detection/LaneDetection/caliberation/caliberation";
    vector<string> files;
    stringstream ss;
    string str;
    for(int i=1; i<=8; i++){
        ss.clear();
        ss<<i;
        ss>>str;
        cout<<str<<endl;
        files.push_back(filePath+"/"+str+".jpg");

    }
    ////获取该路径下的所有文件
    //getFiles(filePath, files);

    const int board_w = 9;
    const int board_h = 6;
    const int NPoints = board_w * board_h;//棋盘格内角点总数
    const int boardSize = 1; //mm
    Mat image,grayimage;
    Size ChessBoardSize = cv::Size(board_w, board_h);
    vector<Point2f> tempcorners;

    int flag = 0;
    flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    //flag |= cv::fisheye::CALIB_CHECK_COND;
    flag |= cv::fisheye::CALIB_FIX_SKEW;
    //flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;

    vector<Point3f> object;
    for (int j = 0; j < NPoints; j++)
    {
        object.push_back(Point3f((j % board_w) * boardSize, (j / board_w) * boardSize, 0));
    }

    cv::Matx33d intrinsics;//z:相机内参
    cv::Vec4d distortion_coeff;//z:相机畸变系数

    vector<vector<Point3f> > objectv;
    vector<vector<Point2f> > imagev;

    Size corrected_size(320, 240);
    Mat mapx, mapy;
    Mat corrected;

    ofstream intrinsicfile("intrinsics_front1103.txt");
    ofstream disfile("dis_coeff_front1103.txt");
    int num = 0;
    bool bCalib = false;
    while (num < files.size())
    {
        cout<<files[num]<<endl;
        image = imread(files[num]);

        if (image.empty())
            break;
        //imshow("corner_image", image);
        //waitKey(10);
        cvtColor(image, grayimage, CV_BGR2GRAY);
        IplImage tempgray = grayimage;
        bool findchessboard = cvCheckChessboard(&tempgray, ChessBoardSize);

        if (findchessboard)
        {
            bool find_corners_result = findChessboardCorners(grayimage, ChessBoardSize, tempcorners, 3);
            if (find_corners_result)
            {
                cornerSubPix(grayimage, tempcorners, cvSize(5, 5), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
                drawChessboardCorners(image, ChessBoardSize, tempcorners, find_corners_result);
                //imshow("corner_image", image);
                //cvWaitKey(10);

                objectv.push_back(object);
                imagev.push_back(tempcorners);
                cout << "capture " << num << " pictures" << endl;
            }
        }
        tempcorners.clear();
        num++;
    }

    cv::fisheye::calibrate(objectv, imagev, cv::Size(image.cols,image.rows), intrinsics, distortion_coeff, cv::noArray(), cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));
    //fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff, cv::Matx33d::eye(), intrinsics, corrected_size, CV_16SC2, mapx, mapy);
    intrinsics(0,0) = intrinsics(0,0)/8.1;
    intrinsics(1,1) = intrinsics(1,1)/8.1;
    intrinsics(0,2) = intrinsics(0,2)/8.1;
    intrinsics(1,2) = intrinsics(1,2)/8.1;
    distortion_coeff(0) = -0.03060708;
    distortion_coeff(1) = -0.01128285289;
    distortion_coeff(2) = 0.0405098482;
    distortion_coeff(3) = -0.053502175458;
    cv::Matx33d new_intrinsics;
    //intrinsics.copyTo(new_intrinsics);
    new_intrinsics(0,0) = intrinsics(0,0)/1.5;
    new_intrinsics(1,1) = intrinsics(1,1)/1.5;
    new_intrinsics(0,1) = intrinsics(0,1);
    new_intrinsics(0,2) = intrinsics(0,2);
    new_intrinsics(1,0) = intrinsics(1,0);
    new_intrinsics(1,2) = intrinsics(1,2);
    new_intrinsics(2,0) = intrinsics(2,0);
    new_intrinsics(2,1) = intrinsics(2,1);
    new_intrinsics(2,2) = intrinsics(2,2);

    fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff, cv::Matx33d::eye(), new_intrinsics, corrected_size, CV_16SC2, mapx, mapy);

    FileStorage fsFeature("./mapx.xml", FileStorage::WRITE);
    fsFeature<<"mapx"<<mapx;
    fsFeature.release();

    FileStorage fyFeature("./mapy.xml", FileStorage::WRITE);
    fyFeature<<"mapy"<<mapy;
    fyFeature.release();

    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
        {
            intrinsicfile<<intrinsics(i,j)<<"\t";
        }
        intrinsicfile<<endl;
    }
    for(int i=0; i<4; ++i)
    {
        disfile<<distortion_coeff(i)<<"\t";
    }
    intrinsicfile.close();
    disfile.close();

    num = 0;
    while (num < files.size())
    {
        image = imread("/home/raosiyue/raspi_lane_detection/LaneDetection/caliberation/caliberation/254.jpg");
        num = files.size();
        if (image.empty())
            break;
        remap(image, corrected, mapx, mapy, INTER_LINEAR);

        //imshow("corner_image", image);
        //imshow("corrected", corrected);
        //cvWaitKey(0);
    }

    //cv::destroyWindow("corner_image");
    //cv::destroyWindow("corrected");

    image.release();
    grayimage.release();
    corrected.release();
    mapx.release();
    mapy.release();

    return 0;
}
