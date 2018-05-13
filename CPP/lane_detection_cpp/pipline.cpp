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


cv::Mat pipeline(Mat input_img)
{
    static cv::Mat mapx, mapy, M, mask, erode_k;
    cv::Mat image_undist, image_dist, img_gray, fushi,dst;
    static Size size_warp(320,300);
    static int counter = 0;
    if(counter == 0){


        //cv::Files
        /*初始化畸变矫正参数*/
        FileStorage fsRead("./mapx.xml", FileStorage::READ);
        fsRead["mapx"]>>mapx;
        fsRead.release();
        FileStorage fyRead("./mapy.xml", FileStorage::READ);
        fyRead["mapy"]>>mapy;
        fyRead.release();

        /*逆透视变换参数初始化*/
        vector<Point2f> pts1(4);
        pts1[0] = Point2f(115,108);
        pts1[1] = Point2f(190,108);
        pts1[2] = Point2f(264,197);
        pts1[3] = Point2f(61,198);
        vector<Point2f> pts2(4);
        pts2[0] = Point2f(100,200);
        pts2[1] = Point2f(200,200);
        pts2[2] = Point2f(200,300);
        pts2[3] = Point2f(100,300);

        M = getPerspectiveTransform(pts1,pts2);

        /*mask初始化*/
        erode_k = getStructuringElement(MORPH_CROSS, Size(5, 5));
        mask = Mat::ones(240,320,CV_8U);
        mask = mask*255;
        remap(mask, mask, mapx, mapy, INTER_LINEAR);
        warpPerspective(mask, mask,M, size_warp, INTER_CUBIC);
        erode(mask, mask, erode_k);
        erode(mask, mask, erode_k);
    }else{
        counter ++;
        if(counter >=100){
            counter = 1;
        }
    }

    /*获取图像*/

    image_undist = input_img;
    /*图像的畸变矫正*/
    cv::remap(image_undist, image_dist, mapx, mapy, INTER_LINEAR);

    /*图像的预处理*/
    //1. 灰度化处理
    cvtColor(image_dist, img_gray ,CV_BGR2GRAY);

    //2.逆透视变换

    warpPerspective(img_gray, fushi,M, size_warp, INTER_CUBIC);

    //3. 高斯模糊
    GaussianBlur(fushi, fushi, Size(5, 5), 0, 0);
    //4. 边缘提取
    Canny(fushi, dst, 40, 90);
    bitwise_and(dst,mask, dst);
    dilate(dst,dst, erode_k);

    return dst;
}
