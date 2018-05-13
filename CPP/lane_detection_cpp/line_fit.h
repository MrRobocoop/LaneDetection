#ifndef LINE_FIT_H
#define LINE_FIT_H
#include "cv.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>


#include <armadillo>
using namespace arma;
using namespace std;

bool line_fit(mat input_img, vector<float> &ret, int windows_size, int line_margin, int vertical_margin, cv::Mat &temp);

//你的代码写在这里

#endif

