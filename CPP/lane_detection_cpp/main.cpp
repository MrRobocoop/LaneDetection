#include "pipline.h"
#include "cv.h"
#include "line_fit.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>


#include <armadillo>
using namespace arma;
using namespace std;

static void Cv_mat_to_arma_mat(const cv::Mat& cv_mat_in, arma::mat& arma_mat_out)
{//convert unsigned int cv::Mat to arma::Mat<double>
    for(int r=0;r<cv_mat_in.rows;r++){
        for(int c=0;c<cv_mat_in.cols;c++){
            arma_mat_out(r,c)=cv_mat_in.data[r*cv_mat_in.cols+c]/255.0;
        }
    }
};
template<typename T>
static void Arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in,cv::Mat_<T>& cv_mat_out)
{
    cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat_in.n_cols),
                              static_cast<int>(arma_mat_in.n_rows),
                              const_cast<T*>(arma_mat_in.memptr())),
                  cv_mat_out);
};






int main(void){
    cv::Mat image_undist, dst;
    mat img(300,320);
    stringstream ss;
    string str;
    vector<float> test;
    test.push_back(0.0);
    test.push_back(0.0);
    test.push_back(0.0);
    int num = 0;
    int key;
    float k1 = 1;
    float k2 = 1;
    float k3 = 5000;
    while(num++<60){
        ss.clear();
        ss<<num;
        ss>>str;
        //cout<<str<<endl;
        image_undist = cv::imread("/home/raosiyue/raspi_lane_detection/LaneDetection/CPP/lane_detection_cpp/data_all/"+str+".jpg",1);
        cv::Mat temp(300, 320, CV_8UC3, cv::Scalar::all(0));

        /*预处理*/
        dst = pipeline(image_undist);
        Cv_mat_to_arma_mat(dst, img);
        //img.print();
        //cout<<img.n_rows<<" "<<img.n_cols<<endl;
        //cout<<img.row(100)<<endl;
        line_fit(img, test,40, 200, 60, temp);
        cout<<"center offset is:"<<test[0]<<"  yaw offset is:"<<test[1]<<endl;


        float steer_angle = 0 + k1*test[0] + k2*test[1] + k3*test[2];
        cout<<k1 <<k2<<endl;
        cout<<"steer angle is:"<<steer_angle<<endl;
        cv::imshow("canny", dst);
        cv::imshow("scan window", temp);

        key = cvWaitKey(0);
        if(key == 'q'){
            break;
        }else if(key == 'k'){
            cin>>k1>>k2;

        }

    }
    return 0;
}




