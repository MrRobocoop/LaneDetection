#include "pipline.h"
#include "cv.h"
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


bool line_fit(mat input_img, vector<mat> &ret, int windows_size, int line_margin, int vertical_margin, cv::Mat &temp){

    mat histogram;
    //求下半部分直方图
    mat intr;
    intr = input_img;
    histogram = intr.rows(intr.n_rows*3/4, intr.n_rows-1);
    histogram = sum(histogram, 0);
    //cout<<histogram<<endl;
    //cout<<histogram.n_elem<<endl;
    int midpoint = histogram.n_elem/2;
    int margin = windows_size;
    int local_maximum_window = 50;
    vector<int> bases;

    //find the local maximum of historgram
    for(int i = 0; i<histogram.n_cols; i++){

        int num_of_loss = 0;
        int num_in_window=0;
        for(int j=-local_maximum_window/2; j<= local_maximum_window/2; j++){
            if((i+j)<0){
                num_of_loss++;
                continue;
            }else if((i+j)>=histogram.n_cols){
                num_of_loss++;
                continue;
            }else{
                //cout<<histogram(i+j)<<" ";
                if((histogram(i)>=histogram(i+j))&&(histogram(i)!=0)){
                    num_in_window++;

                }else{
                    //break;
                }
            }
        }

        if(num_in_window==(local_maximum_window-num_of_loss+1)){
            bases.push_back(i);
        }
    }

    vec bases_before_filter(bases.size());
    for(int i=0; i<bases.size(); i++){
        //cout<<bases[i]<<endl;
        bases_before_filter(i) = bases[i];

    }
    vec left_bases = bases_before_filter(find(bases_before_filter < midpoint));
    vec right_bases = bases_before_filter(find(bases_before_filter >= midpoint));

    int leftx_base = margin/2;
    int rightx_base = intr.n_rows-margin/2;
    if(left_bases.is_empty() != 1){
        leftx_base = min(left_bases);
        //rightx_base = min(right_bases);
    }
    if(right_bases.is_empty() != 1){
        //leftx_base = min(left_bases);
        rightx_base = min(right_bases);
    }

    int nwindows = 10;
    int window_hight = input_img.n_rows/nwindows;
    uvec nonzero = find(input_img>0);
    uvec nonzerox = nonzero/320;
    uvec nonzeroy = nonzero-(nonzero/300)*300;
    int leftx_current = leftx_base;
    int rightx_current = rightx_base;
    int lefty_current = input_img.n_rows-1;
    int righty_current = input_img.n_rows-1;
    int minpix = 5;
    mat left_lane_inds, right_lane_inds;
    /*按照滑动窗搜索非零元素， 保存为左或者右边线*/
    for(int window=0; window<nwindows; window++){
        int win_y_high = input_img.n_rows-1 - (window+1)*window_hight;
        int win_y_low = input_img.n_rows-1 -window*window_hight;
        int win_xleft_low = leftx_current - margin;
        int win_xleft_high = leftx_current + margin;
        int win_xright_low = rightx_current - margin;
        int win_xright_high = rightx_current + margin;
        cv::rectangle(temp,cv::Rect(win_xleft_low,win_y_high,2*margin, window_hight),cv::Scalar(0,0,255),1,1,0);
        cv::rectangle(temp,cv::Rect(win_xright_low,win_y_high,2*margin, window_hight),cv::Scalar(0,255,0),1,1,0);
        mat left_window_inds, right_window_inds;
        for(int i=0; i<nonzero.n_elem; i++){
            if((nonzeroy(i) <= win_y_low)&&(nonzeroy(i) >= win_y_high)&&(nonzerox(i) >= win_xleft_low) && (nonzerox(i) < win_xleft_high)){
                if((lefty_current-nonzeroy(i))<vertical_margin){
                    mat temp2(1,2);
                    temp2(0,0) = nonzeroy(i);
                    temp2(0,1) = nonzerox(i);

                    left_window_inds = join_cols(left_window_inds, temp2);
                    left_lane_inds = join_cols(left_lane_inds, temp2);
                }

            }else if((nonzeroy(i) <= win_y_low)&&(nonzeroy(i) >= win_y_high)&&(nonzerox(i) >= win_xright_low) && (nonzerox(i) < win_xright_high)){
                if((righty_current-nonzeroy(i))<vertical_margin){
                    mat temp2(1,2);
                    temp2(0,0) = nonzeroy(i);
                    temp2(0,1) = nonzerox(i);
                    right_window_inds = join_cols(right_window_inds, temp2);
                    right_lane_inds = join_cols(right_lane_inds, temp2);
                }
            }
        }

        mat templ = mean(left_window_inds);
        if(left_window_inds.is_empty() == 0){
            leftx_current = templ(0,1);
            lefty_current = templ(0,0);

        }
        templ = mean(right_window_inds);
        if(right_window_inds.is_empty() == 0){
            rightx_current = templ(0,1);
            righty_current = templ(0,0);
        }
        //cout<<"lb"<<leftx_base<<"rb"<<rightx_base<<endl;
    }

    /*对左右边线分别做曲线拟合处理*/
    vec left_line_parm, right_line_parm;


    //用指针访问像素，速度最快

    //cout<<"left_inds_num:"<<left_lane_inds.n_rows<<endl;
    if(left_lane_inds.n_rows >= 40){
        vec lxp = left_lane_inds.col(1);
        vec lyp = left_lane_inds.col(0);
        polyfit(left_line_parm,lyp,lxp,2);
        //cout<<left_line_parm<<endl;
        vec yl = linspace<vec>(0, 300, 300);
        vec xl = polyval(left_line_parm,yl);
        for(int i=0;i<xl.n_elem;i++){
            //cout<<xl(i)<<" "<<yl(i)<<endl;
            if((xl(i)<300)&&(xl(i)>0)){
                temp.at<cv::Vec3b>(int(yl(i)),int(xl(i)))[0]=255;
                temp.at<cv::Vec3b>(int(yl(i)),int(xl(i)))[1]=0;
                temp.at<cv::Vec3b>(int(yl(i)),int(xl(i)))[2]=0;
            }


        }
        //polylines(temp, yl, False, (255, 0, 0),1)
    }else{
        left_line_parm.reset();
    }
    if(right_lane_inds.n_rows>=40){
        right_line_parm = polyfit(right_lane_inds.col(0),right_lane_inds.col(1),2);
        vec yr = linspace<vec>(0, 300, 300);
        vec xr = polyval(right_line_parm,yr);

        for(int i=0;i<xr.n_elem;i++){
            //cout<<xr(i)<<" "<<yr(i)<<endl;
            if((xr(i)<300)&&(xr(i)>0)){
                temp.at<cv::Vec3b>(int(yr(i)),int(xr(i)))[0]=0;
                temp.at<cv::Vec3b>(int(yr(i)),int(xr(i)))[1]=255;
                temp.at<cv::Vec3b>(int(yr(i)),int(xr(i)))[2]=255;
            }

        }

    }else{
        right_line_parm.reset();
    }



    return 0;
}
