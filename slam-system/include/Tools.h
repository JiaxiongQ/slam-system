#pragma once
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <stdio.h>
using namespace std;

#define NEIGHBORNUM 5
#define ABS(x) (((x) >= 0) ? (x) : (-(x)))

#define RADIUS 30
#define TEMPORALBUFFERSIZE 60

#define VecPt2f vector<cv::Point2f>
#define VecVecPt2f vector<VecPt2f>   

double MyRound(double v);

void myPrint(cv::Mat &m);

cv::Point2f myHomographyTrans(cv::Point2f pt,cv::Mat H);

vector<cv::Point2f> myHomographyTrans(vector<cv::Point2f> &pts,cv::Mat H);

void DrawFeaturePoints(cv::Mat &Img,vector<cv::Point2f> &pts);

double inline Distance(cv::Point2f pt1,cv::Point2f pt2);

#ifndef __HOMOPARA__
#define __HOMOPARA__
struct HomoParam{
	double theta,phy,s1,s2,tx,ty,v1,v2;
	HomoParam(){
	   theta=phy=tx=ty=v1=v2=0.0;
	   s1=s2=1.0;
	}
	void operator=(const HomoParam &homo){
		theta = homo.theta;
		phy = homo.phy;
		s1 = homo.s1;
		s2 = homo.s2;
		tx = homo.tx;
		ty = homo.ty;
		v1 = homo.v1;
		v2 = homo.v2;
	}
};
#endif


cv::Mat GlobalOutLinerRejectorOneIteration(vector<cv::Point2f> &features_img1,vector<cv::Point2f> &features_img2);

cv::Point2f Trans(cv::Mat H,cv::Point2f &pt);

VecPt2f Trans(cv::Mat H,VecPt2f &pts);