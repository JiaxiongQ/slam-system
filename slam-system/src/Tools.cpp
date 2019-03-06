#include "Tools.h"

double MyRound(double v){
	double a = 1.0*int(v);
	double b = v-a;
	
	if(b>=0.5){
	   return a+1.0;
	}else{
	   return a;
	}
}

void myPrint(cv::Mat &m){
	printf("\n");
	for(int i=0;i<m.rows;i++){
		for(int j=0;j<m.cols;j++){
			printf("%03f ",m.at<double>(i,j));
		}
		printf("\n");
	}
}

cv::Point2f myHomographyTrans(cv::Point2f pt,cv::Mat H){
    
	float x = H.at<double>(0,0)*pt.x + H.at<double>(0,1)*pt.y + H.at<double>(0,2)*1;
	float y = H.at<double>(1,0)*pt.x + H.at<double>(1,1)*pt.y + H.at<double>(1,2)*1;
	float z = H.at<double>(2,0)*pt.x + H.at<double>(2,1)*pt.y + H.at<double>(2,2)*1;

	cv::Point2f result(x/z,y/z);
	return result;
}

vector<cv::Point2f> myHomographyTrans(vector<cv::Point2f> &pts,cv::Mat H){
	vector<cv::Point2f> results;
	for(int i=0;i<pts.size();i++){
		results.push_back(myHomographyTrans(pts[i],H));
	}
	return results;
}

void DrawFeaturePoints(cv::Mat &Img,vector<cv::Point2f> &pts){
    
	for(int i=0;i<pts.size();i++){
		cv::circle(Img,pts[i],2,cv::Scalar(0,255,0),-1);	
	}
}

double inline Distance(cv::Point2f pt1,cv::Point2f pt2){
	return sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y));
}

cv::Mat GlobalOutLinerRejectorOneIteration(vector<cv::Point2f> &features_img1,vector<cv::Point2f> &features_img2){
	
	if(features_img1.size()>10){
		vector<cv::Point2f> temp1,temp2;
		temp1.resize(features_img1.size());
		temp2.resize(features_img2.size());
		for(int i=0;i<features_img1.size();i++){
			temp1[i] = features_img1[i];
			temp2[i] = features_img2[i];
		}
		features_img1.clear();
		features_img2.clear();

		vector<uchar> mask;
		cv::Mat H = cv::findHomography(cv::Mat(temp1),cv::Mat(temp2),mask,CV_RANSAC);

		for(int k=0;k<mask.size();k++){
			if(mask[k]==1){
				features_img1.push_back(temp1[k]);
				features_img2.push_back(temp2[k]);
			}
		}

		return H;
	}else{
		return cv::Mat::eye(3,3,CV_64F);
	}
}

cv::Point2f Trans(cv::Mat H,cv::Point2f &pt){
	cv::Point2f result;

	double a = H.at<double>(0,0) * pt.x + H.at<double>(0,1) * pt.y + H.at<double>(0,2) ;
	double b = H.at<double>(1,0) * pt.x + H.at<double>(1,1) * pt.y + H.at<double>(1,2) ;
	double c = H.at<double>(2,0) * pt.x + H.at<double>(2,1) * pt.y + H.at<double>(2,2) ;

	result.x = a/c;
	result.y = b/c;

	return result;
}

VecPt2f Trans(cv::Mat H,VecPt2f &pts){
	VecPt2f results;	
	for(int i=0;i<pts.size();i++){
		results.push_back(Trans(H,pts[i]));
	}
	return results;
}
