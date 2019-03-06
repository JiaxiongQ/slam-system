#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H
#include "define.h"
#include "gridTracker.h"
#include "myEigen.h"
#include "myCeres.h"
#include "TrackerBall.h"
#define ATD at<double>
extern vector<Mat> motions_Show;
extern vector<Mat> rotations_Show;
extern vector< vector<Point2f> > points2_for_show;//特征点集
extern sem_t sem;
extern vector<vector<SColorPoint3D>> vPointCloud;
extern int NC;
extern int NP;
extern bool startFlag;
extern CMutex g_Lock;
//创建一个互斥锁

class reconstructure
{
public:
	vector< vector<Point2f> >points2_for_all;//特征点集
	vector< vector<Point2f> >features_for_all;//特征点集
	vector< vector<Point2f> >tracked_features_for_all;
	vector<Point2f> points2_before;
	vector<vector<Vec3b>> colors_for_all;
	vector< vector<DMatch> > matches_for_all;	//匹配信息（匹配的特征点在前后帧的位置索引）
	vector<DMatch> matches_before;
	vector<Point3d> structure;//三维点集
	vector<Vec3b> colors;
	vector< vector<Point3d> > structures;
	vector<Point3d> structure_before;
	vector<Vec3b> colors_before;
	vector< vector<Vec3b> > allcolors;
	vector< vector<int> > correspond_struct_index; //保存第i帧图像中第j个特征点对应的structure中点的索引
	vector<int> correspond_struct_index_before;
	vector<int> correspond_struct_index_err;
	vector<Mat> rotations;//旋转矩阵序列
	Mat rotation_before;
	Mat rotation_err;
	vector<Mat> motions;//平移矩阵序列
	Mat motion_before;
	Mat motion_err;
	vector<Point3d> object_points;
	vector<Point2f> image_points;
	vector<Mat> extrinsics;
	vector<vector<double>> rpjError;
	bool firstFlag;
	CMutex g_Lock2;
	CMutex g_Lock3;
	CMutex g_Lock4;
	mutex mtx2;
	mutex mtx3;
	mutex mtx4;
public:
	int NCAMS;
	Mat K;
	vector< vector<Node*> > srcNodes;
	vector<Mat> imgs;//帧序列
	int step;
	int widSize;
	
	reconstructure() :step(3), widSize(5){}

	void init();
	void initNext(int blockNum);
	
	void init_structure();
	
	void run();
	
	void showError();
	
	void dealError();
};

vector<DMatch> featureMatcher(vector<Point2f>& trackedPoints, vector<Point2f>& detectedPoints,vector<int>& right_points_to_find_back_index);

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);

void reconstruct(Mat& K, Mat& R0, Mat& T0, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2,
	vector<Point3d>& structure);

void maskout_points(vector<Point2f>& p1, Mat& mask);

void maskout_colors(vector<Vec3b>& p1, Mat& mask);

void get_matched_points(vector<Point2f>& p1, vector<Point2f>& p2, vector<DMatch> matches,
	vector<Point2f>& out_p1, vector<Point2f>& out_p2);

void get_matched_colors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches,
	vector<Vec3b>& out_c1,vector<Vec3b>& out_c2);

void get_objpoints_and_imgpoints(vector<DMatch>& matches, vector<int>& struct_index,
	vector<Point3d>& structure,
	vector<Point2f>& key_points,
	vector<Point3d>& object_points,
	vector<Point2f>& image_points);

void fusion_structure(vector<DMatch>& matches, vector<int>& struct_index, vector<int>& next_struct_index,
	vector<Point3d>& structure, vector<Point3d>& next_structure);

double normofTransform( cv::Mat rvec, cv::Mat tvec );
#endif