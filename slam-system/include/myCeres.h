#ifndef MYCERES_H
#define MYCERES_H
#include "glog/logging.h"  
#include "ceres/ceres.h" 
#include "ceres/rotation.h" 
#include "define.h"
extern vector<cv::Mat> motions_Show;
extern vector<cv::Mat> rotations_Show;
extern vector<Eigen::Vector3d> angles_Show;
vector<double> Bundle_Adjustment(
	cv::Mat& intrinsic,
	vector<cv::Mat> extrinsics,
	vector< vector<int> >& correspond_struct_idx,
	vector< vector<cv::Point2f> >& key_points_for_all,
	vector<cv::Point3d>& structure,
	bool consFlag
	);
#endif