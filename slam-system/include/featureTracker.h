#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H
#include <string>
#include <vector>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
using namespace cv;
using namespace std;
class featureTracker{
public:
	Mat gray;
	Mat grayPre;
	vector<Point2f> points[2];//0:left 1:right
	vector<Point2f> features[2];
	vector<DMatch> matches;
	int maxCount;
	double qLevel;
	double minDist;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> right_points_to_find;
public:
	featureTracker():maxCount(500),qLevel(0.01),minDist(10.0){}

	void detectfeaturePoints()
	{
		features[0].clear();
		goodFeaturesToTrack(grayPre, features[0], maxCount, qLevel, minDist);
	}

	bool addnewPoints()
	{
		if (right_points_to_find.size() <= 50)
		{
			return true;
		}
		return false;
	}

	void showtrackedPoints(Mat& output)
	{
		for (int i = 0; i < points[1].size(); ++i)
		{
			circle(output, points[1][i], 3, Scalar(0, 0, 255), -1);
		}
	}

	void KeyPointsToPoints(vector<KeyPoint>& kps, vector<Point2f>& ps) 
	{
		ps.clear();
		for (unsigned int i = 0; i<kps.size(); i++) 
			ps.push_back(kps[i].pt);
	}

	void process(Mat& img1,Mat& img2,Mat& output)
	{
		img1.copyTo(output);

		if (img1.channels() == 3) 
		{
			cvtColor(img1, grayPre, CV_RGB2GRAY);
			cvtColor(img2, gray, CV_RGB2GRAY);
		}
		else 
		{
			grayPre = img1;
			gray = img2;
		}

		if (addnewPoints())
		{
			features[0].clear();
			goodFeaturesToTrack(grayPre, features[0], maxCount, qLevel, minDist);
		}

		calcOpticalFlowPyrLK(grayPre, gray, features[0], points[1], status, err);

		//½øÐÐÆ¥Åä
		// First, filter out the points with high error  
		vector<int> right_points_to_find_back_index;
		right_points_to_find.clear();
		int k = 0;
		for (unsigned int i = 0; i< status.size(); i++) 
		{
			if (status[i] && err[i] < 12.0) 
			{
				// Keep the original index of the point in the  
				// optical flow array, for future use  
				right_points_to_find_back_index.push_back(i);
				// Keep the feature point itself  
				right_points_to_find.push_back(points[1][i]);
			}
			else 
			{
				status[i] = 0; // a bad flow  
			}
		}

		// for each right_point see which detected feature it belongs to  
		Mat right_points_to_find_flat = Mat(right_points_to_find).reshape(1, right_points_to_find.size()); //flatten array  

		Mat right_features_flat = Mat(points[1]).reshape(1, points[1].size());

		// Look around each OF point in the right image  
		// for any features that were detected in its area  
		// and make a match.  
		BFMatcher matcher(CV_L2);
		vector<vector<DMatch>>  nearest_neighbors;
		matcher.radiusMatch(right_points_to_find_flat, right_features_flat, nearest_neighbors, 2.0f);
		// Check that the found neighbors are unique (throw away neighbors  
		// that are too close together, as they may be confusing)
		matches.clear();
		set<int> found_in_right_points; // for duplicate prevention  
		for (int i = 0; i<nearest_neighbors.size(); i++) 
		{
			DMatch _m;
			if (nearest_neighbors[i].size() == 1) 
			{
				_m = nearest_neighbors[i][0]; // only one neighbor  
			}
			else if (nearest_neighbors[i].size()>1) 
			{
				// 2 neighbors ¨C check how close they are  
				double ratio = nearest_neighbors[i][0].distance / nearest_neighbors[i][1].distance;
				if (ratio < 0.7)
				{ // not too close  
					// take the closest (first) one  
					_m = nearest_neighbors[i][0];
				}
				else 
				{ // too close ¨C we cannot tell which is better  
					continue; // did not pass ratio test ¨C throw away  
				}
			}
			else 
			{
				continue; // no neighbors... :(  
			}
			// prevent duplicates  
			if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.end())
			{
				// The found neighbor was not yet used:  
				// We should match it with the original indexing   
				// ofthe left point  
				_m.queryIdx = right_points_to_find_back_index[_m.queryIdx];
				matches.push_back(_m); // add this match  
				found_in_right_points.insert(_m.trainIdx);
			}
		}

		showtrackedPoints(output);

		std::swap(points[1], features[0]);
	}
};
#endif