#include "reconstruct.h"
#define eThred 1.0//default:1.0
#define pnpThred 8.0//default:8.0
#define PI 3.1415926535897932384626433832795
#define curWeight 1.0
#define maxNorm 0.3
#define minInliers 5

vector<DMatch> featureMatcher(vector<Point2f>& trackedPoints, vector<Point2f>& detectedPoints,vector<int>& right_points_to_find_back_index)
{
  vector<DMatch> matches;
		// for each right_point see which detected feature it belongs to  
		Mat right_points_to_find_flat = Mat(trackedPoints).reshape(1, trackedPoints.size()); //flatten array  

		Mat right_features_flat = Mat(detectedPoints).reshape(1,detectedPoints.size());

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
				// 2 neighbors – check how close they are  
				double ratio = nearest_neighbors[i][0].distance / nearest_neighbors[i][1].distance;
				if (ratio < 0.7)
				{ // not too close  
					// take the closest (first) one  
					_m = nearest_neighbors[i][0];
				}
				else 
				{ // too close – we cannot tell which is better  
					continue; // did not pass ratio test – throw away  
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
  return matches;
}

void reconstructure::init()
{
	  points2_for_all.clear();
	  matches_for_all.clear();
	  colors_for_all.clear();
	  points2_for_all.resize(widSize);
	  matches_for_all.resize(widSize - 1);
	  colors_for_all.resize(widSize);
	
	  for (int i = 0; i < widSize; ++i)
	  {
		points2_for_all[i].resize(srcNodes[i].size(), Point2f(-1, -1));
		colors_for_all[i].resize(srcNodes[i].size(),Vec3b(0,0,0));
	  }
	  /*
	  for(int i = 0;i < widSize; ++i)
	{
	  for (int j = 0; j < srcNodes[i].size(); ++j)
	  {
		Node* tempNode;
		tempNode = srcNodes[i][j];
		if(tempNode->next != NULL)
		{
		  points2_for_all[i][tempNode->tab] = tempNode->Pts;
		  colors_for_all[i][tempNode->tab] = imgs[i].at<Vec3b>(tempNode->Pts.y,tempNode->Pts.x);
		  if (i < widSize - 1)
		  {
			DMatch match;
			match.queryIdx = tempNode->tab;
			match.trainIdx = tempNode->next->tab;
			matches_for_all[i].push_back(match);
		  }
		}
	  }
	}
	*/
	
	for (int j = 0; j < srcNodes[0].size(); ++j)
	  {
		int temp1 = widSize;
		Node* tempNode = srcNodes[0][j];
		while (temp1--)
		{
			if (tempNode->next != NULL)
			{
				tempNode = tempNode->next;

			}
			else
			{
				break;
			}
		}

		if (temp1 <= 0)
		{
			tempNode = srcNodes[0][j];
			for (int i = 0; i < widSize; ++i)
			{
				DMatch match;
				
				points2_for_all[i][tempNode->tab] = tempNode->Pts;
				colors_for_all[i][tempNode->tab] = imgs[i].at<Vec3b>(tempNode->Pts.y,tempNode->Pts.x);
				if (i < widSize - 1)
				{
					match.queryIdx = tempNode->tab;
					match.trainIdx = tempNode->next->tab;
					matches_for_all[i].push_back(match);
				}
				tempNode = tempNode->next;
			}
		}
	    }
	    
	//将track到的有效特征点显示出来
	for (int i = 0; i < widSize; ++i)
	{
	  for (int j = 0; j < srcNodes[i].size(); ++j)
	  {
	    if (points2_for_all[i][j] != Point2f(-1, -1))
	    {
	      circle(imgs[i], points2_for_all[i][j], 3, Scalar(0, 255, 0), -1);
	    }
	  }
	}
      
}

void reconstructure::initNext(int blockNum)
{
	  points2_for_all.clear();
	  matches_for_all.clear();
	  colors_for_all.clear();
	  points2_for_all.resize(widSize);
	  matches_for_all.resize(widSize - 1);
	  colors_for_all.resize(widSize);
	
	  for (int i = 0; i < widSize; ++i)
	  {
		points2_for_all[i].resize(srcNodes[i + blockNum*step].size(), Point2f(-1, -1));
		colors_for_all[i].resize(srcNodes[i + blockNum*step].size(),Vec3b(0,0,0));
	  }
	  /*
	    for(int i = 0;i < widSize; ++i)
	{
	  for (int j = 0; j < srcNodes[i + blockNum*step].size(); ++j)
	  {
		Node* tempNode;
		tempNode = srcNodes[i + blockNum*step][j];
		if(tempNode->next != NULL)
		{
		  points2_for_all[i][tempNode->tab] = tempNode->Pts;
		  colors_for_all[i][tempNode->tab] = imgs[i + blockNum*step].at<Vec3b>(tempNode->Pts.y,tempNode->Pts.x);
		  if (i < widSize - 1)
		  {
			DMatch match;
			match.queryIdx = tempNode->tab;
			match.trainIdx = tempNode->next->tab;
			matches_for_all[i].push_back(match);
		  }
		}
	  }
	}
	*/
      
	  for (int j = 0; j < srcNodes[blockNum*step].size(); ++j)
	  {
		int temp1 = widSize;
		Node* tempNode = srcNodes[blockNum*step][j];
		while (temp1--)
		{
			if (tempNode->next != NULL)
			{
				tempNode = tempNode->next;

			}
			else
			{
				break;
			}
		}

		if (temp1 <= 0)
		{
			tempNode = srcNodes[blockNum*step][j];
			for (int i = 0; i < widSize; ++i)
			{
				DMatch match;
				
				points2_for_all[i][tempNode->tab] = tempNode->Pts;
				colors_for_all[i][tempNode->tab] = imgs[i + blockNum*step].at<Vec3b>(tempNode->Pts.y,tempNode->Pts.x);
				if (i < widSize - 1)
				{
					match.queryIdx = tempNode->tab;
					match.trainIdx = tempNode->next->tab;
					matches_for_all[i].push_back(match);
				}
				tempNode = tempNode->next;
			}
		}
	    }
	
	//将track到的有效特征点显示出来
	for (int i = widSize - step; i < widSize; ++i)
	{
	  for (int j = 0; j < srcNodes[i].size(); ++j)
	  {
	    if (points2_for_all[i][j] != Point2f(-1, -1))
	    {
	      circle(imgs[i + blockNum*step], points2_for_all[i][j], 3, Scalar(0, 255, 0), -1);
	    }
	  }
	}
	  
	  rotations.clear();
	  motions.clear();
	  rotations.push_back(rotation_before);
	  motions.push_back(motion_before);
		
	  //将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
	  correspond_struct_index.clear();
	  correspond_struct_index.resize(widSize);
	  for (int i = 0; i < widSize; ++i)
	  {
	    correspond_struct_index[i].resize(points2_for_all[i].size(), -1);
	  }
	  
	  int index = 0;
	  structure.clear();
	  colors.clear();
	  for(int i = 0;i < matches_before.size();++i)
	  {
	    int query_idx = matches_before[i].queryIdx;
	    int struct_idx = correspond_struct_index_before[query_idx];
    
	    if(struct_idx < 0)
	      continue;
    
	    if(fabs(points2_for_all[0][query_idx].x - points2_before[query_idx].x) < 0.001 && fabs(points2_for_all[0][query_idx].y - points2_before[query_idx].y) < 0.001)
	    {
	      structure.push_back(structure_before[struct_idx]);
	      colors.push_back(colors_before[struct_idx]);
	      correspond_struct_index[0][query_idx] = index;
	      ++index;
	    }
	  }
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	/*
	Mat F = findFundamentalMat(p1, p2, FM_RANSAC, 3.0, 0.99, mask);
	Mat_<double> E = K.t() * F * K;
	SVD svd(E);
	Matx33d W(0, -1, 0,
	1, 0, 0,
	0, 0, 1);
	R = svd.u * Mat(W) * svd.vt;
	T = svd.u.col(2); 
	*/

	//根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, eThred, mask);
	if (E.empty())
		return false;

	double feasible_count = countNonZero(mask);

	//对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
		return false;

	//分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count*1.0) / feasible_count < 0.7)
		return false;

	return true;

}

void reconstruct(Mat& K, Mat& R0, Mat& T0, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2,
	vector<Point3d>& structure)
{

	//两个相机的投影矩阵[R T]，triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R0.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);//!!!
	T0.convertTo(proj1.col(3), CV_32FC1);//!!!

	R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK*proj1;
	proj2 = fK*proj2;

	//三角重建
	Mat s;
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<double> col = s.col(i);
		col /= col(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		structure.push_back(Point3d(col(0), col(1), col(2)));
	}
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void get_matched_points(vector<Point2f>& p1, vector<Point2f>& p2, vector<DMatch> matches,
	vector<Point2f>& out_p1, vector<Point2f>& out_p2)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx]);
		out_p2.push_back(p2[matches[i].trainIdx]);
	}
}

void get_matched_colors(vector<Vec3b>& c1, vector<Vec3b>& c2, vector<DMatch> matches,
	vector<Vec3b>& out_c1,vector<Vec3b>& out_c2)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void reconstructure::init_structure()
{
	//计算变换矩阵
	vector<Point2f> p1, p2;
	vector<Vec3b>  c2;
	Mat R, T;	//旋转矩阵和平移向量
	Mat mask;	//mask中大于零的点代表匹配点，等于零代表失配点
	get_matched_points(points2_for_all[0], points2_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
	
	firstFlag = find_transform(K, p1, p2, R, T, mask);
	cout << endl;
	if(firstFlag)
	  cout << "good start!" << endl;
	else 
	  cout << "start wrong!" << endl;
	cout << endl;
	
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);
	//三维重建
 
	Mat R0 = rotations[0];
	Mat T0 = motions[0];
      
	reconstruct(K, R0, T0, R, T, p1, p2, structure);
	//保存变换矩阵
	rotations.push_back(R);
	motions.push_back(T);

	//将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
	correspond_struct_index.clear();
	correspond_struct_index.resize(points2_for_all.size());
	for (int i = 0; i < points2_for_all.size(); ++i)
	{
	  correspond_struct_index[i].resize(points2_for_all[i].size(), -1);
	}

	//填写头两幅图像的结构索引
	vector<DMatch>& matches = matches_for_all[0];
	int index = 0;
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;

		correspond_struct_index[0][matches[i].queryIdx] = index;
		correspond_struct_index[1][matches[i].trainIdx] = index;
		++index;
	}
}

void get_objpoints_and_imgpoints(vector<DMatch>& matches, vector<int>& struct_index,
	vector<Point3d>& structure,
	vector<Point2f>& key_points,
	vector<Point3d>& object_points,
	vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_index[query_idx];
		if (struct_idx < 0)
			continue;

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx]);
	}

}

void fusion_structure(vector<DMatch>& matches, vector<int>& struct_index, vector<int>& next_struct_index,
	vector<Point3d>& structure, vector<Point3d>& next_structure, vector<Vec3b>& colors, vector<Vec3b>& next_colors)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_index[query_idx];
		if (struct_idx >= 0) //若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
		{
			next_struct_index[train_idx] = struct_idx;
			continue;
		}

		//若该点在空间中不存在，将该点加入到结构中，且这对匹配点的空间点索引都为新加入的点的索引
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_index[query_idx] = next_struct_index[train_idx] = structure.size() - 1;
	}
}



double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
     return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

void reconstructure::run()
{
	int blockNum = 0;
	Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	vector<int> structureNum;
	structureNum.push_back(0);
	int start = 1;
	
	init();
	
	rotations.clear();
	motions.clear();
	Mat R0;
	R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	rotations.push_back(R0);
	motions.push_back(T0);
	
	//初始化索引数组以及求第1帧和第2帧的三维点以及旋转和平移矩阵
	init_structure();
        Mat showT;
	Mat invR;
	int strucNum = 0;
	NC = 0;
	while(blockNum*step +widSize<=NCAMS)
	{
	      cout << blockNum*step +widSize << endl;
	      if(blockNum != 0)
	      {
		start = 0;
		initNext(blockNum);
		 strucNum = structure.size(); 
		  std::unique_lock<std::mutex> lck (mtx2);
		  invert(rotations[0], invR, DECOMP_SVD);
		  showT = -invR*motions[0];
		  
		  transpose(showT, showT);
		  rotations_Show.push_back(rotations[0]);   
		  motions_Show.push_back(showT);
		  NC++;
		  startFlag = true;
	      }
	      else 
	      {
		for(int i = 0;i < 2;++i)
		{
		  std::unique_lock<std::mutex> lck (mtx3);
	
		  invert(rotations[i], invR, DECOMP_SVD);
		  showT = -invR*motions[i];
		  
		  transpose(showT, showT);
		  rotations_Show.push_back(rotations[i]);   
		  motions_Show.push_back(showT);
		  NC = i;
		  startFlag = true;
		}
	      }
	      
	      int startS = 0;	
	      //增量重建剩余n-2帧图像
	      for (int i = start; i < widSize - 1; ++i)
	      {
		std::unique_lock<std::mutex> lck (mtx4);
		
		Mat r, R, T, inliers;

		//获取第i帧图像中匹配点对应的三维点，以及在第i+1帧图像中对应的像素点
		get_objpoints_and_imgpoints(matches_for_all[i], correspond_struct_index[i], structure, points2_for_all[i + 1], object_points, image_points);

		//求解变换矩阵
		solvePnPRansac(object_points, image_points, K, noArray(), r, T, false, 100, pnpThred, 0.99, inliers);
		cout << "inliers = " << inliers.rows << endl;
		double norm = normofTransform(r, T);
		cout<<"norm = "<<norm<<endl;
		
		//将旋转向量转换为旋转矩阵
		Rodrigues(r, R);
		//保存变换矩阵
		rotations.push_back(R);
		motions.push_back(T);	
		
		 invert(rotations[i + 1], invR, DECOMP_SVD);
		 showT = -invR*motions[i + 1];
		 transpose(showT, showT);
		if(blockNum==0)
		{
		 rotations_Show.push_back(rotations[i + 1]);   
		 motions_Show.push_back(showT);
		 NC = i + 1;
		}
		else
		{
		  if(i >= widSize - step)
		  {
		      rotations_Show.push_back(rotations[i + 1]);
		      motions_Show.push_back(showT);
		      NC++;
		  }
		  else
		  {
		  }
		  
		}
		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(points2_for_all[i], points2_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		//根据之前求得的R，T进行三维重建
		vector<Point3d> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);
		
		//将新的重建结果与之前的融合
		fusion_structure(matches_for_all[i], correspond_struct_index[i], correspond_struct_index[i + 1], structure, next_structure, colors, c1);
		
		if(i == widSize - 2)
		{
		 vector<Point3d> strucTemp(structure.size());
		 vector<Point3d> structs;
		 vector<Vec3b> colorsT;
		 for(int i = 0;i < structure.size();++i) 
		    strucTemp[i] = structure[i] * 10.0;
		  
		 for(int j = strucNum;j < strucTemp.size();++j)
		 {
		      structs.push_back(strucTemp[j]);
		      colorsT.push_back(colors[j]);
		  }
		  
		  structures.push_back(structs);
		  allcolors.push_back(colorsT);
		
		  vPointCloud.resize(structures.size());
		  for(int k = 0;k < structures.size();++k)
		  {
		    vPointCloud[k].resize(structures[k].size());
		  }
		  for(int k = 0;k < structures.size();++k)
		  {
		  for(int j = 0;j < structures[k].size();++j)
		  {
		    vPointCloud[k][j].X = structures[k][j].x;
		    vPointCloud[k][j].Y = structures[k][j].y;
		    vPointCloud[k][j].Z = structures[k][j].z;
		    vPointCloud[k][j].R = allcolors[k][j][2] * 1.0 / 255;
		    vPointCloud[k][j].G = allcolors[k][j][1] * 1.0 / 255;
		    vPointCloud[k][j].B = allcolors[k][j][0] * 1.0 / 255;
		  }
		  }
		  NP = structures.size();
		}
		startFlag = true;
	       }
	       
	        cout << structure.size() << endl; 
		structureNum.push_back(structure.size());
		
		structure_before = structure;
		colors_before = colors;
		correspond_struct_index_before = correspond_struct_index[step];
		points2_before = points2_for_all[step];
		matches_before = matches_for_all[step];
		
		extrinsics.clear(); 
		//BA
		for (int j = 0; j < widSize; ++j)
		{
		      cv::Mat extrinsic(6, 1, CV_64FC1);
		      cv::Mat r;
		      Rodrigues(rotations[j], r);

		      r.copyTo(extrinsic.rowRange(0, 3));
		      motions[j].copyTo(extrinsic.rowRange(3, 6));

		      extrinsics.push_back(extrinsic);
		}
		
		vector<double> blockError = Bundle_Adjustment(intrinsic, extrinsics, correspond_struct_index, points2_for_all,structure,true);
		rpjError.push_back(blockError);
		
		for (int j = 0; j < widSize; ++j)
		{
		    cv::Mat r;
		    r = extrinsics[j].rowRange(0, 3);
		    Rodrigues(r, rotations[j]);
		    motions[j] = extrinsics[j].rowRange(3, 6);
		}
		
		motion_before = motions[step];
		rotation_before = rotations[step];
	      
		motions_Show.resize(widSize + (blockNum) * step);
		rotations_Show.resize(widSize + (blockNum) * step);
		for (int j = 0; j < widSize; ++j)
		{
		    invert(rotations[j], invR, DECOMP_SVD);
		    showT = -invR*motions[j];
		    
		    transpose(showT, showT);
		    if(blockNum == 0)
		    {
		      rotations_Show[j] = rotations[j];
		      motions_Show[j] = showT;
		    }
		    else 
		    {
		      if(j < widSize - step)
		      {
			rotations_Show[j + (blockNum) * step] = rotations[j] * curWeight + rotations_Show[j +(blockNum) * step] * (1 - curWeight);
			motions_Show[j + (blockNum) * step] = showT * curWeight + motions_Show[j + (blockNum) * step] * (1 - curWeight);
		      }
		      else
		      {
			rotations_Show[j + (blockNum) * step] = rotations[j];
			motions_Show[j + (blockNum) * step] = showT;
		      }
		    }
		}
		/*
	        for(int j = 0;j < structure.size() - strucNum;++j)
		{
		  structures[NP-1][j] = structure[j] * 10.0;
		}
		*/
		++blockNum;
	}
  //showError();
  startFlag=false;
}


void reconstructure::showError()
{
	ofstream outfile;
	outfile.open("reprojError96.txt");
	outfile << "initial   after" << endl;
	for (int i = 0; i < rpjError.size(); ++i)
	{
		char *ptrEI = new char[10];
		ptrEI = gcvt(rpjError[i][0], 3, ptrEI);
		outfile << ptrEI << "     ";
		char *ptrEA = new char[10];
		ptrEA = gcvt(rpjError[i][1], 3, ptrEA);
		outfile << ptrEA << "     ";
		outfile << endl;
	}
	
	cout << "camera reprojectError saved..." << endl;
}
