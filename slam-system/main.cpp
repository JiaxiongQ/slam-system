// slamA.cpp : Defines the entry point for the console application.
//
#include "include/myCeres.h"
#include "include/myEigen.h"
#include "include/Video.h"
#include "include/reconstruct.h"
#include "include/define.h"
#include "include/gridTracker.h"
#include "include/Viewer.h"
#include "include/featureTracker.h"

int NC;
int NP;
vector<Mat> motions_Show;
vector<Mat> rotations_Show;
bool startFlag = false;
vector<Mat> imgs;//帧序列
vector< vector<Point2f> >points2_for_show;//特征点集
int NCAMS = 140;//处理的帧数量 top:66:990/5/s 74:584/7/s 76:150/7/h/u 90:125/5/s 94:140/5/s 96:311/7/h 98:195/5/h tum:150/7/h 
sem_t sem;
int step = 5;
int widSize = 10;
int main(int argc,char** argv)
{
		
	//intrinsic
	/*
	Mat K(Matx33d(
		535.4, 0, 320.1,
		0, 539.2, 247.6,
		0, 0, 1));
	*/
	
	Mat K(Matx33d(
		1503.6, 0, 1214.8,
		0, 1497.2, 990.7997,
		0, 0, 1));
	
	//guofeng zhang
	/*
	Mat K(Matx33d(
		1139.3382050, 0, 479.5000000,
		0, 1139.3382050, 269.5000000,
		0, 0, 1));
        */
	//TUM
	/*
	Mat K(Matx33d(
		517.3, 0, 318.6,
		0, 516.5, 255.3,
		0, 0, 1));
	*/
	/*
	Mat K(Matx33d(
		535.4, 0, 320.1,
		0, 539.2, 247.6,
		0, 0, 1));
	*/
	
	//ICL-NUIM
	/*
	Mat K(Matx33d(
		481.20, 0, 319.5,
		0, -480, 239.5,
		0, 0, 1));
	*/
	
	const int64 start = getTickCount();
	int fps;//帧率
	ReadOriginalVideo("..//video//GOPR0494.avi", imgs, fps);//读取视频 GOPR0496.avi rgbd_dataset_freiburg3_structure_texture_near-rgb.avi
	cout << imgs.size() << endl;
	
	/*
	string wName = "//root//projects//slam//keyboard//";
	for (int i = 0; i <= NCAMS; ++i)
	{
		string sOut; 
		if (i < 10)
			sOut = wName + "00" + to_string(i) + ".jpg";
		else if (i < 100)
			sOut = wName + "0" + to_string(i) + ".jpg";
		else 
			sOut = wName + to_string(i) + ".jpg";

		imwrite(sOut, imgs[i]);
	}
	*/
	
	//tracking
	gridTracker gt;
	gt.trackerInit(imgs[0]);
        for (int i = 1; i < NCAMS + 1; i++)
	{
	    gt.Update(imgs[i-1],imgs[i], i);
	}
	
	vector< vector<Node*> > srcNodes = gt.FeasList_src;//track到的特征点
	
	reconstructure* slam = new reconstructure();
	slam->NCAMS = NCAMS;
	slam->K = K;
	slam->srcNodes = srcNodes;
	slam->imgs = imgs;
	slam->step = step;
	slam->widSize = widSize;
	std::thread* mySlam = new thread(&reconstructure::run,slam);
	
	View3D(argc,argv);
	double duration = (getTickCount() - start) / getTickFrequency();
        cout << "Consuming Time(s) : " << duration << endl;
	
	mySlam->join();
	
	return 0;

}
