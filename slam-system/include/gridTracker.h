
#include"MycalcopticalFlow.h"

#define VARIANCE 4
#define MAXTRACKS 1100//2000
#define GRIDSIZE 5
#define MASK_RADIUS 8//15
#define ISRANSAC 1

using namespace std;
using namespace cv;


#ifndef __GRIDTRACKER__
#define __GRIDTRACKER__


inline bool rule(const KeyPoint& p1, const KeyPoint& p2) 
{
  return p1.response > p2.response;
}

typedef struct Node
{
	cv::Point2f Pts;
	int frame;
	int tab;
	bool flag;
	int trackIndex;
	Node*front, *next;

	Node(){
		frame = -1;
		Pts.x = Pts.y = 0; 
		tab = -1; 
		flag = false;
		trackIndex = -1;
		front = next = NULL;
	}

}Node;


class gridTracker
{
private:
	int m_Block_X = 2;
	int m_Block_Y = 2;
	int m_ImgHight;
	int m_ImgWidth;
	int m_BlockSizeX;
	int m_BlockSizeY;
	int m_StepSizeX;
	int m_StepSizeY;

	vector< vector<Node*> >m_sourePts;
	vector< vector<Node*> >m_inliersPts;
	 
	void rejectOutliers(vector<Node*>FrontNodes, vector<Node*>&NextNodes, bool flag = ISRANSAC);

	void rejectOutliers2(vector<Node*>FrontNodes, vector<Node*>&NextNodes, bool flag = ISRANSAC);

  public:

	  vector< vector<Node*> > FeasList_src;//
	  vector<Node*>List_src;
	  vector<Node*>List_src0;

	  void initial(Node*&head);
	  void listTrack(Node*&head, cv::Point2f pts, int frame, Node*frontNode, int tab);

	  void PushImgSize(int rows, int cols);

	  vector< vector<Node*> > getInlierPts(){ m_inliersPts = FeasList_src;  return m_inliersPts; }

	  void HomographyInliers(vector<Node*>&FrontNode, vector<Node*>&NextNode, bool flag = ISRANSAC);

    
	  vector< vector<cv::Point2f> >PtsPosition;
	  vector<cv::Point2f>Position;
	
    //mask used for ignoring regions of the image in the detector and for maintaining minimal feature distance
    cv::Mat curMask; 
    
    //tracked feas of current frame
    vector<cv::Point2f> points1;
    vector<cv::Point2f> trackedFeas;//matched Feas of currFrame
    
    //all feas of current frame
    vector<cv::Point2f> allFeas;
	Mat showOut;
	vector<cv::Point2f> preFeas;//matched Feas of preFrame
	vector<int>allindex;
    
    //num of feas from last frame
    int numActiveTracks;
    
    int TRACKING_HSIZE, LK_PYRAMID_LEVEL, MAX_ITER, fealimitGrid;
    
    double ACCURACY, LAMBDA;
    
    float usableFrac;
    
    //store image pyramid for re-utilization
    std::vector<cv::Mat> prevPyr;
    
	//int overflow;
    int MaxTracks;

	//grids devision
    cv::Point2i hgrids;
    
    //records feature number of each grids
    vector<int> feanumofGrid;
    
    double minAddFrac, minToAdd;
    
    int unusedRoom, gridsHungry;
    
    vector<int> lastNumDetectedGridFeatures;
    
    //feature detector for Fast feature dtection one each gridCols
	vector<Ptr< FeatureDetector> > detector;
	
    //thresholds for Fast detection on each grid
    vector<int> hthresholds;
    
    double DETECT_GAIN;
    
    //constructor
	gridTracker();
	
	bool maskPoint(float x, float y);
       
	bool Update(cv::Mat& im0, cv::Mat& im1, int index);
    
    bool trackerInit(cv::Mat& im);
};


std::vector<cv::Mat> FastTracker(vector<cv::Mat> &frames);

#endif
