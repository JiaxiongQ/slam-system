#include "Video.h"

void ReadOriginalVideo(char* name,vector<cv::Mat> &frames,int &fps){
	cv::VideoCapture pCapture;
	pCapture.open(name);
	
	fps =  (int)pCapture.get(CV_CAP_PROP_FPS);
	printf("%d\n",fps);
	int video_width = pCapture.get(CV_CAP_PROP_FRAME_WIDTH);
	int video_height = pCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int numFrames = pCapture.get(CV_CAP_PROP_FRAME_COUNT)-1;

	if(numFrames<0){
	   printf("Can not decode video!\n");
	   exit(0);
	}

	cv::Mat frame;
	//pCapture >> frame;  // skip first frame
	vector<cv::Mat> subVideo;
	int cc=0;
	int gap = (numFrames/100)+1;
	pCapture >> frame;
	while(frame.data){
	/*
	if(cc%gap==0)
			cout<<cc/gap<<endl;
		cc++;
	*/
		cv::Mat temp = cv::Mat::zeros(frame.rows,frame.cols,CV_8UC3);
		frame.copyTo(temp);
		frames.push_back(temp);
		pCapture >> frame;
	}
	pCapture.release();
}


void VideoRecorder(char* outName){
	cv::VideoCapture capture;
	cv::VideoWriter writer;
	capture.open(0);

	int fps = (int)capture.get(CV_CAP_PROP_FPS);
    int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	
	

	cv::namedWindow("");
	
	cv::Mat frame;
	vector<cv::Mat> frames;
	
	while(true){
	
		capture>>frame;
		cv::imshow("",frame);
		
		frames.push_back(frame.clone());
		int key = cvWaitKey(1);

		if(key == 'q' || key == 'Q') break;
	}


	writer.open(outName,CV_FOURCC('x','v','i','d'),30,cv::Size(width,height));  
	for(int i=0;i<frames.size();i++){
	    writer<<frames[i];
	}
	
}
