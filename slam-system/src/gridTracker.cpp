#include "gridTracker.h"

gridTracker::gridTracker(){

}

void gridTracker::PushImgSize(int rows, int cols){
	m_ImgHight = rows;
	m_ImgWidth = cols;
	m_BlockSizeX = cols / m_Block_X;
	m_BlockSizeY = rows / m_Block_Y;
	m_StepSizeX = m_BlockSizeX / 2;
	m_StepSizeY = m_BlockSizeY / 2;
}

void gridTracker::initial(Node*&head){
	head = new Node;
	head->Pts = cv::Point2f(0, 0);
	head->frame = -1;
	head->front = NULL;
	head->next = NULL;
	head->tab = 0;
}

void gridTracker::listTrack(Node*&head, cv::Point2f pts, int frame, Node*frontNode,int tab){
	head = new Node;
	head->Pts = pts;
	head->frame = frame;
	head->front = frontNode;
	head->next = NULL;
	head->tab = tab;
}

void gridTracker::HomographyInliers(vector<Node*>&FrontNode, vector<Node*>&NextNode, bool flag){
	if (flag){
		if (NextNode.size() > 10){
			int tab = 0;
			vector<Node*>NextNodeInliers;
			int n1 = 0, n2 = 0;
			int n3 = 0;
			vector<cv::Point2f>points1, points2;
			vector<int>indexF;
			for (int i = 0; i < FrontNode.size(); i++){
				if (FrontNode[i]->next){
					n1++;
					points1.push_back(FrontNode[i]->Pts);
					indexF.push_back(n3);
				}
				n3++;
			}
			for (int i = 0; i < NextNode.size(); i++){
				n2++;
				points2.push_back(NextNode[i]->Pts);
			}
			//printf("%d\n%d\n", n1, n2);
			if (n1 != n2){
				cout << "error!!!!\n" << std::endl;
			}
			cv::Mat Mask;
			cv::Mat homography = cv::findHomography(points1, points2, Mask, CV_RANSAC);

			NextNodeInliers.resize(0);
			for (int i = 0; i < Mask.rows; i++){
				int PtsFlag = Mask.at<uchar>(i, 0);
				if (PtsFlag){
					NextNode[i]->tab = tab++;
					NextNode[i]->front = FrontNode[indexF[i]];
					FrontNode[indexF[i]]->next = NextNode[i];

					NextNodeInliers.push_back(NextNode[i]);
				}
				else{
					FrontNode[indexF[i]]->next = NULL;
				}
			}
			NextNode.resize(0);
			for (int i = 0; i < NextNodeInliers.size(); i++){
				NextNode.push_back(NextNodeInliers[i]);
			}
		}
	}
}


void gridTracker::rejectOutliers(vector<Node*>FrontNodes, vector<Node*>&NextNodes, bool flag){
	if (flag){
		int nx = (m_ImgWidth - m_BlockSizeX) / m_StepSizeX + 1;//窗口在X方向滑动的次数
		int ny = (m_ImgHight - m_BlockSizeY) / m_StepSizeY + 1;//窗口在Y方向滑动的次数


		cv::Mat H; vector<uchar>Mask;
		vector<cv::Point2f>SourcePts, TargetPts;
		int size = NextNodes.size();
		vector<int>NextNodsFlag(size, 0);//记录每个点的flag，判断是否符合homography的约束
		vector<int>NextNodeInlierNum(size, 0);
		vector<int>NextIndex;
		for (int y_direct = 0; y_direct < ny; y_direct++){
			for (int x_direct = 0; x_direct < nx; x_direct++){
				SourcePts.clear(); TargetPts.clear();	NextIndex.clear();
				for (int i = 0; i < size; i++){
					int Next_x = NextNodes[i]->Pts.x; int Next_y = NextNodes[i]->Pts.y;
					if (Next_x >= x_direct*m_StepSizeX&&Next_x < (x_direct*m_StepSizeX + m_BlockSizeX)
						&& Next_y >= y_direct*m_StepSizeY&&Next_y < (y_direct*m_StepSizeY + m_BlockSizeY)){
						SourcePts.push_back(NextNodes[i]->front->Pts);
						TargetPts.push_back(NextNodes[i]->Pts);
						NextIndex.push_back(i);//记录每个点的标号
					}
				}
				if (NextIndex.size()>20){//点的数量至少10才进行homography的计算
					H = cv::findHomography(SourcePts, TargetPts, Mask, CV_RANSAC);
					for (int i = 0; i < Mask.size(); i++){
						if (Mask[i] == 1 || NextNodsFlag[NextIndex[i]] == 1)//当前block的homography的inlier的判断 || 有一个homography满足就认为符合条件
							NextNodsFlag[NextIndex[i]] = 1;
					}
				}
			}
		}

		vector<Node*>inlierNodes;
		inlierNodes.clear();
		int tag = 0;
		for (int i = 0; i < size; i++){
			if (NextNodsFlag[i] == 1){
				NextNodes[i]->tab = tag++;
				inlierNodes.push_back(NextNodes[i]);
			}
			else{
				NextNodes[i]->front->next = NULL;
			}
		}
		NextNodes.resize(inlierNodes.size());
		for (int i = 0; i < inlierNodes.size(); i++)
			NextNodes[i] = inlierNodes[i];
	}
}

void gridTracker::rejectOutliers2(vector<Node*>FrontNodes, vector<Node*>&NextNodes, bool flag){

	if (flag){
		int nx = m_Block_X;//窗口在X方向滑动的次数
		int ny = m_Block_Y;//窗口在Y方向滑动的次数


		cv::Mat H; vector<uchar>Mask;
		vector<cv::Point2f>SourcePts, TargetPts;
		int size = NextNodes.size();
		vector<int>NextNodsFlag(size, 0);//记录每个点的flag，判断是否符合homography的约束
		vector<int>NextIndex;
		for (int y_direct = 0; y_direct < ny; y_direct++){
			for (int x_direct = 0; x_direct < nx; x_direct++){
				SourcePts.clear(); TargetPts.clear();	NextIndex.clear();
				for (int i = 0; i < size; i++){
					int Next_x = NextNodes[i]->Pts.x; int Next_y = NextNodes[i]->Pts.y;
					if (Next_x >= x_direct*m_BlockSizeX&&Next_x < (x_direct*m_BlockSizeX + m_BlockSizeX)
						&& Next_y >= y_direct*m_BlockSizeY&&Next_y < (y_direct*m_BlockSizeY + m_BlockSizeY)){
						SourcePts.push_back(NextNodes[i]->front->Pts);
						TargetPts.push_back(NextNodes[i]->Pts);
						NextIndex.push_back(i);//记录每个点的标号
					}
				}
				if (NextIndex.size()>20){//点的数量至少10才进行homography的计算
					H = cv::findHomography(SourcePts, TargetPts, Mask, CV_RANSAC,1.8);
					for (int i = 0; i < Mask.size(); i++){
						if (Mask[i] == 1)//当前block的homography的inlier的判断 || 有一个homography满足就认为符合条件
							NextNodsFlag[NextIndex[i]] = 1;
					}
				}
			}
		}

		vector<Node*>inlierNodes;
		inlierNodes.clear();
		int tag = 0;
		for (int i = 0; i < size; i++){
			if (NextNodsFlag[i] == 1){
				NextNodes[i]->tab = tag++;
				inlierNodes.push_back(NextNodes[i]);
			}
			else{
				NextNodes[i]->front->next = NULL;
			}
		}
		NextNodes.resize(inlierNodes.size());
		for (int i = 0; i < inlierNodes.size(); i++)
			NextNodes[i] = inlierNodes[i];
	}
}



bool gridTracker::maskPoint(float x, float y)
{
	if (curMask.at<unsigned char>(cv::Point(x, y)) == 0)// 0 indicates that this pixel is in the mask, thus is not useable for new features of OF results
		return 1;// means that this feature should be killed
	cv::rectangle(curMask, cv::Point(int(x - MASK_RADIUS / 2 + .5), int(y - MASK_RADIUS / 2 + .5)), cv::Point(int(x + MASK_RADIUS / 2 + .5), int(y + MASK_RADIUS / 2 + .5)), cv::Scalar(0), -1);//define a new image patch
	return 0;// means that this feature can be retained
};

bool gridTracker::trackerInit(cv::Mat& im)
{
	//curMask = Mat(im.rows, im.cols, CV_8UC1, Scalar(255));

	PushImgSize(im.rows, im.cols);

	curMask.create(im.rows, im.cols, CV_8UC1);
	for (int i = 0; i<im.rows; i++){
		for (int j = 0; j<im.cols; j++){
			curMask.at<uchar>(i, j) = 255;
		}
	}

	numActiveTracks = 0;
	TRACKING_HSIZE = 8;
	LK_PYRAMID_LEVEL = 4;
	MAX_ITER = 10;
	ACCURACY = 0.1;
	LAMBDA = 0.0;

	//overflow = 0;

	hgrids.x = GRIDSIZE;
	hgrids.y = GRIDSIZE;

	usableFrac = 0.02;

	MaxTracks = MAXTRACKS;

	minAddFrac = 0.1;
	minToAdd = minAddFrac * MaxTracks;

	//upper limit of features of each grid
	fealimitGrid = floor((double)MaxTracks / (double)(hgrids.x*hgrids.y));

	lastNumDetectedGridFeatures.resize((hgrids.x*hgrids.y), 0);

	DETECT_GAIN = 10;

	for (int i = 0; i<(hgrids.x*hgrids.y); i++)
	{
//		hthresholds.push_back(20);

//		Ptr<FeatureDetector> detectorInit = FastFeatureDetector::create(hthresholds[i],true);
//		detector.push_back(detectorInit);

		feanumofGrid.push_back(0);
	}

	cv::buildOpticalFlowPyramid(im, prevPyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);

	Update(im, im, 0);

	return 1;
}
float newThresh = 20;
bool gridTracker::Update(cv::Mat& im0, cv::Mat& im1, int index)
{
	//int Num = Position.size();
	int Num = List_src.size();
	int Numfront = Num;
	//Position.resize(0);
	List_src.resize(0);

	//clear the mask to be white everywhere  
	//curMask.setTo(Scalar(255));
	for (int i = 0; i<curMask.rows; i++){
		for (int j = 0; j<curMask.cols; j++){
			curMask.at<uchar>(i, j) = 255;
		}
	}

	//num of feas from last frame
	numActiveTracks = allFeas.size();

	//do optical flow if there are feas from last frame
	if (numActiveTracks > 0)
	{
		vector<uchar> status(allFeas.size(), 1);
		vector<float> error(allFeas.size(), -1);

		//image pyramid of curr frame
		std::vector<cv::Mat> nextPyr;
		cv::buildOpticalFlowPyramid(im1, nextPyr, Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1), LK_PYRAMID_LEVEL, true);

		// perform LK tracking from OpenCV, parameters matter a lot
		points1 = allFeas;

		preFeas.clear();
		trackedFeas.clear();

		lkopticalflowt::calcOpticalFlowPyrLK(prevPyr,
			nextPyr,
			cv::Mat(allFeas),
			cv::Mat(points1),
			cv::Mat(status),// '1' indicates successfull OF from points0
			cv::Mat(error),
			Size(2 * TRACKING_HSIZE + 1, 2 * TRACKING_HSIZE + 1),//size of searching window for each Pyramid level
			LK_PYRAMID_LEVEL,// now is 4, the maximum Pyramid levels
			TermCriteria(TermCriteria::COUNT | TermCriteria::EPS,// "type", this means that both termcriteria work here
			MAX_ITER,
			ACCURACY),
			1,//enables optical flow initialization
			LAMBDA);//minEigTheshold

		//renew prevPyr
		prevPyr.swap(nextPyr);

		//clear feature counting for each grid
		for (int k = 0; k< (hgrids.x*hgrids.y); k++)
		{
			feanumofGrid[k] = 0;
		}

		// 	  overflow = 0;
		int tab = 0;
		for (size_t i = 0; i<points1.size(); i++)
		{
			if (status[i] && points1[i].x > usableFrac*im1.cols && points1[i].x < (1.0 - usableFrac)*im1.cols && points1[i].y > usableFrac*im1.rows && points1[i].y < (1.0 - usableFrac)*im1.rows)
			{
				bool shouldKill = maskPoint(points1[i].x, points1[i].y);

				if (shouldKill)
				{
					numActiveTracks--;
					int index1 = index;
					int index2 = index;
				}
				else
				{
					preFeas.push_back(allFeas[i]);
					trackedFeas.push_back(points1[i]);

					if (i < Num){
						Node*head1;
						listTrack(head1, points1[i], index, FeasList_src[index - 1][i],tab);
						FeasList_src[index - 1][i]->next = head1;
						List_src.push_back(head1);
						tab++;
					}
					else if (i >= Num){
						Node*head;
						initial(head);
						Node*head0;
						listTrack(head0, allFeas[i], index - 1, head, Numfront++);
						head->next = head0;
						Node*head1;
						listTrack(head1, points1[i], index, head0,tab);
						head0->next = head1;
						FeasList_src[index - 1].push_back(head0);
						List_src.push_back(head1);
						tab++;
					}


					int hgridIdx =
						(int)(floor((double)(points1[i].x) / (double)((double)(im1.size().width) / (double)(hgrids.x)))
						+ hgrids.x * floor((double)(points1[i].y) / (double)((double)(im1.size().height) / (double)(hgrids.y))));

					feanumofGrid[hgridIdx]++;
				}
			}
			else
			{
				numActiveTracks--;
				int index3 = index;
				int index4 = index;
			}
		}
	}
	else
	{
		//clear feature counting for each grid
		for (int k = 0; k< (hgrids.x*hgrids.y); k++)
		{
			feanumofGrid[k] = 0;
		}

		// 	  overflow = MaxTracks;
	}
	////allFeas = Position;
	//if (index>0)
	//HomographyInliers(FeasList_src[index-1], List_src);

	if (index > 0)
		//rejectOutliers(FeasList_src[index - 1], List_src);
	rejectOutliers2(FeasList_src[index - 1], List_src);


	allFeas.resize(0);
	for (int i = 0; i < List_src.size(); i++){
		allFeas.push_back(List_src[i]->Pts);
	}
	FeasList_src.push_back(List_src);

	int ntoadd = MaxTracks - numActiveTracks;

	if (ntoadd > minToAdd)
	{
		//unusedRoom sum
		unusedRoom = 0;

		//hungry sum
		gridsHungry = 0;

		//hungry grids
		vector<pair<int, double> > hungryGrid;

		//room for adding featurs to each grid
		int room = 0;

		//the hungry degree of a whole frame 
		int hungry = 0;

		//set a specific cell as ROI
		Mat sub_image;

		//set the corresponding mask for the previously choosen cell
		Mat sub_mask;

		//keypoints detected from each grids
		vector<vector<cv::KeyPoint> > sub_keypoints;
		sub_keypoints.resize(hgrids.x * hgrids.y);

		//patch for computing variance
		cv::Mat patch;
		int midGrid = floor((hgrids.x*hgrids.y - 1) / 2.0);

		//the first round resampling on each grid
		for (int q = 0; q < hgrids.x*hgrids.y && numActiveTracks < MaxTracks; q++)
		{

			int i = q;
			if (q == 0)
				i = midGrid;
			if (q == midGrid)
				i = 0;

			room = fealimitGrid - feanumofGrid[i];
			if (room > fealimitGrid*minAddFrac)
			{
				//rowIndx for cells
				int celly = i / hgrids.x;

				//colIndx for cells
				int cellx = i - celly * hgrids.x;

				//rowRang for pixels
				Range row_range((celly*im1.rows) / hgrids.y, ((celly + 1)*im1.rows) / hgrids.y);

				//colRange for pixels
				Range col_range((cellx*im1.cols) / hgrids.x, ((cellx + 1)*im1.cols) / hgrids.x);

				sub_image = im1(cv::Rect(col_range.start,//min_x
					row_range.start,//min_y
					col_range.size(),//length_x
					row_range.size()//length_y
					));

				sub_mask = curMask(cv::Rect(col_range.start,
					row_range.start,
					col_range.size(),
					row_range.size()
					));


				float lastP = ((float)lastNumDetectedGridFeatures[i] - (float)15 * room) / ((float)15 * room);
				
		
				//float newThresh = detector[i].getDouble("threshold");
				newThresh = newThresh + ceil(DETECT_GAIN*lastP);

				if (newThresh > 200)
					newThresh = 200;
				if (newThresh < 5.0)
					newThresh = 5.0;

				//detector[i].set("threshold", newThresh);
				
				//detect keypoints in this cell
				//detector[i]->detect(sub_image, sub_keypoints[i], sub_mask);
				FAST(sub_image, sub_keypoints[i], newThresh);

				lastNumDetectedGridFeatures[i] = sub_keypoints[i].size();
				KeyPointsFilter::retainBest(sub_keypoints[i], 2 * fealimitGrid);

				//sort features
				std::sort(sub_keypoints[i].begin(), sub_keypoints[i].end(), rule);

				//for each feature ...
				std::vector<cv::KeyPoint>::iterator it = sub_keypoints[i].begin(), end = sub_keypoints[i].end();
				int n = 0;

				//first round
				for (; n < room && it != end && numActiveTracks < MaxTracks; ++it)
				{
					//transform grid based position to image based position
					it->pt.x += col_range.start;
					it->pt.y += row_range.start;

					//check is features are being too close
					if (curMask.at<unsigned char>(cv::Point(it->pt.x, it->pt.y)) == 0)
					{
						continue;
					}


					//consider those weak features
					if (it->response < 20)
					{
						//variance of a patch
						double textureness;

						//check if this feature is too close to the image border
						if (it->pt.x - 2 <0 || it->pt.y - 2 <0 ||
							it->pt.x + 2 > im1.cols - 1 || it->pt.y + 2 > im1.rows - 1)
							continue;

						//patch around the feature
						patch = im1(cv::Rect(it->pt.x - 2, it->pt.y - 2, 5, 5));

						
					}

					//runs to here means this feature will be kept
					cv::rectangle(curMask,
								  cv::Point(int(it->pt.x - MASK_RADIUS / 2 + .5), int(it->pt.y - MASK_RADIUS / 2 + .5)),//upperLeft
								  cv::Point(int(it->pt.x + MASK_RADIUS / 2 + .5), int(it->pt.y + MASK_RADIUS / 2 + .5)),//downRight
								  cv::Scalar(0), -1);

					allFeas.push_back(cv::Point2f(it->pt.x, it->pt.y));

					++numActiveTracks;

					n++;
				}

				//recollects unused room
				if (room > n)
				{
				}
				else
				{
					//records hungry grid's index and how hungry they are
					hungryGrid.push_back(std::make_pair(i, (end - it)));

					//sums up to get the total hungry degree
					hungry += hungryGrid.back().second;
				}
			}
		}

		//begin of second round
		unusedRoom = MaxTracks - numActiveTracks;

		//resampling for the second round
		if (unusedRoom > minToAdd)
		{
			vector<pair<int, double> >::iterator it = hungryGrid.begin(), end = hungryGrid.end();
			for (; it != end; it++)
			{
				//rowIndx for cells
				int celly = it->first / hgrids.x;

				//colIndx for cells
				int cellx = it->first - celly * hgrids.x;

				//rowRang for pixels
				Range row_range((celly*im1.rows) / hgrids.y, ((celly + 1)*im1.rows) / hgrids.y);

				//colRange for pixels
				Range col_range((cellx*im1.cols) / hgrids.x, ((cellx + 1)*im1.cols) / hgrids.x);

				//how much food can we give it
				room = floor((double)(unusedRoom * it->second) / (double)hungry);

				//add more features to this grid
				vector<KeyPoint>::iterator itPts = sub_keypoints[it->first].end() - (it->second),
					endPts = sub_keypoints[it->first].end();
				for (int m = 0;
					m < room && itPts != endPts;
					itPts++)
				{
					//transform grid based position to image based position
					itPts->pt.x += col_range.start;
					itPts->pt.y += row_range.start;

					//check is features are being too close
					if (curMask.at<unsigned char>(cv::Point(itPts->pt.x, itPts->pt.y)) == 0)
					{
						continue;
					}

					//consider those weak features
					if (itPts->response < 20)
					{
						//variance of a patch
						double textureness;

						//check if this feature is too close to the image border
						if (itPts->pt.x - 2 <0 || itPts->pt.y - 2 <0 ||
							itPts->pt.x + 2 > im1.cols - 1 || itPts->pt.y + 2 > im1.rows - 1)
							continue;

						//patch around the feature
						patch = im1(cv::Rect(itPts->pt.x - 2, itPts->pt.y - 2, 5, 5));
					}

					//runs to here means this feature will be kept
					cv::rectangle(curMask,
								  cv::Point(int(itPts->pt.x - MASK_RADIUS / 2 + .5), int(itPts->pt.y - MASK_RADIUS / 2 + .5)),//upperLeft
								  cv::Point(int(itPts->pt.x + MASK_RADIUS / 2 + .5), int(itPts->pt.y + MASK_RADIUS / 2 + .5)),//downRight
								  cv::Scalar(0), -1);

					allFeas.push_back(cv::Point2f(itPts->pt.x, itPts->pt.y));
					
					//counts the fatures that have added
					m++;
				}
			}
		}
	}
	return 1;
}


