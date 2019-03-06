#include "MycalcopticalFlow.h"

//------------------------------------------------------------------------//
//                            LK-OpticalFlow                              //
//------------------------------------------------------------------------//
extern void lkopticalflowt::calcSharrDeriv(cv::Mat& src,
	cv::Mat& dst)
{
	using namespace cv;
	//using cv::detail::deriv_type; //short
	int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols*cn, depth = src.depth();
	CV_Assert(depth == CV_8U);
	dst.create(rows, cols, CV_MAKETYPE(DataType<short>::depth, cn * 2));

#ifdef HAVE_TEGRA_OPTIMIZATION
	if (tegra::calcSharrDeriv(src, dst))
		return;
#endif

	int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
	AutoBuffer<short> _tempBuf(delta * 2 + 64);
	short *trow0 = alignPtr(_tempBuf + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

#if CV_SSE2
	__m128i z = _mm_setzero_si128(), c3 = _mm_set1_epi16(3), c10 = _mm_set1_epi16(10);
#endif

	for (y = 0; y < rows; y++)
	{
		const uchar* srow0 = src.ptr<uchar>(y > 0 ? y - 1 : rows > 1 ? 1 : 0);
		const uchar* srow1 = src.ptr<uchar>(y);
		const uchar* srow2 = src.ptr<uchar>(y < rows - 1 ? y + 1 : rows > 1 ? rows - 2 : 0);
		short* drow = dst.ptr<short>(y);

		// do vertical convolution
		x = 0;
#if CV_SSE2
		for (; x <= colsn - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s0, s2), c3), _mm_mullo_epi16(s1, c10));
			__m128i t1 = _mm_sub_epi16(s2, s0);
			_mm_store_si128((__m128i*)(trow0 + x), t0);
			_mm_store_si128((__m128i*)(trow1 + x), t1);
		}
#endif

#if CV_NEON

		const uint16x8_t q8 = vdupq_n_u16(3);
		const uint8x8_t d18 = vdup_n_u8(10);

		const int16x8_t q8i = vdupq_n_s16(3);
		const int16x8_t q9 = vdupq_n_s16(10);

		for (; x <= colsn - 8; x += 8)
		{
			uint8x8_t d0 = vld1_u8((const uint8_t*)&srow0[x]);
			uint8x8_t d1 = vld1_u8((const uint8_t*)&srow1[x]);
			uint8x8_t d2 = vld1_u8((const uint8_t*)&srow2[x]);
			uint16x8_t q4 = vaddl_u8(d0, d2);
			uint16x8_t q11 = vsubl_u8(d2, d0);
			uint16x8_t q5 = vmulq_u16(q4, q8);
			uint16x8_t q6 = vmull_u8(d1, d18);
			uint16x8_t q10 = vaddq_u16(q6, q5);
			vst1q_u16((uint16_t*)&trow0[x], q10);
			vst1q_u16((uint16_t*)&trow1[x], q11);

		}
#endif
		for (; x < colsn; x++)
		{
			int t0 = (srow0[x] + srow2[x]) * 3 + srow1[x] * 10;
			int t1 = srow2[x] - srow0[x];
			trow0[x] = (short)t0;
			trow1[x] = (short)t1;
		}

		// make border
		int x0 = (cols > 1 ? 1 : 0)*cn, x1 = (cols > 1 ? cols - 2 : 0)*cn;
		for (int k = 0; k < cn; k++)
		{
			trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
			trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
		}

		// do horizontal convolution, interleave the results and store them to dst
		x = 0;
#if CV_SSE2
		for (; x <= colsn - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(trow0 + x - cn));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(trow0 + x + cn));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(trow1 + x - cn));
			__m128i s3 = _mm_load_si128((const __m128i*)(trow1 + x));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(trow1 + x + cn));

			__m128i t0 = _mm_sub_epi16(s1, s0);
			__m128i t1 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s2, s4), c3), _mm_mullo_epi16(s3, c10));
			__m128i t2 = _mm_unpacklo_epi16(t0, t1);
			t0 = _mm_unpackhi_epi16(t0, t1);
			// this can probably be replaced with aligned stores if we aligned dst properly.
			_mm_storeu_si128((__m128i*)(drow + x * 2), t2);
			_mm_storeu_si128((__m128i*)(drow + x * 2 + 8), t0);
		}
#endif

#if CV_NEON
		for (; x <= colsn - 8; x += 8)
		{

			int16x8_t q0 = vld1q_s16((const int16_t*)&trow0[x + cn]);
			int16x8_t q1 = vld1q_s16((const int16_t*)&trow0[x - cn]);
			int16x8_t q2 = vld1q_s16((const int16_t*)&trow1[x + cn]);
			int16x8_t q3 = vld1q_s16((const int16_t*)&trow1[x - cn]);
			int16x8_t q5 = vsubq_s16(q0, q1);
			int16x8_t q6 = vaddq_s16(q2, q3);
			int16x8_t q4 = vld1q_s16((const int16_t*)&trow1[x]);
			int16x8_t q7 = vmulq_s16(q6, q8i);
			int16x8_t q10 = vmulq_s16(q4, q9);
			int16x8_t q11 = vaddq_s16(q7, q10);
			int16x4_t d22 = vget_low_s16(q11);
			int16x4_t d23 = vget_high_s16(q11);
			int16x4_t d11 = vget_high_s16(q5);
			int16x4_t d10 = vget_low_s16(q5);
			int16x4x2_t q5x2, q11x2;
			q5x2.val[0] = d10; q5x2.val[1] = d22;
			q11x2.val[0] = d11; q11x2.val[1] = d23;
			vst2_s16((int16_t*)&drow[x * 2], q5x2);
			vst2_s16((int16_t*)&drow[(x * 2) + 8], q11x2);

		}

#endif

		for (; x < colsn; x++)
		{
			short t0 = (short)(trow0[x + cn] - trow0[x - cn]);
			short t1 = (short)((trow1[x + cn] + trow1[x - cn]) * 3 + trow1[x] * 10);
			drow[x * 2] = t0; drow[x * 2 + 1] = t1;
		}
	}
}
int lkopticalflowt::buildOpticalFlowPyramid(cv::InputArray _img,
	cv::OutputArrayOfArrays pyramid,
	cv::Size winSize,
	int maxLevel,
	bool withDerivatives,
	int pyrBorder,
	int derivBorder,
	bool tryReuseInputImage)
{
	cv::Mat img = _img.getMat();
	CV_Assert(img.depth() == CV_8U && winSize.width > 2 && winSize.height > 2);
	int pyrstep = withDerivatives ? 2 : 1;

	pyramid.create(1, (maxLevel + 1) * pyrstep, 0 /*type*/, -1, true, 0);

	int derivType = CV_MAKETYPE(cv::DataType<short>::depth, img.channels() * 2);

	//level 0
	bool lvl0IsSet = false;
	if (tryReuseInputImage && img.isSubmatrix() && (pyrBorder & cv::BORDER_ISOLATED) == 0)
	{
		cv::Size wholeSize;
		cv::Point ofs;
		img.locateROI(wholeSize, ofs);
		if (ofs.x >= winSize.width && ofs.y >= winSize.height
			&& ofs.x + img.cols + winSize.width <= wholeSize.width
			&& ofs.y + img.rows + winSize.height <= wholeSize.height)
		{
			pyramid.getMatRef(0) = img;
			lvl0IsSet = true;
		}
	}

	if (!lvl0IsSet)
	{
		cv::Mat& temp = pyramid.getMatRef(0);

		if (!temp.empty())
			temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
		if (temp.type() != img.type() || temp.cols != winSize.width * 2 + img.cols || temp.rows != winSize.height * 2 + img.rows)
			temp.create(img.rows + winSize.height * 2, img.cols + winSize.width * 2, img.type());

		if (pyrBorder == cv::BORDER_TRANSPARENT)
			img.copyTo(temp(cv::Rect(winSize.width, winSize.height, img.cols, img.rows)));
		else
			copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder);
		temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
	}

	cv::Size sz = img.size();
	cv::Mat prevLevel = pyramid.getMatRef(0);
	cv::Mat thisLevel = prevLevel;

	for (int level = 0; level <= maxLevel; ++level)
	{
		if (level != 0)
		{
			cv::Mat& temp = pyramid.getMatRef(level * pyrstep);

			if (!temp.empty())
				temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
			if (temp.type() != img.type() || temp.cols != winSize.width * 2 + sz.width || temp.rows != winSize.height * 2 + sz.height)
				temp.create(sz.height + winSize.height * 2, sz.width + winSize.width * 2, img.type());

			thisLevel = temp(cv::Rect(winSize.width, winSize.height, sz.width, sz.height));
			pyrDown(prevLevel, thisLevel, sz);

			if (pyrBorder != cv::BORDER_TRANSPARENT)
				copyMakeBorder(thisLevel, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder | cv::BORDER_ISOLATED);
			temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
		}

		if (withDerivatives)
		{
			cv::Mat& deriv = pyramid.getMatRef(level * pyrstep + 1);

			if (!deriv.empty())
				deriv.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
			if (deriv.type() != derivType || deriv.cols != winSize.width * 2 + sz.width || deriv.rows != winSize.height * 2 + sz.height)
				deriv.create(sz.height + winSize.height * 2, sz.width + winSize.width * 2, derivType);

			cv::Mat derivI = deriv(cv::Rect(winSize.width, winSize.height, sz.width, sz.height));
			calcSharrDeriv(thisLevel, derivI);

			if (derivBorder != cv::BORDER_TRANSPARENT)
				copyMakeBorder(derivI, deriv, winSize.height, winSize.height, winSize.width, winSize.width, derivBorder | cv::BORDER_ISOLATED);
			deriv.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
		}

		sz = cv::Size((sz.width + 1) / 2, (sz.height + 1) / 2);
		if (sz.width <= winSize.width || sz.height <= winSize.height)
		{
			pyramid.create(1, (level + 1) * pyrstep, 0 /*type*/, -1, true, 0);//check this
			return level;
		}

		prevLevel = thisLevel;
	}

	return maxLevel;
}
	void lkopticalflowt::calcOpticalFlowPyrLK(cv::InputArray _prevImg,
		cv::InputArray _nextImg,
		cv::InputArray _prevPts,
		cv::InputOutputArray _nextPts,
		cv::OutputArray _status,
		cv::OutputArray _err,
		cv::Size winSize,
	int maxLevel,
	cv::TermCriteria criteria,
	int flags,
	double minEigThreshold)
{

	
	cv::Mat prevPtsMat = _prevPts.getMat();
	//Mat &prevPtsMat = _prevPts;
	


	const int derivDepth = cv::DataType<short>::depth;

	CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2);

	int level = 0, i, npoints;
	CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);

	if (npoints == 0)
	{
		_nextPts.release();
		_status.release();
		_err.release();
		return;
	}


	cv::Mat nextPtsMat = _nextPts.getMat();
	
	CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);

	const cv::Point2f* prevPts = (const cv::Point2f*)prevPtsMat.data;
	cv::Point2f* nextPts = (cv::Point2f*)nextPtsMat.data;

	//_status.create((int)npoints, 1, CV_8U, -1, true);

	cv::Mat statusMat = _status.getMat(), errMat;
	CV_Assert(statusMat.isContinuous());
	uchar* status = statusMat.data;
	float* err = 0;

	for (i = 0; i < npoints; i++)
		status[i] = true;

	err = (float*)errMat.data;

	/*if (_err.needed())
	{
		_err.create((int)npoints, 1, CV_32F, -1, true);
		errMat = _err.getMat();
		CV_Assert(errMat.isContinuous());
		err = (float*)errMat.data;
	}*/

	vector<cv::Mat> prevPyr, nextPyr;
	int levels1 = -1;
	int lvlStep1 = 1;
	int levels2 = -1;
	int lvlStep2 = 1;

	if (_prevImg.kind() == cv::_InputArray::STD_VECTOR_MAT)
	{
		_prevImg.getMatVector(prevPyr);

		levels1 = int(prevPyr.size()) - 1;
		CV_Assert(levels1 >= 0);

		if (levels1 % 2 == 1 && prevPyr[0].channels() * 2 == prevPyr[1].channels() && prevPyr[1].depth() == derivDepth)
		{
			lvlStep1 = 2;
			levels1 /= 2;
		}

		// ensure that pyramid has reqired padding
		if (levels1 > 0)
		{
			cv::Size fullSize;
			cv::Point ofs;
			prevPyr[lvlStep1].locateROI(fullSize, ofs);
			CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
				&& ofs.x + prevPyr[lvlStep1].cols + winSize.width <= fullSize.width
				&& ofs.y + prevPyr[lvlStep1].rows + winSize.height <= fullSize.height);
		}

		if (levels1 < maxLevel)
			maxLevel = levels1;
	}

	
	if (_nextImg.kind() == cv::_InputArray::STD_VECTOR_MAT)
	{
		_nextImg.getMatVector(nextPyr);

		levels2 = int(nextPyr.size()) - 1;
		CV_Assert(levels2 >= 0);

		if (levels2 % 2 == 1 && nextPyr[0].channels() * 2 == nextPyr[1].channels() && nextPyr[1].depth() == derivDepth)
		{
			lvlStep2 = 2;
			levels2 /= 2;
		}

		// ensure that pyramid has reqired padding
		if (levels2 > 0)
		{
			cv::Size fullSize;
			cv::Point ofs;
			nextPyr[lvlStep2].locateROI(fullSize, ofs);
			CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
				&& ofs.x + nextPyr[lvlStep2].cols + winSize.width <= fullSize.width
				&& ofs.y + nextPyr[lvlStep2].rows + winSize.height <= fullSize.height);
		}

		if (levels2 < maxLevel)
			maxLevel = levels2;
	}
	
	if (levels1 < 0)
		maxLevel = lkopticalflowt::buildOpticalFlowPyramid(_prevImg, prevPyr, winSize, maxLevel, false);

	if (levels2 < 0)
		maxLevel = lkopticalflowt::buildOpticalFlowPyramid(_nextImg, nextPyr, winSize, maxLevel, false);

	if ((criteria.type & cv::TermCriteria::COUNT) == 0)
		criteria.maxCount = 30;
	else
		criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
	if ((criteria.type & cv::TermCriteria::EPS) == 0)
		criteria.epsilon = 0.01;
	else
		criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
	criteria.epsilon *= criteria.epsilon;

	// dI/dx ~ Ix, dI/dy ~ Iy
	cv::Mat derivIBuf;
	if (lvlStep1 == 1)
		derivIBuf.create(prevPyr[0].rows + winSize.height * 2, prevPyr[0].cols + winSize.width * 2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));

	
	for (level = maxLevel; level >= 0; level--)
	{
		cv::Mat derivI;
		if (lvlStep1 == 1)
		{
			cv::Size imgSize = prevPyr[level * lvlStep1].size();
			cv::Mat _derivI(imgSize.height + winSize.height * 2,
				imgSize.width + winSize.width * 2, derivIBuf.type(), derivIBuf.data);
			derivI = _derivI(cv::Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
			calcSharrDeriv(prevPyr[level * lvlStep1], derivI);
			copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);
		}
		else
			derivI = prevPyr[level * lvlStep1 + 1];

		CV_Assert(prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size());
		CV_Assert(prevPyr[level * lvlStep1].type() == nextPyr[level * lvlStep2].type());

#ifdef HAVE_TEGRA_OPTIMIZATION
		typedef tegra::LKTrackerInvoker<lkopticalflowt::detail::LKTrackerInvoker> LKTrackerInvoker;
#else
		typedef lkopticalflowt::detail::LKTrackerInvoker LKTrackerInvoker;
#endif
		
		parallel_for_(cv::Range(0, npoints), LKTrackerInvoker(prevPyr[level * lvlStep1], derivI,
			nextPyr[level * lvlStep2], prevPts, nextPts,
			status, err,
			winSize, criteria, level, maxLevel,
			flags, (float)minEigThreshold));
	}
}


lkopticalflowt::detail::LKTrackerInvoker::LKTrackerInvoker(const cv::Mat& _prevImg,
	const cv::Mat& _prevDeriv,
	const cv::Mat& _nextImg,
	const cv::Point2f* _prevPts,
	cv::Point2f* _nextPts,
	uchar* _status,
	float* _err,
	cv::Size _winSize,
	cv::TermCriteria _criteria,
	int _level,
	int _maxLevel,
	int _flags,
	float _minEigThreshold)
{
	prevImg = &_prevImg;
	prevDeriv = &_prevDeriv;
	nextImg = &_nextImg;
	prevPts = _prevPts;
	nextPts = _nextPts;
	status = _status;
	err = _err;
	winSize = _winSize;
	criteria = _criteria;
	level = _level;
	maxLevel = _maxLevel;
	flags = _flags;
	minEigThreshold = _minEigThreshold;
}
void lkopticalflowt::detail::LKTrackerInvoker::operator()(const cv::Range& range) const
{
	cv::Point2f halfWin((winSize.width - 1)*0.5f, (winSize.height - 1)*0.5f);
	const cv::Mat& I = *prevImg;
	const cv::Mat& J = *nextImg;
	const cv::Mat& derivI = *prevDeriv;

	int j, cn = I.channels(), cn2 = cn * 2;
	cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2));
	int derivDepth = cv::DataType<deriv_type>::depth;

	cv::Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf);
	cv::Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf + winSize.area()*cn);

	for (int ptidx = range.start; ptidx < range.end; ptidx++)
	{
		cv::Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
		cv::Point2f nextPt;
		if (level == maxLevel)
		{
			if (flags & 4/*OPTFLOW_USE_INITIAL_FLOW*/)
				nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
			else
				nextPt = prevPt;
		}
		else
			nextPt = nextPts[ptidx] * 2.f;
		nextPts[ptidx] = nextPt;

		cv::Point2i iprevPt, inextPt;
		prevPt -= halfWin;
		iprevPt.x = cvFloor(prevPt.x);
		iprevPt.y = cvFloor(prevPt.y);

		if (iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
			iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows)
		{
			if (level == 0)
			{
				if (status)
					status[ptidx] = false;
				if (err)
					err[ptidx] = 0;
			}
			continue;
		}

		float a = prevPt.x - iprevPt.x;
		float b = prevPt.y - iprevPt.y;
		const int W_BITS = 14, W_BITS1 = 14;
		const float FLT_SCALE = 1.f / (1 << 20);
		int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
		int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
		int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
		int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

		int dstep = (int)(derivI.step / derivI.elemSize1());
		int stepI = (int)(I.step / I.elemSize1());
		int stepJ = (int)(J.step / J.elemSize1());
		float A11 = 0, A12 = 0, A22 = 0;

#if CV_SSE2
		__m128i qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
		__m128i qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
		__m128i z = _mm_setzero_si128();
		__m128i qdelta_d = _mm_set1_epi32(1 << (W_BITS1 - 1));
		__m128i qdelta = _mm_set1_epi32(1 << (W_BITS1 - 5 - 1));
		__m128 qA11 = _mm_setzero_ps(), qA12 = _mm_setzero_ps(), qA22 = _mm_setzero_ps();
#endif

#if CV_NEON

		int CV_DECL_ALIGNED(16) nA11[] = { 0, 0, 0, 0 }, nA12[] = { 0, 0, 0, 0 }, nA22[] = { 0, 0, 0, 0 };
		const int shifter1 = -(W_BITS - 5); //negative so it shifts right
		const int shifter2 = -(W_BITS);

		const int16x4_t d26 = vdup_n_s16((int16_t)iw00);
		const int16x4_t d27 = vdup_n_s16((int16_t)iw01);
		const int16x4_t d28 = vdup_n_s16((int16_t)iw10);
		const int16x4_t d29 = vdup_n_s16((int16_t)iw11);
		const int32x4_t q11 = vdupq_n_s32((int32_t)shifter1);
		const int32x4_t q12 = vdupq_n_s32((int32_t)shifter2);

#endif

		// extract the patch from the first image, compute covariation matrix of derivatives
		int x, y;
		for (y = 0; y < winSize.height; y++)
		{
			const uchar* src = (const uchar*)I.data + (y + iprevPt.y)*stepI + iprevPt.x*cn;
			const deriv_type* dsrc = (const deriv_type*)derivI.data + (y + iprevPt.y)*dstep + iprevPt.x*cn2;

			deriv_type* Iptr = (deriv_type*)(IWinBuf.data + y*IWinBuf.step);
			deriv_type* dIptr = (deriv_type*)(derivIWinBuf.data + y*derivIWinBuf.step);

			x = 0;

#if CV_SSE2
			for (; x <= winSize.width*cn - 4; x += 4, dsrc += 4 * 2, dIptr += 4 * 2)
			{
				__m128i v00, v01, v10, v11, t0, t1;

				v00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x)), z);
				v01 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + cn)), z);
				v10 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + stepI)), z);
				v11 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + stepI + cn)), z);

				t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
					_mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
				t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1 - 5);
				_mm_storel_epi64((__m128i*)(Iptr + x), _mm_packs_epi32(t0, t0));

				v00 = _mm_loadu_si128((const __m128i*)(dsrc));
				v01 = _mm_loadu_si128((const __m128i*)(dsrc + cn2));
				v10 = _mm_loadu_si128((const __m128i*)(dsrc + dstep));
				v11 = _mm_loadu_si128((const __m128i*)(dsrc + dstep + cn2));

				t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
					_mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
				t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
					_mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
				t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta_d), W_BITS1);
				t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta_d), W_BITS1);
				v00 = _mm_packs_epi32(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...

				_mm_storeu_si128((__m128i*)dIptr, v00);
				t0 = _mm_srai_epi32(v00, 16); // Iy0 Iy1 Iy2 Iy3
				t1 = _mm_srai_epi32(_mm_slli_epi32(v00, 16), 16); // Ix0 Ix1 Ix2 Ix3

				__m128 fy = _mm_cvtepi32_ps(t0);
				__m128 fx = _mm_cvtepi32_ps(t1);

				qA22 = _mm_add_ps(qA22, _mm_mul_ps(fy, fy));
				qA12 = _mm_add_ps(qA12, _mm_mul_ps(fx, fy));
				qA11 = _mm_add_ps(qA11, _mm_mul_ps(fx, fx));
			}
#endif

#if CV_NEON

			for (; x <= winSize.width*cn - 4; x += 4, dsrc += 4 * 2, dIptr += 4 * 2)
			{

				uint8x8_t d0 = vld1_u8(&src[x]);
				uint8x8_t d2 = vld1_u8(&src[x + cn]);
				uint16x8_t q0 = vmovl_u8(d0);
				uint16x8_t q1 = vmovl_u8(d2);

				int32x4_t q5 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26);
				int32x4_t q6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27);

				uint8x8_t d4 = vld1_u8(&src[x + stepI]);
				uint8x8_t d6 = vld1_u8(&src[x + stepI + cn]);
				uint16x8_t q2 = vmovl_u8(d4);
				uint16x8_t q3 = vmovl_u8(d6);

				int32x4_t q7 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28);
				int32x4_t q8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29);

				q5 = vaddq_s32(q5, q6);
				q7 = vaddq_s32(q7, q8);
				q5 = vaddq_s32(q5, q7);

				int16x4x2_t d0d1 = vld2_s16(dsrc);
				int16x4x2_t d2d3 = vld2_s16(&dsrc[cn2]);

				q5 = vqrshlq_s32(q5, q11);

				int32x4_t q4 = vmull_s16(d0d1.val[0], d26);
				q6 = vmull_s16(d0d1.val[1], d26);

				int16x4_t nd0 = vmovn_s32(q5);

				q7 = vmull_s16(d2d3.val[0], d27);
				q8 = vmull_s16(d2d3.val[1], d27);

				vst1_s16(&Iptr[x], nd0);

				int16x4x2_t d4d5 = vld2_s16(&dsrc[dstep]);
				int16x4x2_t d6d7 = vld2_s16(&dsrc[dstep + cn2]);

				q4 = vaddq_s32(q4, q7);
				q6 = vaddq_s32(q6, q8);

				q7 = vmull_s16(d4d5.val[0], d28);
				int32x4_t nq0 = vmull_s16(d4d5.val[1], d28);
				q8 = vmull_s16(d6d7.val[0], d29);
				int32x4_t q15 = vmull_s16(d6d7.val[1], d29);

				q7 = vaddq_s32(q7, q8);
				nq0 = vaddq_s32(nq0, q15);

				q4 = vaddq_s32(q4, q7);
				q6 = vaddq_s32(q6, nq0);

				int32x4_t nq1 = vld1q_s32(nA12);
				int32x4_t nq2 = vld1q_s32(nA22);
				nq0 = vld1q_s32(nA11);

				q4 = vqrshlq_s32(q4, q12);
				q6 = vqrshlq_s32(q6, q12);

				q7 = vmulq_s32(q4, q4);
				q8 = vmulq_s32(q4, q6);
				q15 = vmulq_s32(q6, q6);

				nq0 = vaddq_s32(nq0, q7);
				nq1 = vaddq_s32(nq1, q8);
				nq2 = vaddq_s32(nq2, q15);

				vst1q_s32(nA11, nq0);
				vst1q_s32(nA12, nq1);
				vst1q_s32(nA22, nq2);

				int16x4_t d8 = vmovn_s32(q4);
				int16x4_t d12 = vmovn_s32(q6);

				int16x4x2_t d8d12;
				d8d12.val[0] = d8; d8d12.val[1] = d12;
				vst2_s16(dIptr, d8d12);
			}

#endif

			for (; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2)
			{
				int ival = CV_DESCALE(src[x] * iw00 + src[x + cn] * iw01 +
					src[x + stepI] * iw10 + src[x + stepI + cn] * iw11, W_BITS1 - 5);
				int ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
					dsrc[dstep] * iw10 + dsrc[dstep + cn2] * iw11, W_BITS1);
				int iyval = CV_DESCALE(dsrc[1] * iw00 + dsrc[cn2 + 1] * iw01 + dsrc[dstep + 1] * iw10 +
					dsrc[dstep + cn2 + 1] * iw11, W_BITS1);

				Iptr[x] = (short)ival;
				dIptr[0] = (short)ixval;
				dIptr[1] = (short)iyval;

				A11 += (float)(ixval*ixval);
				A12 += (float)(ixval*iyval);
				A22 += (float)(iyval*iyval);
			}
		}

#if CV_SSE2
		float CV_DECL_ALIGNED(16) A11buf[4], A12buf[4], A22buf[4];
		_mm_store_ps(A11buf, qA11);
		_mm_store_ps(A12buf, qA12);
		_mm_store_ps(A22buf, qA22);
		A11 += A11buf[0] + A11buf[1] + A11buf[2] + A11buf[3];
		A12 += A12buf[0] + A12buf[1] + A12buf[2] + A12buf[3];
		A22 += A22buf[0] + A22buf[1] + A22buf[2] + A22buf[3];
#endif

#if CV_NEON
		A11 += (float)(nA11[0] + nA11[1] + nA11[2] + nA11[3]);
		A12 += (float)(nA12[0] + nA12[1] + nA12[2] + nA12[3]);
		A22 += (float)(nA22[0] + nA22[1] + nA22[2] + nA22[3]);
#endif

		A11 *= FLT_SCALE;
		A12 *= FLT_SCALE;
		A22 *= FLT_SCALE;

		float D = A11*A22 - A12*A12;
		float minEig = (A22 + A11 - std::sqrt((A11 - A22)*(A11 - A22) +
			4.f*A12*A12)) / (2 * winSize.width*winSize.height);

		if (err && (flags & 8/*CV_LKFLOW_GET_MIN_EIGENVALS*/) != 0)
			err[ptidx] = (float)minEig;

		if (minEig < minEigThreshold || D < FLT_EPSILON)
		{
			if (level == 0 && status)
				status[ptidx] = false;
			continue;
		}

		D = 1.f / D;

		nextPt -= halfWin;
		cv::Point2f prevDelta;

		for (j = 0; j < criteria.maxCount; j++)
		{
			inextPt.x = cvFloor(nextPt.x);
			inextPt.y = cvFloor(nextPt.y);

			if (inextPt.x < -winSize.width || inextPt.x >= J.cols ||
				inextPt.y < -winSize.height || inextPt.y >= J.rows)
			{
				if (level == 0 && status)
					status[ptidx] = false;
				break;
			}

			a = nextPt.x - inextPt.x;
			b = nextPt.y - inextPt.y;
			iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
			iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
			iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
			iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
			float b1 = 0, b2 = 0;
#if CV_SSE2
			qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
			qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
			__m128 qb0 = _mm_setzero_ps(), qb1 = _mm_setzero_ps();
#endif

#if CV_NEON

			int CV_DECL_ALIGNED(16) nB1[] = { 0, 0, 0, 0 }, nB2[] = { 0, 0, 0, 0 };

			const int16x4_t d26_2 = vdup_n_s16((int16_t)iw00);
			const int16x4_t d27_2 = vdup_n_s16((int16_t)iw01);
			const int16x4_t d28_2 = vdup_n_s16((int16_t)iw10);
			const int16x4_t d29_2 = vdup_n_s16((int16_t)iw11);

#endif

			for (y = 0; y < winSize.height; y++)
			{
				const uchar* Jptr = (const uchar*)J.data + (y + inextPt.y)*stepJ + inextPt.x*cn;
				const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data + y*IWinBuf.step);
				const deriv_type* dIptr = (const deriv_type*)(derivIWinBuf.data + y*derivIWinBuf.step);

				x = 0;

#if CV_SSE2
				for (; x <= winSize.width*cn - 8; x += 8, dIptr += 8 * 2)
				{
					__m128i diff0 = _mm_loadu_si128((const __m128i*)(Iptr + x)), diff1;
					__m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x)), z);
					__m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + cn)), z);
					__m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + stepJ)), z);
					__m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + stepJ + cn)), z);

					__m128i t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
						_mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
					__m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
						_mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
					t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1 - 5);
					t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS1 - 5);
					diff0 = _mm_subs_epi16(_mm_packs_epi32(t0, t1), diff0);
					diff1 = _mm_unpackhi_epi16(diff0, diff0);
					diff0 = _mm_unpacklo_epi16(diff0, diff0); // It0 It0 It1 It1 ...
					v00 = _mm_loadu_si128((const __m128i*)(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
					v01 = _mm_loadu_si128((const __m128i*)(dIptr + 8));
					v10 = _mm_mullo_epi16(v00, diff0);
					v11 = _mm_mulhi_epi16(v00, diff0);
					v00 = _mm_unpacklo_epi16(v10, v11);
					v10 = _mm_unpackhi_epi16(v10, v11);
					qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
					qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
					v10 = _mm_mullo_epi16(v01, diff1);
					v11 = _mm_mulhi_epi16(v01, diff1);
					v00 = _mm_unpacklo_epi16(v10, v11);
					v10 = _mm_unpackhi_epi16(v10, v11);
					qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
					qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
				}
#endif

#if CV_NEON

				for (; x <= winSize.width*cn - 8; x += 8, dIptr += 8 * 2)
				{

					uint8x8_t d0 = vld1_u8(&Jptr[x]);
					uint8x8_t d2 = vld1_u8(&Jptr[x + cn]);
					uint8x8_t d4 = vld1_u8(&Jptr[x + stepJ]);
					uint8x8_t d6 = vld1_u8(&Jptr[x + stepJ + cn]);

					uint16x8_t q0 = vmovl_u8(d0);
					uint16x8_t q1 = vmovl_u8(d2);
					uint16x8_t q2 = vmovl_u8(d4);
					uint16x8_t q3 = vmovl_u8(d6);

					int32x4_t nq4 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26_2);
					int32x4_t nq5 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q0)), d26_2);

					int32x4_t nq6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27_2);
					int32x4_t nq7 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q1)), d27_2);

					int32x4_t nq8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28_2);
					int32x4_t nq9 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q2)), d28_2);

					int32x4_t nq10 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29_2);
					int32x4_t nq11 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q3)), d29_2);

					nq4 = vaddq_s32(nq4, nq6);
					nq5 = vaddq_s32(nq5, nq7);
					nq8 = vaddq_s32(nq8, nq10);
					nq9 = vaddq_s32(nq9, nq11);

					int16x8_t q6 = vld1q_s16(&Iptr[x]);

					nq4 = vaddq_s32(nq4, nq8);
					nq5 = vaddq_s32(nq5, nq9);

					nq8 = vmovl_s16(vget_high_s16(q6));
					nq6 = vmovl_s16(vget_low_s16(q6));

					nq4 = vqrshlq_s32(nq4, q11);
					nq5 = vqrshlq_s32(nq5, q11);

					int16x8x2_t q0q1 = vld2q_s16(dIptr);
					nq11 = vld1q_s32(nB1);
					int32x4_t nq15 = vld1q_s32(nB2);

					nq4 = vsubq_s32(nq4, nq6);
					nq5 = vsubq_s32(nq5, nq8);

					int32x4_t nq2 = vmovl_s16(vget_low_s16(q0q1.val[0]));
					int32x4_t nq3 = vmovl_s16(vget_high_s16(q0q1.val[0]));

					nq7 = vmovl_s16(vget_low_s16(q0q1.val[1]));
					nq8 = vmovl_s16(vget_high_s16(q0q1.val[1]));

					nq9 = vmulq_s32(nq4, nq2);
					nq10 = vmulq_s32(nq5, nq3);

					nq4 = vmulq_s32(nq4, nq7);
					nq5 = vmulq_s32(nq5, nq8);

					nq9 = vaddq_s32(nq9, nq10);
					nq4 = vaddq_s32(nq4, nq5);

					nq11 = vaddq_s32(nq11, nq9);
					nq15 = vaddq_s32(nq15, nq4);

					vst1q_s32(nB1, nq11);
					vst1q_s32(nB2, nq15);
				}

#endif

				for (; x < winSize.width*cn; x++, dIptr += 2)
				{
					int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 +
						Jptr[x + stepJ] * iw10 + Jptr[x + stepJ + cn] * iw11,
						W_BITS1 - 5) - Iptr[x];
					b1 += (float)(diff*dIptr[0]);
					b2 += (float)(diff*dIptr[1]);
				}
			}

#if CV_SSE2
			float CV_DECL_ALIGNED(16) bbuf[4];
			_mm_store_ps(bbuf, _mm_add_ps(qb0, qb1));
			b1 += bbuf[0] + bbuf[2];
			b2 += bbuf[1] + bbuf[3];
#endif

#if CV_NEON

			b1 += (float)(nB1[0] + nB1[1] + nB1[2] + nB1[3]);
			b2 += (float)(nB2[0] + nB2[1] + nB2[2] + nB2[3]);

#endif

			b1 *= FLT_SCALE;
			b2 *= FLT_SCALE;

			cv::Point2f delta((float)((A12*b2 - A22*b1) * D),
				(float)((A12*b1 - A11*b2) * D));
			//delta = -delta;

			nextPt += delta;
			nextPts[ptidx] = nextPt + halfWin;

			if (delta.ddot(delta) <= criteria.epsilon)
				break;

			if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
				std::abs(delta.y + prevDelta.y) < 0.01)
			{
				nextPts[ptidx] -= delta*0.5f;
				break;
			}
			prevDelta = delta;
		}

		if (status[ptidx] && err && level == 0 && (flags & 8/*CV_LKFLOW_GET_MIN_EIGENVALS*/) == 0)
		{
			cv::Point2f nextPoint = nextPts[ptidx] - halfWin;
			cv::Point inextPoint;

			inextPoint.x = cvFloor(nextPoint.x);
			inextPoint.y = cvFloor(nextPoint.y);

			if (inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
				inextPoint.y < -winSize.height || inextPoint.y >= J.rows)
			{
				if (status)
					status[ptidx] = false;
				continue;
			}

			float aa = nextPoint.x - inextPoint.x;
			float bb = nextPoint.y - inextPoint.y;
			iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
			iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
			iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
			iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
			float errval = 0.f;

			for (y = 0; y < winSize.height; y++)
			{
				const uchar* Jptr = (const uchar*)J.data + (y + inextPoint.y)*stepJ + inextPoint.x*cn;
				const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data + y*IWinBuf.step);

				for (x = 0; x < winSize.width*cn; x++)
				{
					int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 +
						Jptr[x + stepJ] * iw10 + Jptr[x + stepJ + cn] * iw11,
						W_BITS1 - 5) - Iptr[x];
					errval += std::abs((float)diff);
				}
			}
			err[ptidx] = errval * 1.f / (32 * winSize.width*cn*winSize.height);
		}
	}
}