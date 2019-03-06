#include "define.h"
#include "shader_util.h"
#include "GLHelper.h"
extern vector< vector<SColorPoint3D> > vPointCloud;
extern int step;
extern int widSize;

extern vector< vector<cv::Point3f> > dypts;
const GLfloat objectColor[] = { 0.9, 0.6, 0.4 };
void DrawPointClouds();
void DrawDynamicPtsCurve();
void DrawCameras();

