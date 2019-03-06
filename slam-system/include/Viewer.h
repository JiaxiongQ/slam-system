#pragma once
#include "TrackerBall.h"
#include "shader_util.h"
#include "GLHelper.h"
#include "Draw.h"
#include "define.h"
#include "lock.h"
#define PI                  3.1415926535897932384626433832795
#define OBJECT_RADIUS       10.0
extern int  NC;
extern int  NP;
extern sem_t sem;
extern bool startFlag;
extern vector<cv::Mat> imgs;
extern vector< vector<cv::Point2f> > points2_for_show;//特征点集
extern int NCAMS;
// Material properties for all objects.
const GLfloat materialSpecular[] = { 1.0, 1.0, 1.0, 1.0 };
const GLfloat materialShininess[] = { 128.0 };
const GLfloat materialEmission[] = { 0.0, 0.0, 0.0, 1.0 };
// Light 0.
const GLfloat light0Ambient[] = { 0.5, 0.5, 0.5, 1.0 };
const GLfloat light0Diffuse[] = { 0.7, 0.7, 0.7, 1.0 };
const GLfloat light0Specular[] = { 0.9, 0.9, 0.9, 1.0 };
const GLfloat light0Position[] = { OBJECT_RADIUS*10.0, OBJECT_RADIUS*10.0, OBJECT_RADIUS*10.0, 1.0 };
// Shaders' filenames.
const char vertShaderFilename[] = "phong_shading.vert";
const char fragShaderFilename[] = "phong_shading.frag";

void DrawAxes( double length );
void MyDisplay(void);
void MyCameraDisplay(void);
void MyKeyboard( unsigned char key, int x, int y );
void MyReshape( int w, int h );
static void MyMouse( int button, int state, int x, int y );
static void MyMotion( int x, int y );
void MyInit( void );
void MyIdle(void);
void Mytimer(int value);
void delay_msec(int msec);
void View3D(int argc,char** argv);
void View3D(int argc,char** argv,vector<vector<SColorPoint3D>> &vP);
void View3D(int argc,char** argv,vector<vector<SColorPoint3D>> &vP,vector< vector<cv::Point3f> > &ds);

