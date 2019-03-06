#pragma once
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include "define.h"
#ifdef WIN32 /*[*/
#include <io.h>
#endif /*]*/
using namespace std;
//function definition for trankBall.h
#ifndef _TRACKBALL_H_
#define _TRACKBALL_H_
class TrackBall 
{
public:
	TrackBall( void );
	TrackBall( int rot_button, int pan_button, int zoom_button );
	void reset( void );
	void applyTransform( void );
	void reshape( int width, int height );
	void mouse( int button, int state, int x, int y );
	void motion( int x, int y );
	void load( const char *file );
	void save( const char *file );
	
private:
	void captureTransform( void );
    GLdouble tb_angle;
	GLdouble tb_axis[3];
	GLdouble tb_transform[4*4];
    GLuint tb_width;
	GLuint tb_height;
    GLdouble tb_pan_x;
	GLdouble tb_pan_y;
    GLdouble tb_zoom;
	GLdouble tb_zoom_inc;
    GLint tb_rot_button;
	GLint tb_pan_button;
	GLint tb_zoom_button;
    GLint tb_mouse_button;
	GLint tb_mouse_x;
	GLint tb_mouse_y;
    
	GLdouble tb_model_mat[4*4];
	GLdouble tb_proj_mat[4*4];
	GLint tb_viewport[4];
};
#endif
#ifndef __SCOLORPOINT3D__
#define __SCOLORPOINT3D__
struct SColorPoint3D
{
	double  X;
	double  Y;
	double  Z;
	double  R;
	double  G;
	double  B;

	SColorPoint3D(){
		X = Y = Z = R = G = B = 0.0;
	}
	SColorPoint3D(const SColorPoint3D &p3d){
		X = p3d.X;
		Y = p3d.Y;
		Z = p3d.Z;
	
		R = p3d.R;
		G = p3d.G;
		B = p3d.B;
	}
};
#endif