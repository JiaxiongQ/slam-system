#pragma once
#include "shader_util.h"

#ifndef GLHELPER_H_
#define GLHELPER_H_
inline void mat44Trans(const double * A, double* AT) {
	AT[0] = A[0];
	AT[4] = A[1];
	AT[8] = A[2];
	AT[12] = A[3];
	AT[1] = A[4];
	AT[5] = A[5];
	AT[9] = A[6];
	AT[13] = A[7];
	AT[2] = A[8];
	AT[6] = A[9];
	AT[10] = A[10];
	AT[14] = A[11];
	AT[3] = A[12];
	AT[7] = A[13];
	AT[11] = A[14];
	AT[15] = A[15];
}


inline void mat33TransProdVec(const double* A, const double* b, double *r) {
	r[0] = A[0] * b[0] + A[3] * b[1] + A[6] * b[2];
	r[1] = A[1] * b[0] + A[4] * b[1] + A[7] * b[2];
	r[2] = A[2] * b[0] + A[5] * b[1] + A[8] * b[2];
}

inline void getCameraCenter(const double R[9], const double t[3], double org[3]) {
	mat33TransProdVec(R, t, org);
	org[0] = -org[0];
	org[1] = -org[1];
	org[2] = -org[2];
}

inline void getCameraCenterAxes(const double R[9], const double t[3], double org[3], double xp[3], double yp[3], double zp[3]) {
	getCameraCenter(R, t, org);
	
	xp[0] = R[0];
	xp[1] = R[1];
	xp[2] = R[2];

	yp[0] = R[3];
	yp[1] = R[4];
	yp[2] = R[5];

	zp[0] = R[6];
	zp[1] = R[7];
	zp[2] = R[8];
}

inline void glpt3(double* p) {
	glVertex3f(p[0], p[1], p[2]);
}
inline void glquiver3(double* p, double *v, double alpha = 1.0) {
	glVertex3f(p[0], p[1], p[2]);
	glVertex3f(alpha * v[0] + p[0], alpha * v[1] + p[1], alpha * v[2] + p[2]);
}
inline void glline(double* p, double *q) {
	glVertex3f(p[0], p[1], p[2]);
	glVertex3f(q[0], q[1], q[2]);
}
/* c = alpha*a + beta*b */
inline void add(const double* a, const double* b, double * c, double alpha = 1.0, double beta = 1.0) {
	c[0] = alpha * a[0] + beta * b[0];
	c[1] = alpha * a[1] + beta * b[1];
	c[2] = alpha * a[2] + beta * b[2];
}
/* b = alpha*a */
inline void mul(const double* a, double* b, double alpha) {
	b[0] = alpha * a[0];
	b[1] = alpha * a[1];
	b[2] = alpha * a[2];
}
void drawCircle(float cx, float cy, float r, int num_segments);
void drawLine(float x0, float y0, float x1, float y1);
void drawBlock(float l, float t, float w, float h, const float* color);
void glutPrint2D(float x, float y, const char* text, float r, float g, float b, float a, bool large);
void glutPrint3D(float x, float y, float z, const char* text, float r, float g, float b, float a, bool large);

/* get row major model view matrix*/
void getModelViewMatrix(double M[16]);
void getModelViewMats(double R[9], double t[3]);
void setModelViewMatrix(const double M[16]);
void setModelViewMats(const double R[9], const double t[3]);
void getRotationTranslation(const double MT[16], double R[9], double t[3]);
void setRotationTranslation(const double R[9], const double t[3], double MT[16]);

/* draw camera pose*/
void drawCameraPose(const double R[9],
	    const double C[3],
		double scale,
		double cr = 0,
		double cg = 0,
		double cb = 0,
		double linewidth = 1.0);
#endif /* GLHELPER_H_ */

/**
 * get the camera center
 */
void getCameraCenter(const double R[9], const double t[3], double org[3]);
/**
 * get the camera center togther with the three-axes
 */
void getCameraCenterAxes(const double R[9], const double t[3], double org[3], double ax[3], double ay[3], double az[3]);