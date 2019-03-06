#include "GLHelper.h"
void drawLine(float x0, float y0, float x1, float y1) {
	glBegin(GL_LINES);
	glVertex2f(x0, y0);
	glVertex2f(x1, y1);
	glEnd();
}
void drawCircle(float cx, float cy, float r, int num_segments) {
	float theta = 2 * 3.1415926 / float(num_segments);
	float c = cosf(theta);//precalculate the sine and cosine
	float s = sinf(theta);
	float t;

	float x = r;//we start at angle = 0 
	float y = 0;

	glBegin(GL_LINE_LOOP);
	for (int ii = 0; ii < num_segments; ii++) {
		glVertex2f(x + cx, y + cy);//output vertex 

		//apply the rotation matrix
		t = x;
		x = c * x - s * y;
		y = s * t + c * y;
	}
	
	glEnd();
}
void drawBlock(float l, float t, float w, float h, const float* color) {
	glColor3f(color[0], color[1], color[2]);
	glBegin(GL_QUADS);
	glVertex3f(l, t, 0);
	glVertex3f(l + w, t, 0);
	glVertex3f(l + w, t + h, 0);
	glVertex3f(l, t + h, 0);
	glEnd();
}

void glutPrint2D(float x, float y, const char* text, float r, float g, float b, float a, bool large = false) {
	if (!text || !strlen(text))
		return;
	bool blending = false;
	if (glIsEnabled(GL_BLEND))
		blending = true;
	glEnable(GL_BLEND);
	glColor4f(r, g, b, a);
	glRasterPos2f(x, y);
	while (*text) {
		glutBitmapCharacter(large ? GLUT_BITMAP_TIMES_ROMAN_24 : GLUT_BITMAP_HELVETICA_12, *text);
		text++;
	}
	if (!blending)
		glDisable(GL_BLEND);
}
void glutPrint3D(float x, float y, float z, const char* text, float r, float g, float b, float a, bool large = false) {
	if (!text || !strlen(text))
		return;
	bool blending = false;
	if (glIsEnabled(GL_BLEND))
		blending = true;
	glEnable(GL_BLEND);
	glColor4f(r, g, b, a);
	glRasterPos3f(x, y, z);
	while (*text) {
		glutBitmapCharacter(large ? GLUT_BITMAP_TIMES_ROMAN_24 : GLUT_BITMAP_HELVETICA_12, *text);
		text++;
	}
	if (!blending)
		glDisable(GL_BLEND);
}

void getModelViewMatrix(double M[16]) {
	double MT[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, MT);
	mat44Trans(MT, M);
}
void getModelViewMats(double R[9], double t[3]) {
	double M[16];
	getModelViewMatrix(M);
	getRotationTranslation(M, R, t);
}
void setModelViewMatrix(const double M[16]) {
	double MT[16];
	mat44Trans(M, MT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMultMatrixd(MT);
}

void setModelViewMats(const double R[9], const double t[3]) {
	double M[16];
	setRotationTranslation(R, t, M);
	setModelViewMatrix(M);
}
void getRotationTranslation(const double M[16], double R[9], double t[3]) {
	R[0] = M[0];
	R[1] = M[1];
	R[2] = M[2];
	R[3] = M[4];
	R[4] = M[5];
	R[5] = M[6];
	R[6] = M[8];
	R[7] = M[9];
	R[8] = M[10];
	t[0] = M[3];
	t[1] = M[7];
	t[2] = M[11];
}
void setRotationTranslation(const double R[9], const double t[3], double M[16]) {
	memset(M, 0, sizeof(double) * 16);
	M[15] = 1;
	M[0] = R[0];
	M[1] = R[1];
	M[2] = R[2];
	M[4] = R[3];
	M[5] = R[4];
	M[6] = R[5];
	M[8] = R[6];
	M[9] = R[7];
	M[10] = R[8];
	M[3] = t[0];
	M[7] = t[1];
	M[11] = t[2];
}
#include <cstdio>

void drawCameraPose(const double R[9],
	    const double C[3],
		double scale,
		double cr,
		double cg,
		double cb,
		double linewidth){
	
	double p0[3],xp[3], yp[3], zp[3];//,tmp[3];
	
	int times = 10;
	p0[0] = C[0]*times;
	p0[1] = C[1]*times;
	p0[2] = C[2]*times;

	xp[0] = R[0];
	xp[1] = R[1];
	xp[2] = R[2];

	yp[0] = R[3];
	yp[1] = R[4];
	yp[2] = R[5];

	zp[0] = R[6];
	zp[1] = R[7];
	zp[2] = R[8];
	
	//draw axes
	glLineWidth(linewidth);
	glBegin(GL_LINES);
	glColor3f(1, 0, 0); //'x' - axis
	glquiver3(p0, xp, scale * 0.5);
	glColor3f(0, 1, 0); //'y' - axis
	glquiver3(p0, yp, scale * 0.5);
	glColor3f(0, 0, 1); //'z' - axis
	glquiver3(p0, zp, scale);
	glEnd();

	double quad[4][3];
	add(p0, zp, quad[0], 1.0, 1 * scale);
	add(quad[0], xp, quad[1], 1.0, 0.8 * scale);
	add(quad[1], yp, quad[0], 1.0, 0.8 * scale);

	glColor3f(cr, cg, cb);

	glBegin(GL_LINE_STRIP);
	glpt3(quad[0]);
	add(quad[0], xp, quad[1], 1.0, -1.6 * scale);
	glpt3(quad[1]);
	add(quad[1], yp, quad[2], 1.0, -1.6 * scale);
	glpt3(quad[2]);
	add(quad[2], xp, quad[3], 1.0, 1.6 * scale);
	glpt3(quad[3]);
	glpt3(quad[0]);
	glEnd();

	glBegin(GL_LINES);
	glline(p0, quad[0]);
	glline(p0, quad[1]);
	glline(p0, quad[2]);
	glline(p0, quad[3]);
	glEnd();
}
