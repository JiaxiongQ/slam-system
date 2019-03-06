#include "Draw.h"
extern vector<cv::Mat> motions_Show;
extern vector<cv::Mat> rotations_Show;
extern int frameIdxP;
extern int frameIdxC;
void DrawPointClouds(){
	glPointSize(2);
	glBegin(GL_POINTS);
	for(int j=0;j <= frameIdxP;++j){
	  for(int i=0;i<vPointCloud[j].size();i++){	
		SColorPoint3D p = vPointCloud[j][i];
		glColor3f(p.R,p.G,p.B);
		glVertex3f(p.X,p.Y,p.Z);
	}
	}
	glEnd();
}

void DrawCameras()
{
  for(int i = 0;i <= frameIdxC;++i)
  {
    double *R = new double[9];
    double *C = new double[3];
    
    for(int j=0;j<9;++j)
    {
      R[j] =  rotations_Show[i].ptr<double>(j/3)[j%3];
    }
    for(int j = 0;j<3;++j)
    {
      C[j] =  motions_Show[i].ptr<double>(0)[j];
    }
    
    drawCameraPose(R,C,10,1.0,1.0,0.0,2.0);
  }
}

void DrawDynamicPtsCurve(){
	glLineWidth(3.0);
	glBegin(GL_LINES);
	glColor3f(0,0,1);
	
	for(int j=0;j <= frameIdxP;++j){
	for(int i=1;i<dypts[j].size();i++){
	  
	   glVertex3f(dypts[j][i].x,dypts[j][i].y,dypts[j][i].z);
	   glVertex3f(dypts[j][i-1].x,dypts[j][i-1].y,dypts[j][i-1].z);
	}
	glVertex3f(dypts[j][dypts[j].size()-1].x,dypts[j][dypts[j].size()-1].y,dypts[j][dypts[j].size()-1].z);
	if(j>0)
	  glVertex3f(dypts[j-1][dypts[j-1].size()-1].x,dypts[j-1][dypts[j-1].size()-1].y,dypts[j-1][dypts[j-1].size()-1].z);
	}
	glEnd();

	glColor3fv( objectColor );
	glPushMatrix();
	//glTranslatef(dypts[frameIdx].x,dypts[frameIdx].y,dypts[frameIdx].z);
	//glutSolidSphere(0.5,50,50);
	glPopMatrix();
}