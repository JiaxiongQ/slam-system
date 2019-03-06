#include "Viewer.h"
#include "define.h"
TrackBall tb( GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON );
bool drawAxes = true;         // Draw world coordinate frame axes iff true.
bool drawWireframe = true;     // Draw polygons in wireframe if true, otherwise polygons are filled.
bool usePhongShading = true;   // Use Phong shading shaders if true, otherwise use Gouraud shading.

int winWidth = 1024;     // Window width in pixels.
int winHeight = 746;    // Window height in pixels.
int frameIdxP = 0;
int frameIdxC = 0;
vector<vector<SColorPoint3D>> vPointCloud;

vector< vector<cv::Point3f> > dypts;
//创建一个互斥锁
CMutex g_Lock;
mutex mtx;
static void MyMotion( int x, int y )
{
	tb.motion(x, y);
    glutPostRedisplay();
}

void DrawAxes( double length )
{
    glPushAttrib( GL_ALL_ATTRIB_BITS );
    glDisable( GL_LIGHTING );
    glLineWidth( 3.0 );
    glBegin( GL_LINES );
        // x-axis.
        glColor3f( 1.0, 0.0, 0.0 );
        glVertex3d( 0.0, 0.0, 0.0 );
        glVertex3d( length, 0.0, 0.0 );
        // y-axis.
        glColor3f( 0.0, 1.0, 0.0 );
        glVertex3d( 0.0, 0.0, 0.0 );
        glVertex3d( 0.0, -length, 0.0 );
        // z-axis.
        glColor3f( 0.0, 0.0, 1.0 );
        glVertex3d( 0.0, 0.0, 0.0 );
        glVertex3d( 0.0, 0.0, length );
    glEnd();
    glPopAttrib();
}

void MyDisplay( void )
{	
	if(backColor)
	  glClearColor (1.0,1.0,1.0,1.0);
	else 
	  glClearColor (0.0,0.0,0.0,1.0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluPerspective( 80.0,(double)winWidth/winHeight, OBJECT_RADIUS, 100.0 * OBJECT_RADIUS );

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	// Put point light source in eye space. 
	glLightfv( GL_LIGHT0, GL_POSITION, light0Position);
	gluLookAt( 0.0, 0.0, 100,0,0,0,0,1,0);
	glPushMatrix();
	
	tb.applyTransform();
	DrawPointClouds();   
	DrawCameras();
	//DrawDynamicPtsCurve();

	glColor3fv( objectColor );
	glPopMatrix();
	glutSwapBuffers();
}
  
void MyIdle(void)
{
  if(startFlag==true)
  {
    sem_wait(&sem);
    if(frameIdxP< NP - 1)
    {
      frameIdxP++;
    }
    glutPostRedisplay();
  }
}

void MyTimer(int value)
{
  if(startFlag==true)
  {
    if(frameIdxC < NC - 1)
    {
      frameIdxC++;
      if(frameIdxC == widSize + frameIdxP*step && frameIdxP < NP - 1)
	frameIdxP++;
    }
    glutPostRedisplay();
  }
  glutTimerFunc(100,MyTimer,0);
}
  
void MyKeyboard(unsigned char key, int x, int y )
{
	switch ( key )
    {
        case 'f':
		case 'F':
			glutPostRedisplay();
            break;

        // Toggle axes.
        case 'x':
        case 'X': 
            glutPostRedisplay();
            break;

        case 's':
		case 'S':
			//tb.save("openGLpose.txt");
		    break;

		case 'l':
		case 'L':
			tb.load("openGLpose.txt");
			tb.applyTransform();
			glutPostRedisplay();
			break;
		
		case 'c':
		case 'C':
			glutPostRedisplay();
            break;

		case 'g':
		case 'G':
			glutPostRedisplay();
            break;
			
		case 'q':
        case 'Q': 
			  glutPostRedisplay();
			break;

		case 'e':
        case 'E': 	  
	  if(frameIdxC < NC - 1)
    {
      frameIdxC++;
      if(frameIdxC == widSize + frameIdxP*step && frameIdxP < NP - 1)
	frameIdxP++;
    }
			  
            break;
		
		default:	
        break;
	}
}

void MyReshape( int w, int h )
{
	tb.reshape( w, h );
    winWidth = w;
    winHeight = h;
    glViewport( 0, 0, w, h );
}

static void MyMouse( int button, int state, int x, int y )
{
	tb.mouse( button, state, x, y );
	glutPostRedisplay();
}

void MyInit( void )
{
    glClearColor( 0.0, 0.0, 0.0, 1.0 ); // Set black background color.
    glEnable( GL_DEPTH_TEST ); // Use depth-buffer for hidden surface removal.
    glShadeModel( GL_SMOOTH );

    // Set Light 0.
    glLightfv( GL_LIGHT0, GL_AMBIENT, light0Ambient );
    glLightfv( GL_LIGHT0, GL_DIFFUSE, light0Diffuse );
    glLightfv( GL_LIGHT0, GL_SPECULAR, light0Specular );
    glEnable( GL_LIGHT0 );
    glEnable( GL_LIGHTING );

	// Set some global light properties.
    GLfloat globalAmbient[] = { 0.1, 0.1, 0.1, 1.0 };
    glLightModelfv( GL_LIGHT_MODEL_AMBIENT, globalAmbient );
    glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE );
    glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE );

    // Set the universal material properties.
    // The diffuse and ambient components can be changed using glColor*().
    glMaterialfv( GL_FRONT, GL_SPECULAR, materialSpecular );
    glMaterialfv( GL_FRONT, GL_SHININESS, materialShininess );
    glMaterialfv( GL_FRONT, GL_EMISSION, materialEmission );
    glColorMaterial( GL_FRONT, GL_AMBIENT_AND_DIFFUSE );
    glEnable( GL_COLOR_MATERIAL );

    glEnable( GL_NORMALIZE ); // Let OpenGL automatically renomarlize all normal vectors.

    tb.applyTransform();
}

void View3D(int argc,char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode ( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowSize( winWidth, winHeight );
	glutCreateWindow( "3D points" );
	MyInit();
    
	glutDisplayFunc( MyDisplay ); 
	//glutTimerFunc(100,MyTimer,0);
	//glutIdleFunc(MyIdle);
        glutReshapeFunc( MyReshape );
        glutKeyboardFunc( MyKeyboard );
	glutMouseFunc( MyMouse );
	glutMotionFunc( MyMotion );
	//glutMainLoop();
	int tempC,tempP;
	bool flag = false; 
	while(1)
	{
	   if(startFlag)
	   {
	    flag = true;
	    std::unique_lock<std::mutex> lck (mtx);
	    if(frameIdxC <= NC)
	    {
	      glutMainLoopEvent();
	      glutPostRedisplay(); 
	      if(showFeature)
	      {
		cv::imshow("images", imgs[frameIdxC]);
		cv::waitKey(40);
	      }
	      if(frameIdxC < NC)
		frameIdxC++;
	      if(frameIdxC == widSize + (frameIdxP+1)*step && frameIdxP < NP - 1)
		frameIdxP++;
	    }
	    
	  }
	  else if(flag)
	  {
	    break;
	  }
	}
}

void delay_msec(int msec)  
{   
    clock_t now = clock();  
    while(clock()-now < msec)
    {
      glutMainLoopEvent();
    };  
}  
    
void View3D(int argc,char** argv,vector< vector<SColorPoint3D> > &vP)
{
	vPointCloud = vP;
    
	glutInit(&argc,argv);
	glutInitDisplayMode ( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowSize( winWidth, winHeight );
	glutCreateWindow( "3D points" );
	MyInit();
    
	glutDisplayFunc( MyDisplay ); 
	//glutTimerFunc(100,MyTimer,0);
	//glutIdleFunc(MyIdle);
        glutReshapeFunc( MyReshape );
        glutKeyboardFunc( MyKeyboard );
	glutMouseFunc( MyMouse );
	glutMotionFunc( MyMotion );
	//glutMainLoop();
	while(1)
	{
	   glutMainLoopEvent();
	   cv::imshow("images", imgs[frameIdxC]);
	   cv::waitKey(20);
	   glutPostRedisplay();
	}
}



void View3D(int argc,char** argv,vector<vector<SColorPoint3D>> &vP,vector< vector<cv::Point3f> > &ds)
{
	vPointCloud = vP;

	dypts = ds;

	glutInit(&argc,argv);
    glutInitDisplayMode ( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
    glutInitWindowSize( winWidth, winHeight );
	glutCreateWindow( "3D points" );
	
	MyInit();
    
	glutDisplayFunc( MyDisplay ); 
    glutReshapeFunc( MyReshape );
    glutKeyboardFunc( MyKeyboard );
	glutMouseFunc( MyMouse );
	glutMotionFunc( MyMotion );

    glutMainLoop();
}


