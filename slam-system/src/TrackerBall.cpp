#include "TrackerBall.h"

static void _tbPointToVector( int x, int y, int width, int height, GLdouble v[3] )
{
	GLdouble d, a;

	// project x, y onto a hemi-sphere centered within width, height.
	v[0] = (2.0 * x - width) / width;
	v[1] = (height - 2.0 * y) / height;
	d = sqrt(v[0] * v[0] + v[1] * v[1]);
	v[2] = cos((3.14159265 / 2.0) * ((d < 1.0) ? d : 1.0));
	a = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	v[0] *= a;
	v[1] *= a;
	v[2] *= a;
}
TrackBall::TrackBall( void )
{
	tb_rot_button = GLUT_LEFT_BUTTON;
	tb_pan_button = GLUT_MIDDLE_BUTTON;
	tb_zoom_button = GLUT_RIGHT_BUTTON;
    reset();
}
TrackBall::TrackBall( int rot_button, int pan_button, int zoom_button )
{
	if ( rot_button == pan_button || rot_button == zoom_button || pan_button == zoom_button )
	{
		tb_rot_button = GLUT_LEFT_BUTTON;
		tb_pan_button = GLUT_MIDDLE_BUTTON;
		tb_zoom_button = GLUT_RIGHT_BUTTON;
	}
	else
	{
		tb_rot_button = rot_button;
		tb_pan_button = pan_button;
		tb_zoom_button = zoom_button;
	}
    reset();
}
void TrackBall::reset( void )
{
	tb_angle = 0.0;
	tb_axis[0] = 1.0; tb_axis[1] = 0.0; tb_axis[2] = 0.0;
	tb_pan_x = 0.0;
	tb_pan_y = 0.0;
	tb_zoom = 1.0;
	tb_zoom_inc = 0.005;

	// put the identity in the trackball transform
    tb_transform[0] = 1.0; tb_transform[4] = 0.0; tb_transform[8] = 0.0; tb_transform[12] = 0.0;
	tb_transform[1] = 0.0; tb_transform[5] = 1.0; tb_transform[9] = 0.0; tb_transform[13] = 0.0;
	tb_transform[2] = 0.0; tb_transform[6] = 0.0; tb_transform[10] = 1.0; tb_transform[14] = 0.0;
	tb_transform[3] = 0.0; tb_transform[7] = 0.0; tb_transform[11] = 0.0; tb_transform[15] = 1.0;
}
void TrackBall::applyTransform( void )
{
	captureTransform();
	glPushMatrix();
	glLoadIdentity();
	glRotated( tb_angle, tb_axis[0], tb_axis[1], tb_axis[2] );
	glMultMatrixd( tb_transform );
	glGetDoublev( GL_MODELVIEW_MATRIX, tb_transform );
	glPopMatrix();

	glTranslated( tb_pan_x, tb_pan_y, 0.0 );
	glScaled( tb_zoom, tb_zoom, tb_zoom );
	glMultMatrixd( tb_transform );

	tb_angle = 0.0;
	tb_axis[0] = 1.0; tb_axis[1] = 0.0; tb_axis[2] = 0.0;
}
void TrackBall::captureTransform( void )
{
	glGetDoublev( GL_MODELVIEW_MATRIX, tb_model_mat );
	glGetDoublev( GL_PROJECTION_MATRIX, tb_proj_mat );
	glGetIntegerv( GL_VIEWPORT, tb_viewport);
}
void TrackBall::reshape( int width, int height )
{
	tb_width  = width;
	tb_height = height;
}
void TrackBall::mouse( int button, int state, int x, int y )
{
	if ( state == GLUT_DOWN ) tb_mouse_button = button;
	tb_mouse_x = x;
	tb_mouse_y = y;
}
void TrackBall::motion( int x, int y )
{
	GLdouble last_position[3], current_position[3], dx, dy, dz;
	GLdouble winx, winy, winz, tmp, old_pan_x, old_pan_y, new_pan_x, new_pan_y;

	tb_angle = 0.0;
	tb_axis[0] = 1.0; tb_axis[1] = 0.0; tb_axis[2] = 0.0;

	// rotating
	if ( tb_mouse_button == tb_rot_button )
	{
		_tbPointToVector( tb_mouse_x, tb_mouse_y, tb_width, tb_height, last_position );
		_tbPointToVector( x, y, tb_width, tb_height, current_position );

		// calculate the angle to rotate by (directly proportional to the
		// length of the mouse movement
		dx = current_position[0] - last_position[0];
		dy = current_position[1] - last_position[1];
		dz = current_position[2] - last_position[2];
		tb_angle = 90.0 * sqrt(dx * dx + dy * dy + dz * dz);

		// calculate the axis of rotation (cross product)
		tb_axis[0] = last_position[1] * current_position[2] - 
				     last_position[2] * current_position[1];
		tb_axis[1] = last_position[2] * current_position[0] - 
				     last_position[0] * current_position[2];
		tb_axis[2] = last_position[0] * current_position[1] - 
				     last_position[1] * current_position[0];
	}


	// panning
	else if ( tb_mouse_button == tb_pan_button )
	{
		gluProject( 0.0, 0.0, 0.0, tb_model_mat, tb_proj_mat, tb_viewport, &winx, &winy, &winz);
		gluUnProject( x, y, winz, tb_model_mat, tb_proj_mat, tb_viewport, 
					  &new_pan_x, &new_pan_y, &tmp );
		new_pan_y = -new_pan_y;

		gluUnProject( tb_mouse_x, tb_mouse_y, winz, tb_model_mat, tb_proj_mat, tb_viewport, 
					  &old_pan_x, &old_pan_y, &tmp );
		new_pan_y = -new_pan_y;

		tb_pan_x += (new_pan_x - old_pan_x);
		tb_pan_y -= (new_pan_y - old_pan_y);
	}


	// zooming
	else if ( tb_mouse_button == tb_zoom_button )
	{
		tb_zoom = tb_zoom + ( tb_mouse_y - y ) * tb_zoom_inc;
		if ( tb_zoom <= 0.0 ) tb_zoom = tb_zoom_inc;
	}

	tb_mouse_x = x;
	tb_mouse_y = y;
}
void TrackBall::save( const char *file )
{
	FILE *fp = fopen( file, "w" );
	if ( fp == NULL )
	{
		fprintf( fp, "TrackBall(): Cannot open file \"%s\" for writing.\n", file );
		return;
	}

	fprintf( fp, "%.10f\n", tb_angle );
	fprintf( fp, "%.10f %.10f %.10f\n", tb_axis[0], tb_axis[1], tb_axis[2] );

	for ( int i = 0; i < 16; i++ ) fprintf( fp, "%.10f ", tb_transform[i] );
	fprintf( fp, "\n" );

	fprintf( fp, "%.10f\n", tb_pan_x );
	fprintf( fp, "%.10f\n", tb_pan_y );
	fprintf( fp, "%.10f\n", tb_zoom );

	fclose( fp );
}
void TrackBall::load( const char *file )
{
	FILE *fp = fopen( file, "r" );
	if ( fp == NULL )
	{
		fprintf( fp, "TrackBall(): Cannot open file \"%s\" for reading.\n", file );
		return;
	}

	fscanf( fp, "%lf", &tb_angle );
	fscanf( fp, "%lf %lf %lf", &tb_axis[0], &tb_axis[1], &tb_axis[2] );

	for ( int i = 0; i < 16; i++ ) fscanf( fp, "%lf", &tb_transform[i] );

	fscanf( fp, "%lf", &tb_pan_x );
	fscanf( fp, "%lf", &tb_pan_y );
	fscanf( fp, "%lf", &tb_zoom );

	fclose( fp );
}