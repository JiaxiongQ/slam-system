#pragma once
#include "TrackerBall.h"

//function definition for shader_tuil.h
#ifndef _SHADER_UTIL_
#define _SHADER_UTIL_
#define printOpenGLError()  printOglError( __FILE__, __LINE__ )
int printOglError( char *file, int line );
void printShaderInfoLog( GLuint shader );
void printProgramInfoLog( GLuint program );
int fileSize( const char *fileName );
int readShader( const char *fileName, char *shaderText, int size );
int readShaderSource( const char *fileName, GLchar **shaderSource );
GLint getUniLoc( GLuint program, const GLchar *name );
GLuint makeShaderProgram( const GLchar *vertShaderSrc, const GLchar *fragShaderSrc ); 
void myBindAttribLocations( GLuint prog );
void mySetUniformValues( GLuint prog );
#endif
