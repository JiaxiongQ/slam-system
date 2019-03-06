#include "shader_util.h"
//fuction for shader_tuil

int printOglError( char *file, int line )
{
    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    while (glErr != GL_NO_ERROR)
    {
        printf("glError 0x%x file %s @ %d: %s.\n", glErr, file, line, gluErrorString(glErr));
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}
void printShaderInfoLog( GLuint shader )
{
    int infologLength = 0;
    int charsWritten  = 0;
    GLchar *infoLog;

    printOpenGLError();  // Check for OpenGL errors.

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

    printOpenGLError();  // Check for OpenGL errors.

    if (infologLength > 0)
    {
        infoLog = (GLchar *)malloc(infologLength);
        if (infoLog == NULL)
        {
            printf("ERROR: Could not allocate InfoLog buffer.\n");
            exit(1);
        }
        glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
        printf("Shader InfoLog:\n%s\n\n", infoLog);
        free(infoLog);
    }
    printOpenGLError();  // Check for OpenGL errors.
}
void printProgramInfoLog( GLuint program )
{
    int infologLength = 0;
    int charsWritten  = 0;
    GLchar *infoLog;

    printOpenGLError();  // Check for OpenGL errors.

    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infologLength);

    printOpenGLError();  // Check for OpenGL errors.

    if (infologLength > 0)
    {
        infoLog = (GLchar *)malloc(infologLength);
        if (infoLog == NULL)
        {
            printf("ERROR: Could not allocate InfoLog buffer.\n");
            exit(1);
        }
        glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
        printf("Program InfoLog:\n%s\n\n", infoLog);
        free(infoLog);
    }
    printOpenGLError();  // Check for OpenGL errors.
}
int fileSize( const char *fileName )
{
    int fd;
    int count = -1;

    // Open the file, seek to the end to find its length.
    count=10; 
    return count;
}
int readShader( const char *fileName, char *shaderText, int size )
{
    FILE *fh;
    int count;

    // Open the file.
    fh = fopen(fileName, "r");
    if (!fh) return 0;

    // Get the shader from a file.
    fseek(fh, 0, SEEK_SET);
    count = (int) fread(shaderText, 1, size, fh);
    shaderText[count] = '\0';

    if (ferror(fh)) count = 0;

    fclose(fh);
    return count;
}
int readShaderSource( const char *fileName, GLchar **shaderSource )
{
    // Allocate memory to hold the source of the shader.

    int fsize = fileSize(fileName);

    if (fsize == -1)
    {
        printf("Cannot determine size of the shader %s.\n", fileName);
        return 0;
    }

    *shaderSource = (GLchar *) malloc(fsize + 1);  // Extra byte for null character.
    if (*shaderSource == NULL)
    {
        printf("Cannot allocate memory for shader source.\n");
        return 0;
    }

    // Read the source code.

    if (!readShader(fileName, *shaderSource, fsize + 1))
    {
        printf("Cannot read the file %s.\n", fileName);
        return 0;
    }

    return 1;
}
GLint getUniLoc( GLuint program, const GLchar *name )
{
    GLint loc;

    loc = glGetUniformLocation(program, name);

    if (loc == -1)
        printf("No such uniform named \"%s\".\n", name);

    printOpenGLError();  // Check for OpenGL errors.
    return loc;
}
GLuint makeShaderProgram( const GLchar *vertShaderSrc, const GLchar *fragShaderSrc )
{
	GLuint vShader, fShader, prog;   // handles to objects.
	GLint  vCompiled, fCompiled;     // status values.
	GLint  linked;

	// Create and compile the vertex shader object.
	if ( vertShaderSrc != NULL )
	{
		vShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vShader, 1, &vertShaderSrc, NULL);
		glCompileShader(vShader);
		printOpenGLError();  // Check for OpenGL errors.
		glGetShaderiv(vShader, GL_COMPILE_STATUS, &vCompiled);
		printShaderInfoLog(vShader);
		if (!vCompiled ) return 0;
	}

	// Create and compile the fragment shader object.
	if ( fragShaderSrc != NULL )
	{
		fShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fShader, 1, &fragShaderSrc, NULL);
		glCompileShader(fShader);
		printOpenGLError();  // Check for OpenGL errors.
		glGetShaderiv(fShader, GL_COMPILE_STATUS, &fCompiled);
		printShaderInfoLog(fShader);
		if (!fCompiled ) return 0;
	}

	// Create a program object and attach the two compiled shaders.
	prog = glCreateProgram();
	if ( vertShaderSrc != NULL ) glAttachShader(prog, vShader);
	if ( fragShaderSrc != NULL ) glAttachShader(prog, fShader);

	// Link the program object.
	glLinkProgram(prog);
    printOpenGLError();  // Check for OpenGL errors.
	glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    printProgramInfoLog(prog);

	if (!linked) return 0;

	return prog;
} 
void myBindAttribLocations( GLuint prog )
{    
	// Example:
	/*
    glBindAttribLocation( prog, 1, "myAttrib_A");
	printOpenGLError();  // Check for OpenGL errors.
    glBindAttribLocation( prog, 2, "myAttrib_B");
	printOpenGLError();  // Check for OpenGL errors.
	*/
}
void mySetUniformValues( GLuint prog )
{
	// Example:
	/*
	glUniform4f( getUniLoc(prog, "Background"), 0.0, 0.0, 0.0, 1.0);
	printOpenGLError();  // Check for OpenGL errors.

	glUniform1f( getUniLoc(prog, "CoolestTemp"), 0.0);
	printOpenGLError();  // Check for OpenGL errors.

	glUniform1f( getUniLoc(prog, "TempRange"), 100.0);
	printOpenGLError();  // Check for OpenGL errors.

	glUniform3f( getUniLoc(prog, "CoolestColor"), 0.0, 0.0, 1.0);
	printOpenGLError();  // Check for OpenGL errors.

	glUniform3f( getUniLoc(prog, "HottestColor"), 1.0, 0.0, 0.0);
	printOpenGLError();  // Check for OpenGL errors.
	*/
}