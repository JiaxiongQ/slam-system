SET(SOURCE_FILES 
BA.cpp
gridTracker.cpp
MycalcopticalFlow.cpp
reconstruct.cpp
Tools.cpp
Video.cpp
Draw.cpp
GLHelper.cpp
shader_util.cpp
TrackerBall.cpp
Viewer.cpp
lock.cpp
)
set(OpenCV_DIR "/usr/local/opencv/3.1.0/share/OpenCV")

find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(Eigen3 REQUIRED) 
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

INCLUDE_DIRECTORIES(
		    ${OPENGL_INCLUDE_DIR}
		    ${GLUT_INCLUDE_DIR}
		    ${GLEW_INCLUDE_DIR}
		    ${EIGEN3_INCLUDE_DIR}
		    ${OpenCV_INCLUDE_DIRS}
		    ${CERES_INCLUDE_DIRS}
		    ${PROJECT_SOURCE_DIR}/include
		   )
	      
add_library(association ${SOURCE_FILES})
target_link_libraries(
association 
${OpenCV_LIBS} 
${CERES_LIBRARIES} 
${OPENGL_LIBRARIES} 
${GLUT_glut_LIBRARY} 
${GLEW_LIBRARY}
)

