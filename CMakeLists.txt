cmake_minimum_required(VERSION 3.20)
project(opencv)
set(CMAKE_CXX_STANDARD 14)
#找到opencv的包
find_package( OpenCV REQUIRED )
#加入依赖库文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
#main.cpp 编译成opencvs
add_executable(main main.cpp 41canny边缘加强.cpp)
target_link_libraries(main ${OpenCV_LIBS} )

add_executable(41 41canny边缘加强.cpp)
target_link_libraries(41 ${OpenCV_LIBS} )