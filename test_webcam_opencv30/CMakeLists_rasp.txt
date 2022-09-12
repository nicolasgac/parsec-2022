#Déclaration du projet
project(camera_test)

#find_package(CUDA REQUIRED)

#link_directories(/usr/local/lib)

#set(OpenCV_DIR "/usr/local/lib/OpenCV/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)


#cuda_add_executable(camera_test ../main_webcam.cpp)
add_executable(camera_test ../main_rasp.cpp)

target_link_libraries(camera_test Qt5::Widgets)
target_link_libraries(camera_test Qt5::Core)
target_link_libraries(camera_test opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio)
