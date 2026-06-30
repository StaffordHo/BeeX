#pragma once

// Docker-only compatibility shim for the provided simulator source.
// The original bx_grid_world.cpp includes <opencv2/opencv.hpp>. The slim local
// Docker image installs only the OpenCV modules used by the simulator, so this
// shim maps that umbrella include to the required module headers without
// modifying BeeX's provided source file.

#include_next <opencv2/core.hpp>
#include_next <opencv2/highgui.hpp>
#include_next <opencv2/imgproc.hpp>
