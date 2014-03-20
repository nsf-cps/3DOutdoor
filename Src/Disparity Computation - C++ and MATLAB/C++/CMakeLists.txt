cmake_minimum_required(VERSION 2.6)
project (CompDisparityConfidence)

FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( OpenMP REQUIRED )

# add the binary tree to the search path for include file
include_directories("${PROJECT_BINARY_DIR}")
#include_directories()

# add the executable
add_executable (CompDisparityConfidence Main_CompDisparityConfidence.cpp stereosgbm_cost.cpp CompConfidence.hpp)
target_link_libraries (CompDisparityConfidence ${OpenCV_LIBS})

# Compiler flags
set(CMAKE_BUILD_TYPE Release)
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3 -funroll-loops")        ## Optimize
endif()

# Where the executable is place
set(EXECUTABLE_OUTPUT_PATH ../Bin)
