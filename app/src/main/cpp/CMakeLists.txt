cmake_minimum_required(VERSION 3.22.1)

project("sfm_vc_ocv")

# -------------------------------
# OpenCV configuration
# -------------------------------
# This path MUST point to OpenCVConfig.cmake in your OpenCV SDK
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../../OpenCV/native/jni/abi-arm64-v8a")


find_package(OpenCV)

# -------------------------------
# Native library
# -------------------------------
add_library(
        native-lib
        SHARED
        native-lib.cpp
)

# Find Android log library
find_library(
        log-lib
        log
)

# -------------------------------
# Link libraries
# -------------------------------
target_link_libraries(
        native-lib
        ${OpenCV_LIBS}
        ${log-lib}
        android
)
