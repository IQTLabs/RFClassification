cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(gamutrf_inference gamutrf_inference.cpp)
target_link_libraries(gamutrf_inference "${TORCH_LIBRARIES}")
set_property(TARGET gamutrf_inference PROPERTY CXX_STANDARD 14)