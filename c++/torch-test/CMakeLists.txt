cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch-test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torch-test test.cpp)
target_link_libraries(torch-test "${TORCH_LIBRARIES}")
set_property(TARGET torch-test PROPERTY CXX_STANDARD 17)
