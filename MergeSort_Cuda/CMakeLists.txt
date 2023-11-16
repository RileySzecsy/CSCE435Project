cmake_minimum_required(VERSION 3.12)

project(cudamergesort_2 LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

add_executable(cudamergesort_2 cudamergesort_2.cu)

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(cudamergesort_2 PRIVATE MPI::MPI_CXX)
target_link_libraries(cudamergesort_2 PRIVATE caliper)