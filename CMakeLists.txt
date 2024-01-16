cmake_minimum_required(VERSION 3.20)
project(gpu_repartition LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED)

add_executable(gpu_repartition main.cu)
target_link_libraries(gpu_repartition MPI::MPI_CXX CUDA::cudart CCCL::CCCL)
