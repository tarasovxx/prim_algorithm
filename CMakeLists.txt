cmake_minimum_required(VERSION 3.29)
project(prim_seminar)

set(CMAKE_C_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#add_executable(prim_seminar prim_sequential.c)
add_executable(prim_seminar_parallel prim_parallel_mpi.c)
#add_executable(prim_benchmark prim_benchmark.c)
#if(OpenMP_C_F)
add_executable(prim_seminar_omp openmp_native/prim_parallel_openmp.c)
add_executable(prim_seminar_mpi parallel_algo/prim_parallel.c)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
target_link_libraries(prim_seminar_parallel PRIVATE ${OpenMP_C_LIBRARIES})
target_link_libraries(prim_seminar_parallel PRIVATE MPI::MPI_C)

target_link_libraries(prim_seminar_omp PRIVATE ${OpenMP_C_LIBRARIES})
target_link_libraries(prim_seminar_mpi PRIVATE MPI::MPI_C)

