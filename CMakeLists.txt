cmake_minimum_required(VERSION 3.5.0)
project(Lie_test)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(eigen)
add_subdirectory(Sophus)

add_executable(derivation_of_rotation_matrix derivation_of_rotation_matrix.cpp)
target_link_libraries(derivation_of_rotation_matrix
    eigen
    sophus
)

add_executable(derivation_of_quaternion derivation_of_quaternion.cpp)
target_link_libraries(derivation_of_quaternion
    eigen
    sophus
)

