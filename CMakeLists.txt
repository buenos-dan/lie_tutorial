cmake_minimum_required(VERSION 3.5.0)
project(Lie_test)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(eigen)
add_subdirectory(Sophus)

add_executable(test lie_theory.cpp)
target_link_libraries(test
    eigen
    sophus
)

