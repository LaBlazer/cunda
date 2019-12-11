cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

project(cunda)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/lib/opencv/build")

file(GLOB SRC_FILES
    "src/*.cpp"
	"src/*.cu"
)

message(STATUS "Sources: ${SRC_FILES}")
cuda_add_executable(cunda ${SRC_FILES})
target_link_libraries(cunda ${OpenCV_LIBS})
target_include_directories(cunda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(cunda PRIVATE ${CUDA_INCLUDE_DIRS})
# set_target_properties(cunda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file(GLOB DLL_FILES
    "${CMAKE_SOURCE_DIR}/lib/opencv/build/x64/vc15/bin/*.dll"
)

file(GLOB RESOURCE_FILES
    "${CMAKE_SOURCE_DIR}/resources/*.*"
)

message(STATUS "Resources: ${RESOURCE_FILES}")

add_custom_command(
    TARGET cunda
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLL_FILES}
        $<TARGET_FILE_DIR:cunda>
)

add_custom_command(
    TARGET cunda
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${RESOURCE_FILES}
        $<TARGET_FILE_DIR:cunda>
)