cmake_minimum_required(VERSION 3.9...3.18)
project(BaryTree VERSION 1.0 LANGUAGES C CXX)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)

    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                "Debug" "Release" "None")
endif()
 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

option(BUILD_EXAMPLES "build examples"                          ON)
option(ENABLE_CUDA_BUILD "build OpenACC GPU version"            ON)
option(BUILD_SHARED_LIBS "build libraries as shared objects"	ON)

find_package(OpenMP REQUIRED)

#Creating an interface for OpenMP
#Some PGI versions seem to require a link flag for OpenMP
set_property(TARGET OpenMP::OpenMP_C APPEND
                                     PROPERTY
                                         INTERFACE_LINK_LIBRARIES
                                         $<$<C_COMPILER_ID:PGI>:-mp>) 

if(ENABLE_CUDA_BUILD)
    enable_language(CUDA)
    set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
endif()

add_subdirectory(src)
add_subdirectory(examples)
