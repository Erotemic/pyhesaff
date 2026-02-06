cmake_minimum_required(VERSION 3.15.0)
if(POLICY CMP0087)
  cmake_policy(SET CMP0087 NEW)
endif()


#######################################
# Clang2 = Clang + OpenMP built for OSX Mavericks
# http://goo.gl/1Tg0Zj
if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    message(STATUS "Detected APPLE system")
    SET(CLANG2 Off)
endif()

if (APPLE AND CLANG2)
    message(STATUS "Using clang2")
    set(CMAKE_C_COMPILER "clang2")
    set(CMAKE_CXX_COMPILER "clang2++")
endif()

#######################################
project(hesaff LANGUAGES C CXX) #### !!!!IMPORTANT!!!! THIS MUST BE DOWN HERE FOR CLANG2
#set(CMAKE_BUILD_TYPE "Release")

if (APPLE)
    #MacPorts
    message(STATUS "USING MACPORTS")
    # Fixme: newstyle include and link
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

#######################################
if(APPLE AND CLANG2)
    # Add flags to support clang2
    message(STATUS "APPLE + CLANG2: Adding stdlib flags for clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++")
else()
    # TODO: this is gcc only, fix for clang
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libc++")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lc++")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++")

# Setup basic python stuff and ensure we have skbuild
#list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/CMake")
#include( skbuild-helpers )

#######################################


# OpenCV_ROOT_DIR=$HOME/.local
# OpenCV_ROOT_DIR=$HOME/.local python setup.py develop
#set(OpenCV_FIND_REQUIRED_COMPONENTS)
if(DEFINED ENV{OpenCV_DIR} AND NOT OpenCV_DIR)
  set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif()
if(DEFINED ENV{OpenCV_ROOT} AND NOT OpenCV_ROOT)
  set(OpenCV_ROOT "$ENV{OpenCV_ROOT}")
endif()
if(DEFINED ENV{OpenCV_ROOT_DIR} AND NOT OpenCV_ROOT_DIR)
  set(OpenCV_ROOT_DIR "$ENV{OpenCV_ROOT_DIR}")
endif()
if(DEFINED ENV{VCPKG_ROOT} AND NOT OpenCV_DIR)
  if(DEFINED ENV{VCPKG_TARGET_TRIPLET})
    set(_vcpkg_triplet "$ENV{VCPKG_TARGET_TRIPLET}")
  elseif(WIN32)
    set(_vcpkg_triplet "x64-windows")
  endif()
  if(_vcpkg_triplet)
    set(_vcpkg_opencv_dir "$ENV{VCPKG_ROOT}/installed/${_vcpkg_triplet}/share/opencv4")
    set(OpenCV_DIR "${_vcpkg_opencv_dir}")
    if(NOT OpenCV_ROOT)
      set(OpenCV_ROOT "$ENV{VCPKG_ROOT}/installed/${_vcpkg_triplet}")
    endif()
  endif()
endif()
if(OpenCV_ROOT AND NOT CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "${OpenCV_ROOT}")
endif()
message(STATUS "OpenCV_DIR (cache/env) = ${OpenCV_DIR}")
message(STATUS "OpenCV_ROOT (cache/env) = ${OpenCV_ROOT}")
message(STATUS "OpenCV_ROOT_DIR (cache/env) = ${OpenCV_ROOT_DIR}")
message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")
message(STATUS "Derived VCPKG triplet = ${_vcpkg_triplet}")
message(STATUS "ENV{OpenCV_DIR} = $ENV{OpenCV_DIR}")
message(STATUS "ENV{OpenCV_ROOT} = $ENV{OpenCV_ROOT}")
message(STATUS "ENV{OpenCV_ROOT_DIR} = $ENV{OpenCV_ROOT_DIR}")
message(STATUS "ENV{VCPKG_ROOT} = $ENV{VCPKG_ROOT}")
message(STATUS "ENV{VCPKG_TARGET_TRIPLET} = $ENV{VCPKG_TARGET_TRIPLET}")
message(STATUS "ENV{VCPKG_INSTALLED_DIR} = $ENV{VCPKG_INSTALLED_DIR}")
message(STATUS "ENV{CMAKE_PREFIX_PATH} = $ENV{CMAKE_PREFIX_PATH}")
message(STATUS "ENV{PATH} = $ENV{PATH}")
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
set(_opencv_probe_dirs
    "${OpenCV_DIR}"
    "$ENV{OpenCV_DIR}"
    "${OpenCV_ROOT}"
    "$ENV{OpenCV_ROOT}"
    "${OpenCV_ROOT_DIR}"
    "$ENV{OpenCV_ROOT_DIR}"
    "$ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_TARGET_TRIPLET}/share/opencv4"
    "$ENV{VCPKG_ROOT}/installed/x64-windows/share/opencv4"
)
foreach(_opencv_probe_dir IN LISTS _opencv_probe_dirs)
  if(_opencv_probe_dir)
    message(STATUS "OpenCV probe dir: ${_opencv_probe_dir}")
    if(EXISTS "${_opencv_probe_dir}")
      file(GLOB _opencv_probe_entries "${_opencv_probe_dir}/*")
      list(LENGTH _opencv_probe_entries _opencv_probe_count)
      message(STATUS "  exists with ${_opencv_probe_count} entries")
      if(EXISTS "${_opencv_probe_dir}/OpenCVConfig.cmake")
        message(STATUS "  OpenCVConfig.cmake found")
      elseif(EXISTS "${_opencv_probe_dir}/opencv-config.cmake")
        message(STATUS "  opencv-config.cmake found")
      else()
        message(STATUS "  OpenCV config file not found in probe dir")
      endif()
    else()
      message(STATUS "  does not exist")
    endif()
  endif()
endforeach()
find_package(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  message(STATUS "Found OpenCV! ^_^")
  message(STATUS "OpenCV_FOUND = ${OpenCV_FOUND}")
  message(STATUS "OpenCV_INCLUDE_DIR = ${OpenCV_INCLUDE_DIR}")
  message(STATUS "OpenCV2_INCLUDE_DIRS = ${OpenCV2_INCLUDE_DIRS}")
  message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")
  message(STATUS "OpenCV_LINK_DIRECTORIES = ${OpenCV_LINK_DIRECTORIES}")
  message(STATUS "OpenCV_ROOT_DIR = ${OpenCV_ROOT_DIR}")

  # HACK ON CENTOS
  include_directories(/usr/local/include)

  set(_opencv_includes "${OpenCV_INCLUDE_DIRS}")
  if(NOT _opencv_includes)
    set(_opencv_includes "${OpenCV_INCLUDE_DIR}")
  endif()
ELSE()
  message(FATAL_ERROR "Missing OpenCV! x.x")
ENDIF()

#######################################
find_package(OpenMP)
IF(OPENMP_FOUND)
    message(STATUS "Found OpenMP! ^_^")
    # add flags for OpenMP
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
  message(STATUS "Missed OpenMP! x_x")
ENDIF()

option(ENABLE_GPROF Off)
IF(ENABLE_GPROF)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
ENDIF()

#######################################
set(HESAFF_CORE_SOURCES
    src/hesaff.cpp
    src/pyramid.cpp
    src/helpers.cpp
    src/affine.cpp
    src/orientation.cpp
    src/siftdesc.cpp)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "OpenMP_SHARED_LINKER_FLAGS = ${OpenMP_SHARED_LINKER_FLAGS}")
message(STATUS "OpenMP_EXE_LINKER_FLAGS = ${OpenMP_EXE_LINKER_FLAGS}")
message(STATUS "OpenCV_INCLUDE_DIR = ${OpenCV_INCLUDE_DIR}")
message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")

pybind11_add_module(_hesaff MODULE
    src/pybind_hesaff.cpp
    ${HESAFF_CORE_SOURCES}
)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(WIN32)
    target_compile_definitions(_hesaff PRIVATE HESAFF_WIN_EXPORT)
endif()

target_include_directories(_hesaff PRIVATE ${_opencv_includes} "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(_hesaff PRIVATE ${OpenCV_LIBRARIES})

install(
  TARGETS _hesaff
  RUNTIME DESTINATION "pyhesaff"
  LIBRARY DESTINATION "pyhesaff"
  ARCHIVE DESTINATION "pyhesaff"
)

option(BUILD_HESAFF_CLI "Build hesaff command line tool" OFF)
if(BUILD_HESAFF_CLI)
  add_executable(hesaff_cli src/hesaff_cli.cpp ${HESAFF_CORE_SOURCES})
  target_include_directories(hesaff_cli PRIVATE ${_opencv_includes} "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(hesaff_cli PRIVATE ${OpenCV_LIBRARIES})
endif()

#message(STATUS CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE})


# Annotate code with ctags
#set_source_files_properties( tags PROPERTIES GENERATED true)
#add_custom_command ( OUTPUT tags
#    COMMAND ctags -R --c++-kinds=+p --fields=+iaS --extra=+q .
#    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} )
#add_executable(hesaff_tags tags)
#add_executable(hesaffexe ${HESAFF_CORE_SOURCES})
#target_include_directories(hesaffexe PRIVATE ${OpenCV_INCLUDE_DIR})
#target_link_libraries(hesaffexe ${OpenCV_LIBRARIES})
