cmake_minimum_required(VERSION 3.20.0)

project(xir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fno-gnu-unique)
endif()
add_compile_options(-fno-operator-names)

set(XIR_SRC_DIR ${CMAKE_SOURCE_DIR})
set(XIR_BIN_DIR ${CMAKE_BINARY_DIR})

# 用于设置静态库(static libraries, .a 文件)的输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${XIR_BINARY_DIR}/lib)
# 用于设置动态库(shared libraries, .so 文件)的输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${XIR_BINARY_DIR}/lib)
# 用于设置可执行文件(executable files, .exe 文件)的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${XIR_BINARY_DIR}/bin)

set(XIR_SRC_INCLUDE_DIR ${XIR_SRC_DIR}/include)
set(XIR_BIN_INCLUDE_DIR ${XIR_BIN_DIR}/include)
set(XIR_BIN_LIB_DIR ${XIR_BIN_DIR}/lib)
set(XIR_EXTERNAL_DIR ${XIR_SRC_DIR}/external)

#-------------------------------------------------------------------------------
# LLVM&MLIR configuration
#-------------------------------------------------------------------------------

option(LLVM_INSTALL_PATH "The path to the built LLVM library")
get_filename_component(LLVM_INSTALL_PATH ${LLVM_INSTALL_PATH} ABSOLUTE)

set(LLVM_INSTALL_DIR "${LLVM_INSTALL_PATH}")
set(LLVM_TOOLS_DIR "${LLVM_INSTALL_PATH}/bin")

set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir")

include(cmake/MLIR.cmake)
# include(cmake/StableHLO.cmake)

#-------------------------------------------------------------------------------
# Include directories configuration
#-------------------------------------------------------------------------------

include_directories(${XIR_SRC_INCLUDE_DIR})
include_directories(${XIR_BIN_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# XIR Options
#-------------------------------------------------------------------------------

option(XIR_ENABLE_STRICT_BUILD "Build XIR with strict warnings and warnings as errors" OFF)
option(XIR_ENABLE_LLD "Use LLD as the linker if available" OFF)
option(XIR_ENABLE_BINDINGS_PYTHON "Enables XIR Python bindings" OFF)

#-------------------------------------------------------------------------------
# Strict mode configuration
#-------------------------------------------------------------------------------

if (XIR_ENABLE_STRICT_BUILD)
  set(LLVM_ENABLE_WARNINGS ON)
  set(LLVM_ENABLE_WERROR ON)
  set(LLVM_ENABLE_PEDANTIC ON)
endif()

#-------------------------------------------------------------------------------
# Linker configuration
#-------------------------------------------------------------------------------

if (XIR_ENABLE_LLD)
  message(STATUS "Enabling LLD as the linker")
  add_link_options("-fuse-ld=lld")
endif()

#-------------------------------------------------------------------------------
# Python configuration
#-------------------------------------------------------------------------------

if(XIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include/xir)
add_subdirectory(lib)
# add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(examples)


install(DIRECTORY ${XIR_SRC_INCLUDE_DIR}/xir
  DESTINATION include
  COMPONENT xir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
)

install(DIRECTORY ${XIR_BIN_INCLUDE_DIR}/xir
  DESTINATION include
  COMPONENT xir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
)

add_llvm_install_targets(install-xir-headers
                        DEPENDS xir-headers
                        COMPONENT xir-headers)
