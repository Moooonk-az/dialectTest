##===- CMakeLists.txt - dialectTest cmake root -----------------*- cmake -*-===//
##
## Configure the dialectTest build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(dialectTest LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "dialectTest two-step build.")
    # Two-step build
    #-------------------------------------------------------------------------------
    # MLIR/LLVM Configuration
    #-------------------------------------------------------------------------------
    find_package(MLIR REQUIRED CONFIG)
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    set(LLVM_MLIR_BINARY_DIR ${MLIR_DIR}/../../../bin)
    set(LLVM_MLIR_LIBRARY_DIR ${MLIR_DIR}/../../../lib)
    set(LLVM_PROJECT_BUILD_DIR ${MLIR_DIR}/../../../)
    if(NOT DEFINED LLVM_PROJECT_SOURCE_DIR)
      get_filename_component(LLVM_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm/ ABSOLUTE)
    endif()
    set(LLVM_MLIR_SOURCE_DIR ${LLVM_PROJECT_SOURCE_DIR}/mlir)

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

    find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
            NO_DEFAULT_PATH)

    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
else()
    message(STATUS "dialectTest one-step build.")
    # one-step build with LLVM_EXTERNAL_PROJECTS=dialectTest
    #-------------------------------------------------------------------------------
    # MLIR/LLVM Configuration
    #-------------------------------------------------------------------------------

    # Allow using out-of-tree llvm directory
    set(LLVM_PROJECT_SOURCE_DIR ${LLVM_MAIN_SRC_DIR}/..)
    message(STATUS "Using LLVM Project ${LLVM_PROJECT_SOURCE_DIR}")

    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
    set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
    set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
    set(LLVM_MLIR_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
    set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

#-------------------------------------------------------------------------------
# DTST configuration
#-------------------------------------------------------------------------------

# DTST project.
set(DTST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DTST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DTST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(DTST_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DTST_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${DTST_LIBRARY_DIR})

# Generate libraries into `lib` of build directory.
set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Add DTST files to the include path
include_directories(${DTST_MAIN_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${DTST_SOURCE_DIR}/lib)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Configure CMake.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(src)
