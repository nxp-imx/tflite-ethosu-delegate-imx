#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(tflite_ethosu_delegate)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH}
)

find_package(tensorflow REQUIRED)

list(APPEND ETHOSU_DELEGATES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_drv.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_delegate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_delegate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/external_delegate_adaptor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_delegate_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ethosu_supported_ops.cc
)

find_package (Python COMPONENTS Interpreter Development)
get_filename_component(PYTHON_LIB_NAME ${Python_LIBRARIES} NAME)
include_directories(${Python_INCLUDE_DIRS})
list(APPEND ETHOSU_DELEGATE_DEPENDENCIES ${Python_LIBRARIES})

add_library(ethosu_delegate SHARED ${ETHOSU_DELEGATES_SRCS})
target_link_libraries(ethosu_delegate ${ETHOSU_DELEGATE_DEPENDENCIES})
target_compile_options(ethosu_delegate PUBLIC "-DPYTHON_LIB=${PYTHON_LIB_NAME}")
