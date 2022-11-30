#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
)

add_library(ethosu_delegate SHARED ${ETHOSU_DELEGATES_SRCS})

target_link_libraries(ethosu_delegate ${ETHOSU_DELEGATE_DEPENDENCIES})