#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)
FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG v2.14.0
)

FetchContent_GetProperties(tensorflow)
if(NOT tensorflow_POPULATED)
  FetchContent_Populate(tensorflow)
endif()

add_subdirectory("${tensorflow_SOURCE_DIR}/tensorflow/lite"
                 "${tensorflow_BINARY_DIR}")
get_target_property(TFLITE_SOURCE_DIR tensorflow-lite SOURCE_DIR)

if(NOT TFLITE_LIB_LOC OR NOT EXISTS ${TFLITE_LIB_LOC})
  add_library(TensorFlow::tensorflow-lite ALIAS tensorflow-lite)
else()
  add_library(TensorFlow::tensorflow-lite UNKNOWN IMPORTED)
  set_target_properties(TensorFlow::tensorflow-lite PROPERTIES
    IMPORTED_LOCATION ${TFLITE_LIB_LOC}
    INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:tensorflow-lite,INTERFACE_INCLUDE_DIRECTORIES>
  )
  set_target_properties(tensorflow-lite PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

list(APPEND ETHOSU_DELEGATE_DEPENDENCIES TensorFlow::tensorflow-lite)
list(APPEND ETHOSU_DELEGATES_SRCS ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc)
