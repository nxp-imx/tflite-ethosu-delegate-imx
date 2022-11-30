/*
 * Copyright 2022 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_DELEGATES_ETHOSU_DELEGATES_H_
#define TENSORFLOW_LITE_DELEGATES_ETHOSU_DELEGATES_H_

#include <memory>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "ethosu_drv.h"

#define ETHOSU_DEFAULT_DEVICE_NAME (char*)"/dev/ethosu0"
#define ETHOSU_DEFAULT_TIMEOUT 60000000000
#define ETHOSU_CUSTOM_NAME "ethos-u"

#define OFFLINE_MEM_ALLOC_METADATA "OfflineMemoryAllocation"
#define METADATA_SIZE(number) ((number + 3) * 4)
#define METADATA_TO_OFFSET(data) reinterpret_cast<const int32_t*>(data + 12)

#define CMS_TENSOR_INDEX 0
#define FLASH_TENSOR_INDEX 1
#define SCRATCH_TENSOR_INDEX 2
#define SCRATCH_FAST_TENSOR_INDEX 3
#define INPUT_TENSOR_INDEX 4

#define BUFFER_ALIGNMENT 16
#define ALIGN_SIZE(size) ((size + BUFFER_ALIGNMENT - 1) & (~(BUFFER_ALIGNMENT - 1)))


typedef struct {
  // Device name for ethosu.
  std::string device_name;
  // Timeout in nanoseconds for inferencing.
  int64_t timeout;
  // If enbale cycle counter when inference.
  bool enable_cycle_counter;
  // Pmu counter config.
  int32_t pmu_counter_config[ETHOSU_PMU_EVENT_MAX];
} EthosuDelegateOptions;

// Returns a structure with the default delegate options.
EthosuDelegateOptions EthosuDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `EthosuDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* EthosuDelegateCreate(const EthosuDelegateOptions* options);

// Destroys a delegate created with `EthosuDelegateCreate` call.
void EthosuDelegateDelete(TfLiteDelegate* delegate);

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
EthosuDelegateCreateUnique(const EthosuDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      EthosuDelegateCreate(options), EthosuDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_ETHOSU_DELEGATES_H_
