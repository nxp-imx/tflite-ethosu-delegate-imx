/*
 * Copyright 2022 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
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
#define METADATA_SIZE(buffer) ((reinterpret_cast<const uint32_t*>(buffer))[2])
#define METADATA_TO_OFFSET(data) reinterpret_cast<const int32_t*>(data + 12)

#define CMS_TENSOR_INDEX 0
#define FLASH_TENSOR_INDEX 1
#define SCRATCH_TENSOR_INDEX 2
#define SCRATCH_FAST_TENSOR_INDEX 3
#define INPUT_TENSOR_INDEX 4

#define BUFFER_ALIGNMENT 16
#define ALIGN_SIZE(size) ((size + BUFFER_ALIGNMENT - 1) & (~(BUFFER_ALIGNMENT - 1)))

#define DEFAULT_QREAD_BUFFER_SIZE 2048
typedef struct {
    uint64_t cycleCount;
    uint32_t qread;
    uint32_t status;
    struct {
        uint32_t eventConfig;
        uint32_t eventCount;
    } pmu[ETHOSU_PMU_EVENT_MAX];
} EthosuQreadEvent;

typedef struct {
  // Device name for ethosu.
  std::string device_name;
  // Timeout in nanoseconds for inferencing.
  int64_t timeout;
  //vela cache binary path
  std::string cache_file_path;
  // If enable cycle counter when inference.
  bool enable_cycle_counter;
  // Pmu counter config.
  int32_t pmu_counter_config[ETHOSU_PMU_EVENT_MAX];
  // If enable layer by layer profiling
  bool enable_profiling;
  // Qread buffer size for layer by layer profiling
  int32_t profiling_buffer_size;
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
