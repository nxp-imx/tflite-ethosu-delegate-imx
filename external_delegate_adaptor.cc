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

#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

#include "ethosu_delegate.h"

namespace tflite {
namespace ethosu {

TfLiteDelegate* CreateEthosuDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  EthosuDelegateOptions options = EthosuDelegateOptionsDefault();

  // Parse key-values options to EthosuDelegateOptions by mimicking them as
  // command-line flags.
  std::vector<const char*> argv;
  argv.reserve(num_options + 1);
  constexpr char kEthosuDelegateParsing[] = "ethosu_delegate_parsing";
  argv.push_back(kEthosuDelegateParsing);

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv.push_back(option_args.rbegin()->c_str());
  }

  constexpr char kDeviceName[] = "device_name";
  constexpr char kTimeout[] = "timeout";
  constexpr char kEnableCycleCounter[] = "enable_cycle_counter";
  constexpr char kPmuEvent0[] = "pmu_event0";
  constexpr char kPmuEvent1[] = "pmu_event1";
  constexpr char kPmuEvent2[] = "pmu_event2";
  constexpr char kPmuEvent3[] = "pmu_event3";

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kDeviceName, &options.device_name,
                               "Device name for ethosu."),
      tflite::Flag::CreateFlag(kTimeout, &options.timeout,
                               "Timeout in nanoseconds for inferencing."),
      tflite::Flag::CreateFlag(kEnableCycleCounter,
                               &options.enable_cycle_counter,
                               "If enbale cycle counter when inference."),
      tflite::Flag::CreateFlag(kPmuEvent0,
                               &options.pmu_counter_config[0],
                               "Pmu event 0."),
      tflite::Flag::CreateFlag(kPmuEvent1,
                               &options.pmu_counter_config[1],
                               "Pmu event 1."),
      tflite::Flag::CreateFlag(kPmuEvent2,
                               &options.pmu_counter_config[2],
                               "Pmu event 2."),
      tflite::Flag::CreateFlag(kPmuEvent3,
                               &options.pmu_counter_config[3],
                               "Pmu event 3."),
  };

  int argc = num_options + 1;
  if (!tflite::Flags::Parse(&argc, argv.data(), flag_list)) {
      return nullptr;
  }

  TFLITE_LOG(INFO) << "Ethosu delegate: device_name set to "
                   << options.device_name << ".";
  TFLITE_LOG(INFO) << "Ethosu delegate: timeout set to "
                   << options.timeout << ".";
  TFLITE_LOG(INFO) << "Ethosu delegate: enable_cycle_counter set to "
                   << options.enable_cycle_counter << ".";
  TFLITE_LOG(INFO) << "Ethosu delegate: pmu_event0 set to "
                   << options.pmu_counter_config[0] << ".";
  TFLITE_LOG(INFO) << "Ethosu delegate: pmu_event1 set to "
                   << options.pmu_counter_config[1] << ".";
  TFLITE_LOG(INFO) << "Ethosu delegate: pmu_event2 set to "
                   << options.pmu_counter_config[2] << ".";
  TFLITE_LOG(INFO) << "Ethosu delegate: pmu_event3 set to "
                   << options.pmu_counter_config[3] << ".";
  return EthosuDelegateCreate(&options);
}

}  // namespace ethosu
}  // namespace tflite

extern "C" {

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::ethosu::CreateEthosuDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  EthosuDelegateDelete(delegate);
}

}  // extern "C"