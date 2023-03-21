/*
 * Copyright 2022 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/minimal_logging.h"

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

  constexpr char kHelp[] = "help";
  constexpr char kDeviceName[] = "device_name";
  constexpr char kTimeout[] = "timeout";
  constexpr char kEnableCycleCounter[] = "enable_cycle_counter";
  constexpr char kPmuEvent0[] = "pmu_event0";
  constexpr char kPmuEvent1[] = "pmu_event1";
  constexpr char kPmuEvent2[] = "pmu_event2";
  constexpr char kPmuEvent3[] = "pmu_event3";
  constexpr char kCacheFilePath[] = "cache_file_path";

  bool show_help = false;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kHelp,
                               &show_help,
                               "Show the help information."),
      tflite::Flag::CreateFlag(kDeviceName, &options.device_name,
                               "Device name for ethosu."),
      tflite::Flag::CreateFlag(kCacheFilePath, &options.cache_file_path,
                               "Set the path if need to save/load vela binary."),
      tflite::Flag::CreateFlag(kTimeout, &options.timeout,
                               "Timeout in nanoseconds for inferencing."),
      tflite::Flag::CreateFlag(kEnableCycleCounter,
                               &options.enable_cycle_counter,
                               "If enable cycle counter when inference."),
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
  if (!tflite::Flags::Parse(&argc, argv.data(), flag_list) || show_help) {
      std::string usage = Flags::Usage(argv[0], flag_list);
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, usage.c_str());
      return nullptr;
  }

  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: device_name set to %s.",
                   options.device_name.c_str());
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: cache_file_path set to %s.",
                   options.cache_file_path.c_str());
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: timeout set to %ld.",
                   options.timeout);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: enable_cycle_counter set to %d.",
                   options.enable_cycle_counter);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: pmu_event0 set to %d.",
                   options.pmu_counter_config[0]);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: pmu_event1 set to %d.",
                   options.pmu_counter_config[1]);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: pmu_event2 set to %d.",
                   options.pmu_counter_config[2]);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                   "Ethosu delegate: pmu_event3 set to %d.",
                   options.pmu_counter_config[3]);
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
