/*
 * Copyright 2022 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <utility>
#include <string.h>
#include <vector>

#include "ethosu_delegate.h"
#include "simple_delegate.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"

using namespace std;
using namespace EthosU;

namespace tflite {
namespace ethosu {

// Ethosu delegate kernel.
class EthosuDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit EthosuDelegateKernel(const EthosuDelegateOptions& opt)
      : options(opt) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    //Get arena offset for each tensor from meta data
    const char* buffer = nullptr;
    size_t bytes;
    TF_LITE_ENSURE_OK(context, context->GetModelMetadata(context,
                      OFFLINE_MEM_ALLOC_METADATA, &buffer, &bytes));
    if (bytes != METADATA_SIZE(context->tensors_size)) {
        TF_LITE_KERNEL_LOG(context, "Failed to get address offsets from metadata\n");
        return kTfLiteDelegateError;
    }
    address_offsets = METADATA_TO_OFFSET(buffer);

    try {
        device = Device::GetSingleton(options.device_name.c_str());
    } catch (exception &e) {
        TF_LITE_KERNEL_LOG(context, "Failed to create ethos_u driver.\n");
        return kTfLiteDelegateError;
    }

    for (int i = 0; i < ETHOSU_PMU_EVENT_MAX; i ++) {
        if (options.pmu_counter_config[i] != 0)
            pmu_counter_config.push_back(options.pmu_counter_config[i]);
    }

    operations.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
        TfLiteNode* node;
        TfLiteRegistration* reg;
        int node_idx = params->nodes_to_replace->data[i];
        context->GetNodeAndRegistration(context, node_idx, &node, &reg);
        tflite::TfLiteIntArrayView inputs(node->inputs);
        tflite::TfLiteIntArrayView outputs(node->outputs);

        auto& op = operations[i];
        copy(inputs.begin(), inputs.end(), back_inserter(op.inputs));
        copy(outputs.begin(), outputs.end(), back_inserter(op.outputs));
    }

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    try {
        size_t arena_data_size = 0;

        for (auto& op : operations) {
            size_t tensor_count = op.inputs.size() + op.outputs.size() - 1; //cms tensor not included
            size_t layout_buffer_size = 2 * sizeof(uint32_t) + //Space for in/out tensor count
                             tensor_count * sizeof(uint32_t) + //Space for base_addr_size
                             tensor_count * sizeof(uint64_t);  //Space for the base_addr

            op.tensor_layout_buffer = make_shared<Buffer>(*device, layout_buffer_size);
            uint32_t *layout_data =reinterpret_cast<uint32_t*>(op.tensor_layout_buffer->data());
            uint32_t *base_addr_size = layout_data + 2;
            layout_data[0] = op.inputs.size() - 4;
            layout_data[1] = op.outputs.size();
            // Get command stream data size and create buffer
            auto cms_idx = op.inputs[CMS_TENSOR_INDEX];
            auto cms_tensor = &context->tensors[cms_idx];
            size_t cms_data_size = cms_tensor->bytes;

            op.net_buffer = make_shared<Buffer>(*device, cms_data_size);
            op.net_buffer->resize(cms_data_size);
            memcpy(op.net_buffer->data(), cms_tensor->data.raw, cms_data_size);
            op.network = make_shared<Network>(*device, op.net_buffer);
        
            // Get flash tensor data size
            auto flash_idx = op.inputs[FLASH_TENSOR_INDEX];
            auto flash_tensor = &context->tensors[flash_idx];
            size_t flash_data_size = flash_tensor->bytes;
            base_addr_size[0] = static_cast<uint32_t>(flash_data_size);//flash size at first
            if (flash_data_size != 0 && flash_buffer == nullptr) {
                flash_buffer = make_shared<Buffer>(*device, flash_data_size);
                memcpy(flash_buffer->data(), flash_tensor->data.raw, flash_data_size);
            }

            // Get the arena data size
            size_t tmp_arena_size = 0;
            // Get addresses of outputs data
            for (int i = 0; i < op.outputs.size(); ++i) {
                auto tensor = &context->tensors[op.outputs[i]];
                tmp_arena_size += ALIGN_SIZE(tensor->bytes);
                base_addr_size[i + op.outputs.size() - 1] = tensor->bytes;//output size at last
            }
            // Get addresses to inputs data
            for (int i = INPUT_TENSOR_INDEX; i < op.inputs.size(); ++i) {
                auto tensor = &context->tensors[op.inputs[i]];
                tmp_arena_size += ALIGN_SIZE(tensor->bytes);
                base_addr_size[i - 1] = tensor->bytes; //inputs tensor
            }
            // Get addresses to scratch data
            for (int i = SCRATCH_TENSOR_INDEX; i < INPUT_TENSOR_INDEX; ++i) {
                auto tensor = &context->tensors[op.inputs[i]];
                tmp_arena_size += ALIGN_SIZE(tensor->bytes);
                base_addr_size[i - 1] = tensor->bytes; //scratch tensor
                tensor->data.raw = (char*)layout_data; //Avoid no data ptr error in tflite
            }

            if (arena_data_size < tmp_arena_size)
                arena_data_size = tmp_arena_size;
        }

        arena_buffer = make_shared<Buffer>(*device, arena_data_size);
    } catch (exception &e) {
        TF_LITE_KERNEL_LOG(context, "Failed to alloc ethos_u buffer.\n");
        return kTfLiteDelegateError;
    }

    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    try {
        char* arena_data = arena_buffer->data();

        for (auto& op : operations) {
            // Get addresses to input data, copy input data
            for (int i = INPUT_TENSOR_INDEX; i < op.inputs.size(); ++i) {
                auto tensor = &context->tensors[op.inputs[i]];
                int32_t data_offset = address_offsets[op.inputs[i]];
                memcpy(arena_data + data_offset, tensor->data.raw, tensor->bytes);
            }

            vector<shared_ptr<Buffer>> ifm {arena_buffer, op.tensor_layout_buffer};
            vector<shared_ptr<Buffer>> ofm {};
            if (flash_buffer != nullptr)
                ifm.push_back(flash_buffer);

            Inference inference(op.network, ifm.begin(), ifm.end(), ofm.begin(),
                    ofm.end(), pmu_counter_config, options.enable_cycle_counter);
            /* make sure the wait completes ok */
            if (inference.wait(options.timeout) <= 0) {
                TF_LITE_KERNEL_LOG(context, "Ethos_u inference failed\n");
                return kTfLiteDelegateError;
            }

            /* Read out PMU counters if configured */
            if (pmu_counter_config.size() > 0) {
                const vector<uint32_t> pmus = inference.getPmuCounters();
                cout << "Ethos_u PMUs : [";
                for (auto p : pmus) {
                    cout << " " << p;
                }
                cout << " ]" << endl;
            }
            if (options.enable_cycle_counter) {
                cout << "Ethos-u cycle counter: " << inference.getCycleCounter() << endl;
            }

            // Get addresses to output data, copy output data
            for (int i = 0; i < op.outputs.size(); ++i) {
                auto tensor = &context->tensors[op.outputs[i]];
                int32_t data_offset = address_offsets[op.outputs[i]];
                memcpy(tensor->data.raw, arena_data + data_offset, tensor->bytes);
            }
        }
    } catch (exception &e) {
        TF_LITE_KERNEL_LOG(context, "Failed to invoke ethos_u op.\n");
        return kTfLiteDelegateError;
    }

    return kTfLiteOk;
  }

 private:
  struct OperationDataType {
    vector<int> inputs;
    vector<int> outputs;
    shared_ptr<Buffer> net_buffer;  //Buffer for cms tensor
    shared_ptr<Buffer> tensor_layout_buffer;   //Buffer for layout of in/out/scratch
    shared_ptr<Network> network;
  };

  const EthosuDelegateOptions options;

  Device* device;
  shared_ptr<Buffer> arena_buffer;  //Output buffer for input/ouput/scratch tensor
  shared_ptr<Buffer> flash_buffer;  //Input buffer for weight tensor
  vector<OperationDataType> operations;

  const int32_t* address_offsets;
  vector<uint32_t> pmu_counter_config;
  
};

// EthosuDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class EthosuDelegate : public SimpleDelegateInterface {
 public:
  explicit EthosuDelegate(const EthosuDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    return registration->builtin_code == kTfLiteBuiltinCustom
           && !strcmp(registration->custom_name, ETHOSU_CUSTOM_NAME);
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "EthosuDelegate";
    return kName;
  }

  unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return make_unique<EthosuDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const EthosuDelegateOptions options_;
};

}  // namespace ethosu
}  // namespace tflite

EthosuDelegateOptions EthosuDelegateOptionsDefault() {
  EthosuDelegateOptions options;

  options.device_name = ETHOSU_DEFAULT_DEVICE_NAME;
  options.timeout = ETHOSU_DEFAULT_TIMEOUT;
  options.enable_cycle_counter = false;
  memset(options.pmu_counter_config, 0, sizeof(options.pmu_counter_config));

  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteEthosuDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* EthosuDelegateCreate(const EthosuDelegateOptions* options) {
  auto delegate = make_unique<tflite::ethosu::EthosuDelegate>(
          options ? *options : EthosuDelegateOptionsDefault());
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(move(delegate), 
             kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `EthosuDelegateCreate` call.
void EthosuDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
