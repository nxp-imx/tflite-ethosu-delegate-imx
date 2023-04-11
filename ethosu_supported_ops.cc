/*
 * Copyright 2022-2023 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

#include <set>
#include <map>
#include <cmath>
#include <algorithm>

namespace tflite {
namespace ethosu {

#define DEFINEGETPARAM(ParName, ParType) \
template <typename T, typename U = int> \
struct Has_##ParName{ \
  static ParType get(T* ptr, ParType value) { return value;} \
}; \
template <typename T> \
struct Has_##ParName<T, decltype((void)T::ParName, 0)>{  \
  static ParType get(T* ptr, ParType value) {return ptr->ParName;} \
};

DEFINEGETPARAM(dilation_width_factor, int);
DEFINEGETPARAM(dilation_height_factor, int);

#define GETVALUE(structName, ptr, ParName, value) \
        Has_##ParName<structName>::get(ptr, value)


typedef int EthosuValueRange[2];

EthosuValueRange tens_dim_range{1, 65535};
EthosuValueRange stride_range{1, 3};
EthosuValueRange dilated_height_range{1, 64};
EthosuValueRange dilated_product_range{1, 64 * 64};
EthosuValueRange filter_range{1, 8};
EthosuValueRange filter_height_range{1, 256};
EthosuValueRange filter_product_range{1, 256 * 256};
const int weights_limit = 127 * 65536;
const int mean_kernel_product = 64 * 64;
const int mean_kernel_product_avgpool = 256 * 256;

typedef std::set<TfLiteType> TypeList;
const TypeList supported_dtypes{kTfLiteUInt8, kTfLiteInt8, kTfLiteInt16, kTfLiteInt32};
const TypeList supported_faf_dtypes{kTfLiteUInt8, kTfLiteInt8, kTfLiteInt16};
const TypeList supported_weights_dtypes{kTfLiteUInt8, kTfLiteInt8};
const TypeList supported_bias_dtypes{kTfLiteInt32, kTfLiteInt64};
const TypeList supported_pad_dtypes{kTfLiteInt32, kTfLiteInt64};

typedef std::set<TfLiteFusedActivation> ActivationList;
const ActivationList supported_activation{kTfLiteActNone, kTfLiteActRelu, kTfLiteActReluN1To1, kTfLiteActRelu6, kTfLiteActTanh, kTfLiteActSigmoid};

typedef std::vector<std::vector<int>> TensorIndices;
const int TensorIndices_IFMS = 0;
const int TensorIndices_WEIGHTS = 1;
const int TensorIndices_BIASES = 2;
const TensorIndices NO_INDICES{{}, {}, {}};
const TensorIndices IFM_INDICES{{0}, {}, {}};
const TensorIndices IFM_WEIGHTS_INDICES{{0}, {1}, {}};
const TensorIndices IFM_WEIGHTS_BIAS_INDICES{{0}, {1}, {2}};
const TensorIndices IFM_IFM2_INDICES{{0, 1}, {}, {}};
const TensorIndices CONV2D_BACKPROP_INDICES{{2}, {1}, {3}};
const TensorIndices TRANSPOSE_CONV_INDICES{{0}, {1}, {3}};
const TensorIndices SPLIT_IFM_INDICES{{1}, {}, {}};
const TensorIndices BLOCK_LSTM_INDICES{{3}, {4}, {}};


typedef bool (*ConstraintFunc)(TfLiteContext*, const TfLiteNode*, int32_t);

struct OperatorFeature{
  TensorIndices indices;
  std::vector<ConstraintFunc> constraints;
};
extern const std::map<int, OperatorFeature> OPERATOR_MAP;

typedef std::set<int32_t> BuiltinOperatorList;
const BuiltinOperatorList shapeless_output_ops{kTfLiteBuiltinQuantize};
const BuiltinOperatorList shapeless_input_ops{kTfLiteBuiltinQuantize, kTfLiteBuiltinSplit, kTfLiteBuiltinSplitV, kTfLiteBuiltinMean,
                                              kTfLiteBuiltinExpandDims, kTfLiteBuiltinMaximum, kTfLiteBuiltinMinimum, kTfLiteBuiltinAdd,
                                              kTfLiteBuiltinMul, kTfLiteBuiltinSub};
const BuiltinOperatorList convolution_like_ops{kTfLiteBuiltinConv2d, kTfLiteBuiltinDepthwiseConv2d, kTfLiteBuiltinTransposeConv};
const BuiltinOperatorList supported_int32_tensor_ops{kTfLiteBuiltinAdd, kTfLiteBuiltinMul, kTfLiteBuiltinSub, kTfLiteBuiltinShape};
const BuiltinOperatorList multiple_batch_ops{kTfLiteBuiltinSplitV, kTfLiteBuiltinShape, kTfLiteBuiltinSqueeze, kTfLiteBuiltinSlice,
                                             kTfLiteBuiltinSoftmax, kTfLiteBuiltinUnpack, kTfLiteBuiltinSplit, kTfLiteBuiltinReshape,
                                             kTfLiteBuiltinStridedSlice, kTfLiteBuiltinFullyConnected};

inline bool ValueInRange(int value, const EthosuValueRange range){
    return range[0] <= value && value <= range[1];
}

bool ConstraintTensorPre(TfLiteContext* context,
                         const TfLiteNode* node,
                         int32_t builtin_code){
  for (int i = 0; i < node->inputs->size; i ++) {
    if (node->inputs->data[i] < 0)
       continue;
    auto &tensor = context->tensors[node->inputs->data[i]];
    //Input(s) and Output tensors must not be dynamic
    if (tensor.allocation_type == kTfLiteDynamic && builtin_code != kTfLiteBuiltinQuantize)
      return false;
    //Input(s) and Output tensors must have a defined shape
    if (tensor.dims == nullptr)
      return false;
    //Scalar Input tensors are only valid for op type: {}
    if (tensor.dims->size == 0 && shapeless_input_ops.count(builtin_code) == 0)
      return false;
    //Input(s) and Output tensors must not be greater than 4D
    if (tensor.dims->size > 4)
      return false;
    //Constant tensors should have values
    if (tensor.allocation_type == kTfLiteMmapRo && tensor.data.data == nullptr)
      return false;
  }
  for (int i = 0; i < node->outputs->size; i ++) {
    auto &tensor = context->tensors[node->outputs->data[i]];
    //Input(s) and Output tensors must not be dynamic
    if (tensor.allocation_type == kTfLiteDynamic && builtin_code != kTfLiteBuiltinQuantize)
      return false;
    //Input(s) and Output tensors must have a defined shape
    if (tensor.dims == nullptr)
      return false;
    //Scalar Output tensors are only valid for op type: {}
    if (tensor.dims->size == 0 && shapeless_output_ops.count(builtin_code) == 0)
      return false;
    //Input(s) and Output tensors must not be greater than 4D
    if (tensor.dims->size > 4)
      return false;
  }
  return true;
}

bool IsValidQuant(TfLiteQuantization &quant, int32_t builtin_code){
  auto params = reinterpret_cast<TfLiteAffineQuantization*>(quant.params);
  //Input(s), Output and Weight tensors must have quantization parameters
  if(quant.type == kTfLiteNoQuantization || quant.params == nullptr)
    return false;
  //Per-axis quantization is only supported for the following op types: {}
  if (convolution_like_ops.count(builtin_code) == 0 &&
      (params->scale->size > 1 || params->zero_point->size > 1))
    return false;
  //Input(s), Output and Weight tensors with quantization scales must be finite
  for (int i = 0; i < params->scale->size; i ++) {
    if (std::isinf(params->scale->data[i]))
      return false;
  }
  return true;
}

bool ConstraintTensorPost(TfLiteContext* context,
                          const TfLiteNode* node,
                          int32_t builtin_code){
  auto& op_feature = OPERATOR_MAP.at(builtin_code);

  std::vector<TfLiteTensor*> tensors;
  for (auto i : op_feature.indices[TensorIndices_IFMS]){
    if (i >= node->inputs->size)
      return false;
    tensors.push_back(&context->tensors[node->inputs->data[i]]);
  }
  for (auto i : op_feature.indices[TensorIndices_WEIGHTS]){
    if (i >= node->inputs->size)
      return false;
    tensors.push_back(&context->tensors[node->inputs->data[i]]);
  }
  for (int i = 0; i < node->outputs->size; i ++) {
    tensors.push_back(&context->tensors[node->outputs->data[i]]);
  }

  for (auto tensor : tensors){
    //Tensors must be of type: {}
    if (supported_dtypes.count(tensor->type) == 0)
      return false;
    //Tensors which are int32 are only valid when op type is: {}
    if (tensor->type == kTfLiteInt32 && supported_int32_tensor_ops.count(builtin_code) == 0)
      return false;
    //Tensor dimensions must be in the range [{}, {}]
    for (int i = 0; i < tensor->dims->size; i ++) {
        if (!(tens_dim_range[0] < tensor->dims->data[i] < tens_dim_range[1]))
          return false;
    }
    if (builtin_code != kTfLiteBuiltinShape && !IsValidQuant(tensor->quantization, builtin_code))
      return false;
    //TODO
    //Input and Output tensors must have quantization scales that fit within float32 precision
  }
  return true;
}

bool ConstraintMatchingInOutTypes(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  int32_t builtin_code) {
  //IFM and OFM data types must match
  auto& ifm = context->tensors[node->inputs->data[0]];
  auto& ofm = context->tensors[node->outputs->data[0]];
  return (ifm.type == ofm.type);
}

bool ConstraintBatchSize(TfLiteContext* context,
                         const TfLiteNode* node,
                         int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);

  if (multiple_batch_ops.count(builtin_code) != 0)
    return true;

  for (auto i : op_feature.indices[TensorIndices_IFMS]){
    if (i >= node->inputs->size)
      return false;
    auto& tensor = context->tensors[node->inputs->data[i]];
    if (tensor.dims->size == 4 && tensor.dims->data[0] != 1)
      return false;
  }
  return true;
}

template <typename T>
bool ConstraintFaf(TfLiteContext* context,
                   const TfLiteNode* node,
                   int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //The fused activation function (if present) must be one of type: {}
  if (supported_activation.count(param->activation) == 0)
    return false;

  if (param->activation != kTfLiteActNone) {
    for (int i = 0; i < node->outputs->size; i ++) {
      auto& tensor = context->tensors[node->outputs->data[i]];
      if (supported_faf_dtypes.count(tensor.type) == 0)
        return false;
    }
  }
  return true;
}

template <typename T>
bool ConstraintStrideRange(TfLiteContext* context,
                           const TfLiteNode* node,
                           int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Stride values for both width and height must be in the range {}
  return ValueInRange(param->stride_width, stride_range) &&
         ValueInRange(param->stride_height, stride_range);
}

template <typename T>
bool ConstraintDilatedRange(TfLiteContext* context,
                           const TfLiteNode* node,
                           int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Dilated kernel height must be in the range {}
  //Product of dilated kernel width and height must be in the range {}
  auto index = op_feature.indices[TensorIndices_WEIGHTS][0];
  auto& weight = context->tensors[node->inputs->data[index]];
  auto kernel_w = weight.dims->data[2];
  auto kernel_h = weight.dims->data[1];

  auto dilation_w_factor = GETVALUE(T, param, dilation_width_factor, 1);
  auto dilation_h_factor = GETVALUE(T, param, dilation_height_factor, 1);
  auto area_w = (kernel_w - 1) * dilation_w_factor + 1;
  auto area_h = (kernel_h - 1) * dilation_h_factor + 1;
  return ValueInRange(area_h, dilated_height_range) &&
         ValueInRange(area_w * area_h, dilated_product_range);
}

bool ConstraintWeights(TfLiteContext* context,
                       const TfLiteNode* node,
                       int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);

  //Weight tensor must be 8-bit
  //Weight tensor must be constant
  auto index = op_feature.indices[TensorIndices_WEIGHTS][0];
  auto& weight = context->tensors[node->inputs->data[index]];

  if (supported_weights_dtypes.count(weight.type) == 0)
    return false;
  if (weight.allocation_type != kTfLiteMmapRo || weight.data.data == nullptr)
    return false;
  return true;
}

bool ConstraintWeightsLimit(TfLiteContext* context,
                            const TfLiteNode* node,
                            int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);

  //The sum of the weights cannot exceed {}
  auto index = op_feature.indices[TensorIndices_WEIGHTS][0];
  auto& weight = context->tensors[node->inputs->data[index]];
  return true;
}

bool ConstraintBias(TfLiteContext* context,
                    const TfLiteNode* node,
                    int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);

  auto index = op_feature.indices[TensorIndices_BIASES][0];
  if (index >= node->inputs->size)
    return true;

  if (node->inputs->data[index] < 0)
    return true;
  //Optional Bias tensor must be of type: {}
  auto& biases = context->tensors[node->inputs->data[index]];
  if (supported_bias_dtypes.count(biases.type) == 0)
    return false;

  //Optional Bias tensor values must fit within 40-bits
  if (biases.type == kTfLiteInt64) {
    auto data = GetTensorData<int64_t>(&biases);
    for (int i = 0; i < NumElements(&biases); i ++){
      if (data[i] >= 0 && data[i] > (int64_t)0xffffffffff)
        return false;
      if (data[i] < 0 && -data[i] > (int64_t)0x7fffffffff)
        return false;
    }
  }

  return true;
}

template <typename T>
bool ConstraintTconv(TfLiteContext* context,
                     const TfLiteNode* node,
                     int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Stride values for both width and height must be 2
  if (param->stride_width != 2 || param->stride_height != 2)
    return false;

  auto idx0 = op_feature.indices[TensorIndices_IFMS][0];
  auto& input = context->tensors[node->inputs->data[idx0]];
  auto& output = context->tensors[node->outputs->data[0]];
  auto& in_shape = input.dims->data;
  auto& out_shape = output.dims->data;

  //SAME padding: OFM dimensions must equal IFM dimensions multiplied by stride
  if (param->padding == kTfLitePaddingSame) {
    if (out_shape[1] != in_shape[1] * 2 || out_shape[2] != in_shape[2] * 2)
      return false;
  }

  //VALID padding: OFM dimensions must equal IFM dimensions multiplied by stride,
  //      minus difference between kernel size and stride
  auto idx1 = op_feature.indices[TensorIndices_WEIGHTS][0];
  auto& weight = context->tensors[node->inputs->data[idx1]];
  auto kernel_h = weight.dims->data[1];
  auto kernel_w = weight.dims->data[2];
  if (param->padding == kTfLitePaddingValid) {
    if (out_shape[1] != (in_shape[1] * 2 + std::max(kernel_h - 2, 0)) ||
        out_shape[2] != (in_shape[2] * 2 + std::max(kernel_w - 2, 0)))
      return false;
  }

  return true;
}

template <typename T>
bool ConstraintDepthMultiplier(TfLiteContext* context,
                               const TfLiteNode* node,
                               int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);
  T* param = reinterpret_cast<T*>(node->builtin_data);

  auto index = op_feature.indices[TensorIndices_IFMS][0];
  auto& input = context->tensors[node->inputs->data[index]];
  auto& output = context->tensors[node->outputs->data[0]];
  auto& in_shape = input.dims->data;
  auto& out_shape = output.dims->data;

  //For depth multipliers > 1, IFM channels must be 1
  // and OFM channels must be equal to the depth multiplier
  if (param->depth_multiplier > 1) {
    if (in_shape[3] != 1 || out_shape[3] != param->depth_multiplier)
      return false;
  }
  return true;
}

template <typename T>
bool ConstraintFilterRange(TfLiteContext* context,
                           const TfLiteNode* node,
                           int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Kernel filter values for both width and height must be in the range {}
  if (param->padding == kTfLitePaddingSame) {
    return ValueInRange(param->filter_width, filter_range) &&
           ValueInRange(param->filter_height, filter_range);
  }
  return true;
}

template <typename T>
bool ConstraintFilterHeightRange(TfLiteContext* context,
                                 const TfLiteNode* node,
                                 int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Kernel filter height must be in the range {}
  return ValueInRange(param->filter_height, filter_height_range);
}

template <typename T>
bool ConstraintFilterProductRange(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Product of kernel filter width and height must be in the range{}
  auto product = param->filter_height * param->filter_width;
  return ValueInRange(product, filter_product_range);
}

template <typename T>
bool ConstraintFilterHeightRangeValidPad(TfLiteContext* context,
                                         const TfLiteNode* node,
                                         int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Kernel filter height must be in the range {}
  if (param->padding == kTfLitePaddingValid) {
    return ValueInRange(param->filter_height, filter_height_range);
  }
  return true;
}

template <typename T>
bool ConstraintFilterProductRangeValidPad(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Product of kernel filter width and height must be in the range{}
  auto product = param->filter_height * param->filter_width;
  if (param->padding == kTfLitePaddingValid) {
    return ValueInRange(product, filter_product_range);
  }
  return true;
}

template <typename T>
bool ConstraintResize(TfLiteContext* context,
                      const TfLiteNode* node,
                      int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  /*The width and height of the IFM and OFM must match one of the following criteria:
    IFM W and H must both be 1
    IFM must match OFM
    W and H scaling must be equal and OFM W-1 and H-1 must be 2x/4x/8x IFM W-1 and H-1, if align_corners is True
    W and H scaling must be equal and OFM W and H must be 2x/4x/8x IFM W and H, if align_corners is False */
  auto& input = context->tensors[node->inputs->data[0]];
  auto& output = context->tensors[node->outputs->data[0]];
  auto& in_shape = input.dims->data;
  auto& out_shape = output.dims->data;

  if (input.dims->size != 4 || output.dims->size != 4)
    return false;

  //Valid if IFM W and H are both 1, or IFM and OFM shape are the same
  if (in_shape[1] == 1 && in_shape[2] == 1)
    return true;
  if (in_shape[0] == out_shape[0] && in_shape[1] == out_shape[1]
      && in_shape[2] == out_shape[2] && in_shape[3] == out_shape[3])
    return true;

  // Valid if OFM is 2/4/8x IFM (-1 for align corners)
  int h_factor, w_factor;
  if (param->align_corners) {
    h_factor = (out_shape[1] - 1) / (in_shape[1] - 1);
    w_factor = (out_shape[2] - 1) / (in_shape[2] - 1);
  } else {
    h_factor = out_shape[1] / in_shape[1];
    w_factor = out_shape[2] / in_shape[2];
  }
  return ((h_factor == w_factor) && 
          (h_factor == 2 || h_factor == 4 || h_factor == 8));
}

bool ConstraintResizeSize(TfLiteContext* context,
                          const TfLiteNode* node,
                          int32_t builtin_code) {
  //The size tensor (the second input) exists
  if (node->inputs->size != 2)
    return false;
  auto& size = context->tensors[node->inputs->data[1]];
  auto& output = context->tensors[node->outputs->data[0]];
  auto& size_shape = size.dims->data;
  auto& out_shape = output.dims->data;

  //The size tensor must match the output tensor shape
  if (size.dims->size != 1 || size_shape[0] != 2)
    return false;
  auto size_data = GetTensorData<int32_t>(&size);
  return (size_data[0] == out_shape[1] && size_data[1] == out_shape[2]);
}

template <typename T>
bool ConstraintResizeAttrs(TfLiteContext* context,
                           const TfLiteNode* node,
                           int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Both align_corners and half_pixel_centers can't be True
  return !(param->align_corners && param->half_pixel_centers);
}

template <typename T>
bool ConstraintResizebiHalfPixelCentersDims(TfLiteContext* context,
                                            const TfLiteNode* node,
                                            int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Half_pixel_centers for resize bilinear requires that OFM W and H is 2x IFM W and H
  auto& input = context->tensors[node->inputs->data[0]];
  auto& output = context->tensors[node->outputs->data[0]];
  auto& in_shape = input.dims->data;
  auto& out_shape = output.dims->data;
  if (!param->half_pixel_centers)
    return true;
  return (out_shape[1] / in_shape[1] == 2 && out_shape[2] / in_shape[2] == 2);
}

bool ConstraintMatchingQuantizationParams(TfLiteContext* context,
                                          const TfLiteNode* node,
                                          int32_t builtin_code) {
  auto& input = context->tensors[node->inputs->data[0]];
  auto& output = context->tensors[node->outputs->data[0]];
  auto in_quant = reinterpret_cast<TfLiteAffineQuantization*>(input.quantization.params);
  auto out_quant = reinterpret_cast<TfLiteAffineQuantization*>(output.quantization.params);

  //Both Input quantization parameters must match OFM quantization parameters
  if (input.quantization.type == kTfLiteNoQuantization ||
      output.quantization.type == kTfLiteNoQuantization)
    return false;
  if (in_quant->scale->data[0] != out_quant->scale->data[0] ||
      in_quant->zero_point->data[0] != out_quant->zero_point->data[0])
    return false;
  if (node->inputs->size > 1) {
    auto &input2 =  context->tensors[node->inputs->data[1]];
    auto in2_quant = reinterpret_cast<TfLiteAffineQuantization*>(input2.quantization.params);
    if (input2.quantization.type == kTfLiteNoQuantization ||
        in2_quant->scale->data[0] != out_quant->scale->data[0] ||
        in2_quant->zero_point->data[0] != out_quant->zero_point->data[0])
      return false;
  }
  return true;
}

bool ConstraintBroadcastShapes(TfLiteContext* context,
                               const TfLiteNode* node,
                               int32_t builtin_code) {
  //Broadcasting is only allowed for rank indices with dimension 1, from either IFM1 or IFM2
  if (node->inputs->size == 1)
    return true;

  auto& in1 = context->tensors[node->inputs->data[0]];
  auto& in2 = context->tensors[node->inputs->data[1]];
  auto& out = context->tensors[node->outputs->data[0]];
  auto& in1_shape = in1.dims->data;
  auto& in2_shape = in2.dims->data;
  auto& out_shape = out.dims->data;
  auto size = std::min(in1.dims->size, in2.dims->size);
  for (int idx = 0; idx < size; idx ++){
    auto i1 = in1_shape[in1.dims->size - 1 - idx];
    auto i2 = in2_shape[in2.dims->size - 1 - idx];
    auto o = out_shape[out.dims->size - 1 - idx];
    auto mi = std::max(i1, i2);
    //Input dimensions should match or one should be of dimension 1
    //Output dimension should match the largest input dimension, together
    //with constraint_match_either_shapes ensures broadcast from only one input
    if (!(i1 == i2 || i1 == 1 || i2 == 1) || o != mi)
      return false;
  }
  return true;
}

template <typename T>
bool ConstraintStridedSliceValues(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Exactly 4 Input tensors are required
  if (node->inputs->size != 4)
    return false;
  auto& ifm = context->tensors[node->inputs->data[0]];
  auto& begin = context->tensors[node->inputs->data[1]];
  auto& end = context->tensors[node->inputs->data[2]];
  auto& strides = context->tensors[node->inputs->data[3]];
  auto begin_data = GetTensorData<int32_t>(&begin);
  auto end_data = GetTensorData<int32_t>(&end);
  auto strides_data = GetTensorData<int32_t>(&strides);

  //Begin, End and Stride Input tensors must be constant
  if (begin.allocation_type != kTfLiteMmapRo || begin.data.data == nullptr)
    return false;
  if (end.allocation_type != kTfLiteMmapRo || end.data.data == nullptr)
    return false;
  if (strides.allocation_type != kTfLiteMmapRo || strides.data.data == nullptr)
    return false;

  //ellipsis_mask must be 0
  if (param->ellipsis_mask != 0)
    return false;

  //new_axis_mask and shrink_axis_mask cannot both be set
  if (param->new_axis_mask != 0 && param->shrink_axis_mask != 0)
    return false;

  //Slice 'end' values must be greater than 'begin' values"
  for (int i = 0; i < ifm.dims->size; i ++){
    int32_t end, begin;
    //If the i:th bit in the mask is set then the value on offset_tens[i] should be ignored
    if (param->begin_mask & (1 << i) == 0) {
      begin = begin_data[i];
      begin += begin < 0 ? ifm.dims->data[i] : 0;
    } else {
      begin = 0;
    }
    if (param->end_mask & (1 << i) == 0) {
      end = end_data[i];
      end += end < 0 ? ifm.dims->data[i] : 0;
    } else {
      end = ifm.dims->data[i];
    }
    if (end <= begin)
      return false;
  }


  //All Strides values must be 1
  for (int i = 0; i < NumElements(&strides); i ++){
    if (strides_data[i] != 1)
      return false;
  }
  return true;
}

bool ConstraintPad(TfLiteContext* context,
                   const TfLiteNode* node,
                   int32_t builtin_code) {
  //Number of input tensors must be exactly 2
  if (node->inputs->size != 2)
    return false;
  auto& pad = context->tensors[node->inputs->data[1]];
  //The padding tensor must be constant
  if (pad.allocation_type != kTfLiteMmapRo || pad.data.data == nullptr)
    return false;
  //Pad tensor must be of type: {}
  //The padding tensor must have the shape [3,2] or [4,2]
  if (supported_pad_dtypes.count(pad.type) == 0)
    return false;
  if (pad.dims->size != 2)
    return false;
  return ((pad.dims->data[0] == 3 && pad.dims->data[1] == 2) ||
          (pad.dims->data[0] == 4 && pad.dims->data[1] == 2));
}

bool
ConstraintMeanAxisValue(TfLiteContext* context,
                        const TfLiteNode* node,
                        int32_t builtin_code) {
  auto& in = context->tensors[node->inputs->data[0]];
  auto& axis = context->tensors[node->inputs->data[1]];

  //Input tensor must be at least 2D
  if (!(2 <= in.dims->size <= 4))
    return false;

  auto axis_data = GetTensorData<int32_t>(&axis); 
  auto num_axis = NumElements(&axis);
  //Axis indices must correspond to height and width axes
  if (in.dims->size == 2 || in.dims->size == 3) {
    if (num_axis == 1) {
      if (axis_data[0] == 0 || axis_data[0] == 1)
        return true;
    } else if (num_axis == 2) {
      if ((axis_data[0] == 0 && axis_data[1] == 1) || (axis_data[0] == 1 && axis_data[1] == 0))
        return true;
    }
  } else if (in.dims->size == 4){
    if (num_axis == 1) {
      if (axis_data[0] == 1 || axis_data[0] == 2)
        return true;
    } else if (num_axis == 2) {
      if ((axis_data[0] == 1 && axis_data[1] == 2) || (axis_data[0] == 2 && axis_data[1] == 1))
        return true;
    }
  }
  return false;
}

template <typename T>
bool ConstraintMeanProduct(TfLiteContext* context,
                           const TfLiteNode* node,
                           int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  //Product of height and width must be no greater than {} when:
  //IFM and OFM have different scale or zero point; or
  //'keep_dims' is True
  auto& in = context->tensors[node->inputs->data[0]];
  auto& out = context->tensors[node->outputs->data[0]];
  auto in_quant = reinterpret_cast<TfLiteAffineQuantization*>(in.quantization.params);
  auto out_quant = reinterpret_cast<TfLiteAffineQuantization*>(out.quantization.params);

  int max_prod = 0;
  if (!param->keep_dims && in_quant->scale->data[0] == out_quant->scale->data[0] &&
      in_quant->zero_point->data[0] == out_quant->zero_point->data[0])
    max_prod = mean_kernel_product_avgpool;
  else
    max_prod = mean_kernel_product;
  
  auto& in_shape = in.dims->data;
  int w,h;
  if (in.dims->size > 3){
    h = in_shape[1];
    w = in_shape[2];
  } else {
    h = in_shape[0];
    w = in_shape[1];
  }
  return (h * w <= max_prod);
}

bool ConstraintMeanHeightSingleAxis(TfLiteContext* context,
                                    const TfLiteNode* node,
                                    int32_t builtin_code) {
  //For single axis averages across the height dimension:
  //IFM height must be no greater than {} if the IFM and OFM scale and zero point match; otherwise
  //IFM height must be no greater than {} if the IFM and OFM scale or zero point do not match
  auto& in = context->tensors[node->inputs->data[0]];
  auto& axis = context->tensors[node->inputs->data[1]];
  auto& out = context->tensors[node->outputs->data[0]];
  auto& in_shape = in.dims->data;
  auto& axis_shape = axis.dims->data;
  auto in_quant = reinterpret_cast<TfLiteAffineQuantization*>(in.quantization.params);
  auto out_quant = reinterpret_cast<TfLiteAffineQuantization*>(out.quantization.params);

  int32_t axis_value;
  if (NumElements(&axis) == 1)  // single axis
    axis_value = GetTensorData<int32_t>(&axis)[0];
  else
    return true;

  //No height dimension present in IFM
  if (in.dims->size < 3)
    return true;

  //Not averaging across the height dimension
  if (axis_value != in.dims->size - 3)
    return true;

  int max_height = 0;
  if (in_quant->scale->data[0] == out_quant->scale->data[0] &&
      in_quant->zero_point->data[0] == out_quant->zero_point->data[0]) {
    max_height = filter_height_range[1];
  } else {
    max_height = dilated_height_range[1];
  }
  return (in_shape[axis_value] <= max_height);
}

bool ConstraintReshapeShapeConstant(TfLiteContext* context,
                                    const TfLiteNode* node,
                                    int32_t builtin_code) {
  //Shape must be constant
  auto& shape = context->tensors[node->inputs->data[1]];
  return (shape.allocation_type == kTfLiteMmapRo && shape.data.data != nullptr);
}

template <typename T>
bool ConstraintConcatAxis(TfLiteContext* context,
                          const TfLiteNode* node,
                          int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  auto& output = context->tensors[node->outputs->data[0]];
  int out_dims = output.dims->size;
  int axis = param->axis > 0 ? param->axis : param->axis + out_dims;
  //Axis attribute must be in the range [0, <out_dims>)
  if (!(0 <= axis < out_dims))
    return false;
  int sum_ifm_axis = 0;
  for (int i = 0; i < node->inputs->size; i ++){
    auto& input = context->tensors[node->inputs->data[i]];
    //All Input dimensionalities must match OFM dimensionality
    if (input.dims->size != out_dims)
      return false;
    for (int dim = 0; dim < out_dims; dim ++){
      //All Input dimensions must match OFM dimension in all axes except <axis>
      if (dim != axis && output.dims->data[dim] != input.dims->data[dim])
        return false;
    }
    sum_ifm_axis += input.dims->data[axis];
  }
  //The size of the OFM axis must match the sum of all IFM axis defined by the axis attribute
  return sum_ifm_axis == output.dims->data[axis];
}

template <typename T>
bool ConstraintFullyConnected(TfLiteContext* context,
                              const TfLiteNode* node,
                              int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  auto& in = context->tensors[node->inputs->data[0]];
  auto& weight = context->tensors[node->inputs->data[1]];
  auto& out = context->tensors[node->outputs->data[0]];

  //The output tensor(s) must have 2D shape
  if (in.dims->size == 1)
    return false;
  if (NumElements(&in) % weight.dims->data[1] != 0)
    return false;
  //The IFM and OFM must have the same number of dimensions if keep_num_dims is set to true
  if (param->keep_num_dims && in.dims->size != out.dims->size)
    return false;

  return true;
}

template <typename T>
bool ConstraintBetaRange(TfLiteContext* context,
                         const TfLiteNode* node,
                         int32_t builtin_code) {
  T* param = reinterpret_cast<T*>(node->builtin_data);

  return param->beta >= 0;
}

bool ConstraintMatchingEitherShapes(TfLiteContext* context,
                                    const TfLiteNode* node,
                                    int32_t builtin_code) {
  //At least one Input's shape must match the OFM's shape
  auto& in = context->tensors[node->inputs->data[0]];
  auto& out = context->tensors[node->outputs->data[0]];

  if (HaveSameShapes(&in, &out))
    return true;
  if (node->inputs->size == 1)
    return false;
  auto& in2 = context->tensors[node->inputs->data[1]];
  return HaveSameShapes(&in2, &out);
}

bool ConstraintInOutTypesValid(TfLiteContext* context,
                               const TfLiteNode* node,
                               int32_t builtin_code) {
  auto& in = context->tensors[node->inputs->data[0]];
  auto& in2 = context->tensors[node->inputs->data[1]];
  auto& out = context->tensors[node->outputs->data[0]];
  //Both Input data types must match
  if (in.type != in2.type)
    return false;
  if (in.type != kTfLiteUInt8) {
    //For IFM that are signed, OFM must also be signed
    if (out.type == kTfLiteUInt8)
      return false;
  } else {
    //For IFM that are unsigned, OFM must either be the same type or int32
    if (out.type != in.type && out.type != kTfLiteInt32)
      return false;
  }
  return true;
}

bool ConstraintSplitVInferred(TfLiteContext* context,
                              const TfLiteNode* node,
                              int32_t builtin_code) {
  if (node->inputs->size != 3)
    return false;
  //Only one size is allowed to be inferred
  auto& splits = context->tensors[node->inputs->data[1]];
  if (splits.type == kTfLiteInt64)
    return false;
  auto splits_data = GetTensorData<int32_t>(&splits);
  int num_inferred = 0;
  for (int i = 0; i < NumElements(&splits); i ++) {
    if (splits_data[i] < 0)
      num_inferred ++;
  }
  return (num_inferred <= 1);
}

bool ConstraintInput8bit(TfLiteContext* context,
                         const TfLiteNode* node,
                         int32_t builtin_code) {
  auto& op_feature = OPERATOR_MAP.at(builtin_code);
  auto index = op_feature.indices[TensorIndices_IFMS][0];
  auto& in = context->tensors[node->inputs->data[index]];

  return (in.type == kTfLiteInt8 || in.type == kTfLiteUInt8);
}

//TODO
std::vector<ConstraintFunc> common_constraints{ConstraintTensorPre, ConstraintTensorPost, ConstraintBatchSize};

//Add Input and output quantisation must match constraint.
const std::map<int, OperatorFeature> OPERATOR_MAP{
  { kTfLiteBuiltinConcatenation,
     { IFM_IFM2_INDICES,
       {ConstraintFaf<TfLiteConcatenationParams>, ConstraintConcatAxis<TfLiteConcatenationParams>}
     }
  },
  { kTfLiteBuiltinAbs,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinExpandDims,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinExp,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinSplitV,
     { IFM_INDICES,
       {ConstraintSplitVInferred}
     }
  },
  { kTfLiteBuiltinShape,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinSqueeze,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinRelu,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinRelu6,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinPrelu,
     { IFM_IFM2_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinReluN1To1,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinLeakyRelu,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinTanh,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinSlice,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinLogistic,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinHardSwish,
     { IFM_INDICES,
       {ConstraintInput8bit, ConstraintMatchingInOutTypes}
     }
  },
  { kTfLiteBuiltinQuantize,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinPack,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinSoftmax,
     { IFM_INDICES,
       {ConstraintMatchingInOutTypes, ConstraintMatchingEitherShapes, ConstraintBetaRange<TfLiteSoftmaxParams>}
     }
  },
  { kTfLiteBuiltinUnpack,
     { IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinSplit,
     { SPLIT_IFM_INDICES,
       {}
     }
  },
  { kTfLiteBuiltinAdd,
     { IFM_IFM2_INDICES,
       {ConstraintMatchingEitherShapes, ConstraintInOutTypesValid, ConstraintFaf<TfLiteAddParams>, ConstraintBroadcastShapes}
     }
  },
  { kTfLiteBuiltinSub,
     { IFM_IFM2_INDICES,
       {ConstraintMatchingEitherShapes, ConstraintInOutTypesValid, ConstraintFaf<TfLiteSubParams>, ConstraintBroadcastShapes}
     }
  },
  { kTfLiteBuiltinMul,
     { IFM_IFM2_INDICES,
       {ConstraintMatchingEitherShapes, ConstraintInOutTypesValid, ConstraintFaf<TfLiteMulParams>, ConstraintBroadcastShapes}
     }
  },
  { kTfLiteBuiltinReshape,
     { IFM_INDICES,
       {ConstraintReshapeShapeConstant}
     }
  },
  { kTfLiteBuiltinMaximum,
     { IFM_IFM2_INDICES,
       {ConstraintMatchingEitherShapes, ConstraintMatchingInOutTypes, ConstraintMatchingQuantizationParams, ConstraintBroadcastShapes}
     }
  },
  { kTfLiteBuiltinMinimum,
     { IFM_IFM2_INDICES,
       {ConstraintMatchingEitherShapes, ConstraintMatchingInOutTypes, ConstraintMatchingQuantizationParams, ConstraintBroadcastShapes}
     }
  },
  { kTfLiteBuiltinPad,
     { IFM_INDICES,
       {ConstraintPad}
     }
  },
  { kTfLiteBuiltinStridedSlice,
     { IFM_INDICES,
       {ConstraintStridedSliceValues<TfLiteStridedSliceParams>}
     }
  },
  { kTfLiteBuiltinMean,
     { IFM_INDICES,
       {ConstraintInput8bit, ConstraintMeanProduct<TfLiteReducerParams>, ConstraintMeanHeightSingleAxis}
     }
  },
  { kTfLiteBuiltinAveragePool2d,
     { IFM_INDICES,
       {ConstraintMatchingInOutTypes, ConstraintFaf<TfLitePoolParams>, ConstraintStrideRange<TfLitePoolParams>,
        ConstraintFilterRange<TfLitePoolParams>, ConstraintFilterHeightRangeValidPad<TfLitePoolParams>,
        ConstraintFilterProductRangeValidPad<TfLitePoolParams>}
     }
  },
  { kTfLiteBuiltinMaxPool2d,
     { IFM_INDICES,
       {ConstraintMatchingInOutTypes, ConstraintFaf<TfLitePoolParams>, ConstraintStrideRange<TfLitePoolParams>,
        ConstraintFilterHeightRange<TfLitePoolParams>, ConstraintFilterProductRange<TfLitePoolParams>}
     }
  },
  { kTfLiteBuiltinConv2d,
     { IFM_WEIGHTS_BIAS_INDICES,
       {ConstraintFaf<TfLiteConvParams>, ConstraintStrideRange<TfLiteConvParams>,
        ConstraintDilatedRange<TfLiteConvParams>, ConstraintWeights, ConstraintWeightsLimit, ConstraintBias}
     }
  },
  { kTfLiteBuiltinDepthwiseConv2d,
     { IFM_WEIGHTS_BIAS_INDICES,
       {ConstraintFaf<TfLiteDepthwiseConvParams>, ConstraintStrideRange<TfLiteDepthwiseConvParams>,
        ConstraintDilatedRange<TfLiteDepthwiseConvParams>, ConstraintWeights, ConstraintWeightsLimit, ConstraintBias,
        ConstraintDepthMultiplier<TfLiteDepthwiseConvParams>}
     }
  },
  { kTfLiteBuiltinTransposeConv,
     { CONV2D_BACKPROP_INDICES,
       {ConstraintStrideRange<TfLiteTransposeConvParams>,ConstraintDilatedRange<TfLiteTransposeConvParams>, ConstraintWeights,
        ConstraintWeightsLimit, ConstraintBias, ConstraintTconv<TfLiteTransposeConvParams>}
     }
  },
  { kTfLiteBuiltinFullyConnected,
     { IFM_WEIGHTS_BIAS_INDICES,
       {ConstraintFaf<TfLiteFullyConnectedParams>, ConstraintFullyConnected<TfLiteFullyConnectedParams>,
        ConstraintDilatedRange<TfLiteFullyConnectedParams>, ConstraintWeights, ConstraintBias}
     }
  },
  { kTfLiteBuiltinResizeBilinear,
     { IFM_INDICES,
       {ConstraintResize<TfLiteResizeBilinearParams>, ConstraintResizeSize, ConstraintResizeAttrs<TfLiteResizeBilinearParams>}
     }
  },
  { kTfLiteBuiltinResizeNearestNeighbor,
     { IFM_INDICES,
       {ConstraintResize<TfLiteResizeNearestNeighborParams>, ConstraintResizeSize, ConstraintResizeAttrs<TfLiteResizeNearestNeighborParams>,
        ConstraintResizebiHalfPixelCentersDims<TfLiteResizeNearestNeighborParams>}
     }
  },
};

bool IsNodeSupportedByEthosU(TfLiteContext* context,
                             const TfLiteNode* node,
                             int32_t builtin_code) {
  if (OPERATOR_MAP.count(builtin_code) == 0)
    return false;

  for (auto constraint_func : common_constraints) {
    if (constraint_func(context, node, builtin_code) == false)
      return false;
  }

  auto& op_feature = OPERATOR_MAP.at(builtin_code);
  for (auto constraint_func : op_feature.constraints) {
    if (constraint_func(context, node, builtin_code) == false)
      return false;
  }
  return true;
}

}  // namespace ethosu
}  // namespace tflite
