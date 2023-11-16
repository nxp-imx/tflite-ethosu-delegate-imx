/*
 * Copyright 2022-2023 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "tflite_enum_mapping.h"
#include "ethosu_delegate_utils.h"

#include <fstream>
#include <numeric>
#include <dlfcn.h>
#include <exception>

namespace tflite {
namespace ethosu {

std::unique_ptr<tflite::ModelT> readTFLiteModel(const std::string &filename) {
  // Open file.
  std::ifstream modelFile;
  modelFile.open(filename, std::ios::binary);
  if (!modelFile.is_open()) {
    throw "Error opening model file!";
  }

  // Get model size.
  modelFile.seekg(0, std::ios::end);
  std::streamsize modelSize = modelFile.tellg();
  modelFile.seekg(0, std::ios::beg);

  // Read model data.
  std::vector<char> modelData = std::vector<char>(modelSize);
  modelFile.read(modelData.data(), modelSize);
  modelFile.close();

  // Read model object and unpack.
  const tflite::Model *model = tflite::GetModel(modelData.data());
  return std::unique_ptr<tflite::ModelT>(model->UnPack());
}

void writeTFLiteModel(const tflite::ModelT *model, const std::string &filename) {
  // Pack model into buffer.
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<tflite::Model> root = tflite::Model::Pack(fbb, model);
  tflite::FinishModelBuffer(fbb, root);
  uint32_t size = fbb.GetSize();
  uint8_t *buff = fbb.GetBufferPointer();

  // Open file.
  std::ofstream modelFile;
  modelFile.open(filename, std::ios::binary);
  if (!modelFile.is_open()) {
    throw "Error opening model file!";
  }

  // Write model data.
  modelFile.write((const char *)buff, size);
  modelFile.close();
}

inline std::string CharPtrToStr(const char *in) {
  if (in == nullptr) {
    return std::string("");
  } else {
    return std::string(in);
  }
}

template <class A, class B>
std::vector<A> TfLiteArrayToVector(const B* int_array) {
  std::vector<A> values;
  if (!int_array) {
  return values;
  }

  values.resize(int_array->size);
  for (size_t i = 0; i < int_array->size; i++) {
  values[i] = int_array->data[i];
  }

  return values;
}

int64_t GetQuantizedMin(TensorType type) {
  switch (type) {
  case TensorType::TensorType_INT32:
    return std::numeric_limits<int32_t>::min();
  case TensorType::TensorType_UINT8:
    return std::numeric_limits<uint8_t>::min();
  case TensorType::TensorType_INT64:
    return std::numeric_limits<int64_t>::min();
  case TensorType::TensorType_INT16:
    return std::numeric_limits<int16_t>::min();
  case TensorType::TensorType_INT8:
    return std::numeric_limits<int8_t>::min();
  default:
    break;
  }
  return std::numeric_limits<int8_t>::min();
}

int64_t GetQuantizedMax(TensorType type) {
  switch (type) {
  case TensorType::TensorType_INT32:
    return std::numeric_limits<int32_t>::max();
  case TensorType::TensorType_UINT8:
    return std::numeric_limits<uint8_t>::max();
  case TensorType::TensorType_INT64:
    return std::numeric_limits<int64_t>::max();
  case TensorType::TensorType_INT16:
    return std::numeric_limits<int16_t>::max();
  case TensorType::TensorType_INT8:
    return std::numeric_limits<int8_t>::max();
  default:
    break;
  }
  return std::numeric_limits<int8_t>::max();
}

std::vector<float> GetFloatMin(TensorType type,
                               std::vector<float> scale,
                               std::vector<int64_t> offset) {
  assert(scale.size() == offset.size());
  auto size = scale.size();
  std::vector<float> ret;

  ret.reserve(size);
  for (int i = 0; i < size; i ++) {
      ret[i] = scale[i] * static_cast<float>(GetQuantizedMin(type) - offset[i]);
  }
  return ret;
}

std::vector<float> GetFloatMax(TensorType type,
                               std::vector<float> scale,
                               std::vector<int64_t> offset) {
  assert(scale.size() == offset.size());
  auto size = scale.size();
  std::vector<float> ret;

  ret.reserve(size);
  for (int i = 0; i < size; i ++) {
      ret[i] = scale[i] * static_cast<float>(GetQuantizedMax(type) - offset[i]);
  }
  return ret;
}

void SetBuiltinOptions(OperatorT *op, int32_t op_code, void* data){
  switch (op_code) {
    case BuiltinOperator_ADD: {
      auto params = reinterpret_cast<TfLiteAddParams*>(data);
      auto option = AddOptionsT();
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SUB: {
      auto params = reinterpret_cast<TfLiteSubParams*>(data);
      auto option = SubOptionsT();
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.pot_scale_int16 = params->pot_scale_int16;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_CONV_2D: {
      auto params = reinterpret_cast<TfLiteConvParams*>(data);
      auto option = Conv2DOptionsT();
      option.padding = TfLitePaddingToSchemaPadding(params->padding);
      option.stride_w = params->stride_width;
      option.stride_h = params->stride_height;
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.dilation_w_factor = params->dilation_width_factor;
      option.dilation_h_factor = params->dilation_height_factor;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      auto params = reinterpret_cast<TfLiteDepthwiseConvParams*>(data);
      auto option = DepthwiseConv2DOptionsT();
      option.padding = TfLitePaddingToSchemaPadding(params->padding);
      option.stride_w = params->stride_width;
      option.stride_h = params->stride_height;
      option.depth_multiplier = params->depth_multiplier;
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.dilation_w_factor = params->dilation_width_factor;
      option.dilation_h_factor = params->dilation_height_factor;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_TRANSPOSE_CONV: {
      auto params = reinterpret_cast<TfLiteTransposeConvParams*>(data);
      auto option = TransposeConvOptionsT();
      option.padding = TfLitePaddingToSchemaPadding(params->padding);
      option.stride_w = params->stride_width;
      option.stride_h = params->stride_height;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_PACK: {
      auto params = reinterpret_cast<TfLitePackParams*>(data);
      auto option = PackOptionsT();
      option.values_count = params->values_count;
      option.axis = params->axis;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SPLIT_V: {
      auto params = reinterpret_cast<TfLiteSplitVParams*>(data);
      auto option = SplitVOptionsT();
      option.num_splits = params->num_splits;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SHAPE: {
      auto params = reinterpret_cast<TfLiteShapeParams*>(data);
      auto option = ShapeOptionsT();
      option.out_type = TfLiteTypeToSchemaType(params->out_type);
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SQUEEZE: {
      auto params = reinterpret_cast<TfLiteSqueezeParams*>(data);
      auto option = SqueezeOptionsT();
      for (int i = 0; i < params->num_squeeze_dims; i ++) {
          option.squeeze_dims.push_back(params->squeeze_dims[i]);
      }
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_EXPAND_DIMS: {
      auto option = ExpandDimsOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_EXP: {
      auto option = ExpOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_STRIDED_SLICE: {
      auto params = reinterpret_cast<TfLiteStridedSliceParams*>(data);
      auto option = StridedSliceOptionsT();
      option.begin_mask = params->begin_mask;
      option.end_mask = params->end_mask;
      option.ellipsis_mask = params->ellipsis_mask;
      option.new_axis_mask = params->new_axis_mask;
      option.shrink_axis_mask = params->shrink_axis_mask;
      option.offset = params->offset;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_RESIZE_BILINEAR: {
      auto params = reinterpret_cast<TfLiteResizeBilinearParams*>(data);
      auto option = ResizeBilinearOptionsT();
      option.align_corners = params->align_corners;
      option.half_pixel_centers = params->half_pixel_centers;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
      auto params = reinterpret_cast<TfLiteResizeNearestNeighborParams*>(data);
      auto option = ResizeNearestNeighborOptionsT();
      option.align_corners = params->align_corners;
      option.half_pixel_centers = params->half_pixel_centers;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_AVERAGE_POOL_2D: {
      auto params = reinterpret_cast<TfLitePoolParams*>(data);
      auto option = Pool2DOptionsT();
      option.padding = TfLitePaddingToSchemaPadding(params->padding);
      option.stride_w = params->stride_width;
      option.stride_h = params->stride_height;
      option.filter_width = params->filter_width;
      option.filter_height = params->filter_height;
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_FULLY_CONNECTED: {
      auto params = reinterpret_cast<TfLiteFullyConnectedParams*>(data);
      auto option = FullyConnectedOptionsT();
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.weights_format = FullyConnectedOptionsWeightsFormatToSchema(params->weights_format);
      option.keep_num_dims = params->keep_num_dims;
      option.asymmetric_quantize_inputs = params->asymmetric_quantize_inputs;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_PAD: {
      auto option = PadOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_LEAKY_RELU: {
      auto params = reinterpret_cast<TfLiteLeakyReluParams*>(data);
      auto option = LeakyReluOptionsT();
      option.alpha = params->alpha;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SPLIT: {
      auto params = reinterpret_cast<TfLiteSplitParams*>(data);
      auto option = SplitOptionsT();
      option.num_splits = params->num_splits;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SOFTMAX: {
      auto params = reinterpret_cast<TfLiteSoftmaxParams*>(data);
      auto option = SoftmaxOptionsT();
      option.beta = params->beta;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_QUANTIZE: {
      auto option = QuantizeOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_MAXIMUM:
    case BuiltinOperator_MINIMUM: {
      auto option = MaximumMinimumOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_HARD_SWISH: {
      auto option = HardSwishOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_UNPACK: {
      auto params = reinterpret_cast<TfLiteUnpackParams*>(data);
      auto option = UnpackOptionsT();
      option.num = params->num;
      option.axis = params->axis;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_RESHAPE: {
      auto params = reinterpret_cast<TfLiteReshapeParams*>(data);
      auto option = ReshapeOptionsT();
      for (int i = 0; i < params->num_dimensions; i ++) {
          option.new_shape.push_back(params->shape[i]);
      }
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_MEAN: {
      auto params = reinterpret_cast<TfLiteReducerParams*>(data);
      auto option = ReducerOptionsT();
      option.keep_dims = params->keep_dims;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SPACE_TO_BATCH_ND: {
      auto option = SpaceToBatchNDOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_BATCH_TO_SPACE_ND: {
      auto option = BatchToSpaceNDOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_GELU: {
      auto params = reinterpret_cast<TfLiteGeluParams*>(data);
      auto option = GeluOptionsT();
      option.approximate = params->approximate;
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_SQUARED_DIFFERENCE: {
      auto option = SquaredDifferenceOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_LOGISTIC:
    case BuiltinOperator_LOG:
    case BuiltinOperator_RSQRT:
    case BuiltinOperator_RELU:
    case BuiltinOperator_PRELU:
    case BuiltinOperator_TANH:
    case BuiltinOperator_RELU_N1_TO_1:
    case BuiltinOperator_RELU6: {
      break;
    }
    case BuiltinOperator_SLICE: {
      auto option = SliceOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_ABS: {
      auto option = AbsOptionsT();
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_CONCATENATION: {
      auto params = reinterpret_cast<TfLiteConcatenationParams*>(data);
      auto option = ConcatenationOptionsT();
      option.axis = params->axis;
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_MUL: {
      auto params = reinterpret_cast<TfLiteMulParams*>(data);
      auto option = MulOptionsT();
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      op->builtin_options.Set(option);
      break;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: {
      auto params = reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(data);
      auto option = UnidirectionalSequenceLSTMOptionsT();
      option.fused_activation_function = TfLiteActivationToSchemaActivation(params->activation);
      option.cell_clip = params->cell_clip;
      option.proj_clip = params->proj_clip;
      option.time_major = params->time_major;
      option.asymmetric_quantize_inputs = params->asymmetric_quantize_inputs;
      op->builtin_options.Set(option);
      break;
    }
    default: {
       throw "Can't support Operator.";
    }
  }
}

std::unique_ptr<ModelT> PrepareModel(TfLiteContext* context,
                                     const TfLiteDelegateParams* params) {
  ModelT *modelT = new ModelT;

  // Copy model version.
  modelT->version = 3;

  // Copy model buffers.
  // The model must always have the first buffer (sentinel) an empty buffer used for empty tensors/metadata.
  modelT->buffers.emplace_back(new BufferT);

  // Copy model graphs.
  modelT->subgraphs.reserve(1);

  // Create new graph.
  modelT->subgraphs.emplace_back(new SubGraphT);
  SubGraphT *graphNew = modelT->subgraphs.back().get();

  // Copy graph tensors.
  graphNew->tensors.reserve(context->tensors_size);
  for (int i = 0; i < context->tensors_size; i ++) {
    auto tensor = &context->tensors[i];
    // Create new tensor.
    graphNew->tensors.emplace_back(new TensorT);
    TensorT *tensorNew = graphNew->tensors.back().get();

    // Copy tensor data.
    tensorNew->shape = TfLiteArrayToVector<int, TfLiteIntArray>(tensor->dims);
    tensorNew->type = TfLiteTypeToSchemaType(tensor->type);
    if (tensor->bytes == 0 || tensor->data.raw == nullptr || tensor->allocation_type != kTfLiteMmapRo) {
      tensorNew->buffer = 0;
    } else {
      // Create new buffer.
      modelT->buffers.emplace_back(new BufferT);
      modelT->buffers.back()->data = std::vector<uint8_t>(tensor->data.uint8,
                       tensor->data.uint8 + tensor->bytes);
      tensorNew->buffer = modelT->buffers.size() - 1;
    }
    tensorNew->name = CharPtrToStr(tensor->name);
    if (tensor->quantization.type == kTfLiteNoQuantization) {
      tensorNew->quantization = std::unique_ptr<QuantizationParametersT>(nullptr);
    } else {
      tensorNew->quantization = std::unique_ptr<QuantizationParametersT>(new QuantizationParametersT);
      auto affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
      tensorNew->quantization->scale = TfLiteArrayToVector<float, TfLiteFloatArray>(affine_quantization->scale);
      tensorNew->quantization->zero_point = TfLiteArrayToVector<int64_t, TfLiteIntArray>(affine_quantization->zero_point);
      tensorNew->quantization->min = GetFloatMin(tensorNew->type, tensorNew->quantization->scale, tensorNew->quantization->zero_point);
      tensorNew->quantization->max = GetFloatMax(tensorNew->type, tensorNew->quantization->scale, tensorNew->quantization->zero_point);
      tensorNew->quantization->quantized_dimension = affine_quantization->quantized_dimension;
    }
    tensorNew->is_variable = tensor->is_variable;
    tensorNew->sparsity = std::unique_ptr<SparsityParametersT>(nullptr);
    tensorNew->shape_signature = TfLiteArrayToVector<int, TfLiteIntArray>(tensor->dims_signature);
  }

  // Copy graph inputs.
  auto inputs = params->input_tensors;
  graphNew->inputs.reserve(inputs->size);
  for (int i = 0; i < inputs->size; i ++) {
    if (context->tensors[inputs->data[i]].allocation_type != kTfLiteMmapRo) {
      graphNew->inputs.push_back(inputs->data[i]);
    }
  }

  // Copy graph outputs.
  auto outputs = params->output_tensors;
  graphNew->outputs.reserve(outputs->size);
  for (int i = 0; i < outputs->size; i ++) {
    graphNew->outputs.push_back(outputs->data[i]);
  }

  // Copy model operator codes.
  auto addModelOperatorCode = [&](const TfLiteRegistration* reg) -> uint32_t {
    // Reuse operator code if already added.
    for (size_t idx = 0; idx < modelT->operator_codes.size(); ++idx) {
      auto opcode = *(modelT->operator_codes[idx]);
      if (opcode.builtin_code == reg->builtin_code
          && reg->custom_name && opcode.custom_code == reg->custom_name
          && opcode.version == reg->version) {
        return static_cast<uint32_t>(idx);
      }
    }
    // Create new operator code.
    modelT->operator_codes.emplace_back(new OperatorCodeT);
    modelT->operator_codes.back()->builtin_code = (BuiltinOperator)reg->builtin_code;
    if (reg->custom_name)
      modelT->operator_codes.back()->custom_code = reg->custom_name;
    else
      modelT->operator_codes.back()->custom_code = "";
    modelT->operator_codes.back()->version = reg->version;
    return modelT->operator_codes.size() - 1;
  };

  // Copy graph operators.
  auto nodes_index = params->nodes_to_replace;
  graphNew->operators.reserve(nodes_index->size);
  for (int i = 0; i < nodes_index->size; i ++) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, nodes_index->data[i], &node, &reg);
    // Create new operator.
    graphNew->operators.emplace_back(new OperatorT);
    OperatorT *opNew = graphNew->operators.back().get();

    // Copy operator data.
    opNew->opcode_index = addModelOperatorCode(reg);
    opNew->inputs.reserve(node->inputs->size);
    for (int i = 0; i < node->inputs->size; i ++) {
      opNew->inputs.push_back(node->inputs->data[i]);
    }
    opNew->outputs.reserve(node->outputs->size);
    for (int i = 0; i < node->outputs->size; i ++) {
      opNew->outputs.push_back(node->outputs->data[i]);
    }
    SetBuiltinOptions(opNew, reg->builtin_code, node->builtin_data);
    opNew->custom_options_format = CustomOptionsFormat_FLEXBUFFERS;
    opNew->intermediates = TfLiteArrayToVector<int, TfLiteIntArray>(node->intermediates);
  }

  // Copy graph name.
  graphNew->name = "ethosu-delegate";
  // Copy model description.
  modelT->description = "ethosu-delegate";

  return std::move(std::unique_ptr<ModelT>(modelT));
}

size_t GetElementSize(TensorType type) {
  switch (type) {
  case TensorType::TensorType_FLOAT32:
    return 4;
  case TensorType::TensorType_FLOAT16:
    return 2;
  case TensorType::TensorType_INT32:
    return 4;
  case TensorType::TensorType_UINT8:
    return 1;
  case TensorType::TensorType_INT64:
    return 8;
  case TensorType::TensorType_STRING:
    // Here we return the size of one character.
    return 1;
  case TensorType::TensorType_BOOL:
    return 1;
  case TensorType::TensorType_INT16:
    return 2;
  case TensorType::TensorType_COMPLEX64:
    return 16;
  case TensorType::TensorType_INT8:
    return 1;
  case TensorType::TensorType_FLOAT64:
    return 8;
  }
  return 0;
}

size_t GetTensorDataSize(std::unique_ptr<tflite::TensorT> &tensor) {
  size_t size = GetElementSize(tensor->type);

  for(auto dim : tensor->shape) {
      size = size * dim;
  }

  return size;
}

ModelConverter::ModelConverter() {
#define XSTR(x) #x
#define STR(x) XSTR(x)
    needInitialization = !Py_IsInitialized();
    if (needInitialization) {
        Py_Initialize();
        if (dlopen(STR(PYTHON_LIB), RTLD_LAZY | RTLD_GLOBAL) == nullptr)
            throw "Failed to load python library";
    }
    pyModule = PyImport_ImportModule("ethosu.vela.vela");
    if (pyModule == nullptr)
        throw "Failed to import vela python package";
    pyCvtFunc = PyObject_GetAttrString(pyModule, "convert_bytes");
    if (pyCvtFunc == nullptr)
        throw "Failed to import vela python package";
}

ModelConverter *ModelConverter::GetSingleton() {
    static ModelConverter singleton;
    return &singleton;
}

std::unique_ptr<ModelT> ModelConverter::convert(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    auto model = PrepareModel(context, params);

    return std::move(convert(model.get()));
}

std::unique_ptr<ModelT> ModelConverter::convert(ModelT* model) {
    flatbuffers::FlatBufferBuilder fbb;
    flatbuffers::Offset<tflite::Model> root = tflite::Model::Pack(fbb, model);
    tflite::FinishModelBuffer(fbb, root);
    uint32_t size = fbb.GetSize();
    uint8_t *buff = fbb.GetBufferPointer();

    auto pyArgs = PyTuple_New(1);
    PyObject *pyMemory = PyMemoryView_FromMemory((char*)buff, size, PyBUF_READ);
    PyTuple_SetItem(pyArgs, 0, pyMemory);
    PyObject *pyValue = PyObject_CallObject(pyCvtFunc, pyArgs);

    Py_buffer *pyBuf = PyMemoryView_GET_BUFFER(pyValue);
    model = tflite::GetModel(pyBuf->buf)->UnPack();
    Py_DECREF(pyArgs);
    Py_DECREF(pyMemory);
    Py_DECREF(pyBuf);
    Py_DECREF(pyValue);

    return std::move(std::unique_ptr<ModelT>(model));
}

ModelConverter::~ModelConverter() {
    if (needInitialization) {
        Py_Finalize();
    }
}

}  // namespace ethosu
}  // namespace tflite
