/*
 * Copyright 2022-2023 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#ifndef TENSORFLOW_LITE_DELEGATES_ETHOSU_DELEGATE_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_ETHOSU_DELEGATE_UTILS_H_

#include <Python.h>
#include "flatbuffers/flexbuffers.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"


namespace tflite {
namespace ethosu {

class ModelConverter {
public:
    ~ModelConverter();
    static ModelConverter *GetSingleton();

    std::unique_ptr<ModelT> convert(TfLiteContext* context,
                    const TfLiteDelegateParams* params);
    std::unique_ptr<ModelT> convert(ModelT* model);
private:
    ModelConverter();

    bool needInitialization;
    PyObject *pyModule;
    PyObject *pyCvtFunc;
};

std::unique_ptr<ModelT> PrepareModel(TfLiteContext* context,
                                     const TfLiteDelegateParams* params);
std::unique_ptr<tflite::ModelT> readTFLiteModel(const std::string &filename);
void writeTFLiteModel(const tflite::ModelT *model, const std::string &filename);

size_t GetTensorDataSize(std::unique_ptr<tflite::TensorT> &tensor);

bool IsNodeSupportedByEthosU(TfLiteContext* context,
                              const TfLiteNode* node,
                              int32_t builtin_code);
} //namespace ethosu
} //namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_ETHOSU_DELEGATE_UTILS_H_
