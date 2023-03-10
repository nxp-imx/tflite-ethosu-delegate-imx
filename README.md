# TfLite-ethosu-delegate
TfLite-ethosu-delegate is an delegate for tensorflow lite.

# Use tflite-ethosu-delegate

## Prepare source code
```sh
mkdir wksp && cd wksp
git clone https://github.com/nxpmicro/tflite-ethosu-delegate.git
# tensorflow is optional, it will be downloaded automatically if not present
git clone https://github.com/tensorflow/tensorflow.git
```
# Build from source with cmake

```sh
# set the toolchain env
source /PATH_TO_TOOLCHAIN/environment-setup-cortexa53-crypto-poky-linux

# build the delegate
cd tflite-ethosu-delegate
mkdir build && cd build
cmake ..
make -j12

# benchmark_model
make benchmark_model -j12
# label_image
make lable_image -j12
```

If you would like to build using local version of tensorflow, you can use `FETCHCONTENT_SOURCE_DIR_TENSORFLOW` cmake variable. Point this variable to your tensorflow tree. For additional details on this variable please see the [official cmake documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html#command:fetchcontent_populate)

``` sh
cmake -DFETCHCONTENT_SOURCE_DIR_TENSORFLOW=/my/copy/of/tensorflow \
    -DOTHER_CMAKE_DEFINES...\
    ..
```
After cmake execution completes, build and run as usual.

## Enable external delegate support in benchmark_model/label_image

If tensorflow source code downloaded by cmake, you can find it in <build_output_dir>/_deps/tensorflow-src

## Run
```sh
./benchmark_model --external_delegate_path=<patch_to_libethosu_delegate.so> --graph=<tflite_model.tflite>
# If you would like to enable PMU counter
./benchmark_model --external_delegate_path=<patch_to_libethosu_delegate.so> \
                  --external_delegate_options='enable_cycle_counter:true;pmu_event0:3;pmu_event1:4;pmu_event2:5;pmu_event3:6' \
                  --graph=<tflite_model.tflite>
```

BuiltinOperator.SQUEEZE			kTfLiteBuiltinSqueeze
BuiltinOperator.SHAPE			kTfLiteBuiltinShape
BuiltinOperator.SPLIT_V			kTfLiteBuiltinSplitV
BuiltinOperator.EXP			kTfLiteBuiltinExp
BuiltinOperator.EXPAND_DIMS		kTfLiteBuiltinExpandDims
BuiltinOperator.PRELU			kTfLiteBuiltinPrelu
BuiltinOperator.ADD			kTfLiteBuiltinAdd
BuiltinOperator.AVERAGE_POOL_2D		kTfLiteBuiltinAveragePool2d
BuiltinOperator.MAX_POOL_2D		kTfLiteBuiltinMaxPool2d
BuiltinOperator.CONV_2D			kTfLiteBuiltinConv2d
BuiltinOperator.DEPTHWISE_CONV_2D	kTfLiteBuiltinDepthwiseConv2d
BuiltinOperator.TRANSPOSE_CONV		kTfLiteBuiltinTransposeConv
BuiltinOperator.FULLY_CONNECTED		kTfLiteBuiltinFullyConnected
BuiltinOperator.RESHAPE			kTfLiteBuiltinReshape
BuiltinOperator.CONCATENATION		kTfLiteBuiltinConcatenation
BuiltinOperator.PAD			kTfLiteBuiltinPad
BuiltinOperator.SPLIT			kTfLiteBuiltinSplit
BuiltinOperator.UNPACK			kTfLiteBuiltinUnpack
BuiltinOperator.ABS			kTfLiteBuiltinAbs
BuiltinOperator.SOFTMAX			kTfLiteBuiltinSoftmax
BuiltinOperator.HARD_SWISH		kTfLiteBuiltinHardSwish
BuiltinOperator.PACK			kTfLiteBuiltinPack
BuiltinOperator.STRIDED_SLICE		kTfLiteBuiltinStridedSlice
BuiltinOperator.RELU			kTfLiteBuiltinRelu
BuiltinOperator.RELU6			kTfLiteBuiltinRelu6
BuiltinOperator.RELU_N1_TO_1		kTfLiteBuiltinReluN1To1
BuiltinOperator.LEAKY_RELU		kTfLiteBuiltinLeakyRelu
BuiltinOperator.SUB			kTfLiteBuiltinSub
BuiltinOperator.MEAN			kTfLiteBuiltinMean
BuiltinOperator.QUANTIZE		kTfLiteBuiltinQuantize
BuiltinOperator.MAXIMUM			kTfLiteBuiltinMaximum
BuiltinOperator.MUL			kTfLiteBuiltinMul
BuiltinOperator.TANH			kTfLiteBuiltinTanh
BuiltinOperator.MINIMUM			kTfLiteBuiltinMinimum
BuiltinOperator.RESIZE_BILINEAR		kTfLiteBuiltinResizeBilinear
BuiltinOperator.RESIZE_NEAREST_NEIGHBOR	kTfLiteBuiltinResizeNearestNeighbor
BuiltinOperator.SLICE			kTfLiteBuiltinSlice
BuiltinOperator.LOGISTIC		kTfLiteBuiltinLogistic
