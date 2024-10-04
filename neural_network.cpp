
extern "C" {
#include <stdlib.h>
#include <stdio.h>

#include "doorhandle.h"
}

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

#include "neural_network.h"

// #define DEBUG_COLOR_NN__PRINT    /* Print debug information */
// #define DEBUG_COLOR_NN__PROFILER /* Run the profiler */

/* Color classification model to use */

// This model recognizes all colors
//#define CC_MODEL cc_model_01_FINAL_dis2__TUNER_32_96__epoch16_optimized_default_tflite

// This model recognizes the base colors, but was trained on all colors while grouping them into the base colors
//#define CC_MODEL cc_model_01_FINAL_dis2__TUNER_simplified_minDropout_maxEpoch96_optimized_default_tflite

// Same as above, but with a different balancing of the training data
//#define CC_MODEL cc_model_01_FINAL_dis2__TUNER_simplified_max2000_noBalancing_optimized_default_tflite // THIS ONE WORKS BEST

// This model was trained only on the base colors (as in the early days, where it worked quite well) with different light conditions
// #define CC_MODEL cc_model_01_FINAL_dis2__TUNER_simplified_baseColors_max2000_noBalancing_optimized_default_tflite // doesn't work too well, better skip

// Same as above, but without different lighting conditions (though validated against them); this would equal the model from the early days
// #define CC_MODEL cc_model_01_FINAL_dis2__TUNER_simplified_baseColors_offset0_optimized_default_tflite // doesn't work well, better skip

// Trained only on and validated against the base colors; this would come closest to the model from the early days
// #define CC_MODEL cc_model_01_FINAL_dis2__TUNER_simplified_baseColors_offset0_split_optimized_default_tflite // doesn't work well, better skip 

#define CC_MODEL cc_model_BASIC_Doorhandle_float_tflite

namespace {
  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

TfLiteStatus Setup() {
  /* Load the model */
  const tflite::Model* model = tflite::GetModel(CC_MODEL);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  /* Initialize an operations resolver and add utilized operations */
  static tflite::MicroMutableOpResolver<6> op_resolver;
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED()));

  /* Add activation functions, as required by the model definition */
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax(tflite::Register_SOFTMAX()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
//  TF_LITE_ENSURE_STATUS(op_resolver.AddLeakyRelu());
//  TF_LITE_ENSURE_STATUS(op_resolver.AddExp());

  /* Create an arena for storing the values calculated during inference.
   * A reasonable size can be found by running ProfileMemoryAndLatency(). */
  constexpr int kTensorArenaSize = 2048;
  static uint8_t tensor_arena[kTensorArenaSize];

  /* Initialize the interpreter */
  static tflite::MicroInterpreter static_interpreter(
    model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  /* Allocate memory from the tensor_arena for the model's tensors*/
  TF_LITE_ENSURE_STATUS(interpreter->AllocateTensors());

  /* Reference input and output vectors for ease of use */
  input = interpreter->input(0);
  TFLITE_CHECK_NE(input, nullptr);

  output = interpreter->output(0);
  TFLITE_CHECK_NE(output, nullptr);

  return kTfLiteOk;
}

TfLiteStatus PerformInference(CapsenseValues_t* cap_values, int8_t* index, uint8_t* probability) {
  constexpr uint16_t divider = 24575; /* Max. 16-bit Capsense value */

  /* Fill the input vector with data */
  input->data.f[0] = ((float)cap_values->Frame1_M / divider);
  input->data.f[1] = ((float)cap_values->Frame1_I / divider);
  input->data.f[2] = ((float)cap_values->Frame1_O / divider);
  input->data.f[3] = ((float)cap_values->Frame2_M / divider);
  input->data.f[4] = ((float)cap_values->Frame2_I / divider);
  input->data.f[5] = ((float)cap_values->Frame2_O / divider);
  input->data.f[6] = ((float)cap_values->Frame3_M / divider);
  input->data.f[7] = ((float)cap_values->Frame3_I / divider);
  input->data.f[8] = ((float)cap_values->Frame3_O / divider);

  /* Run inference */
  TF_LITE_ENSURE_STATUS(interpreter->Invoke());

  /* The length of the output vector equals the amount of all available outputs/colors.
   * The values of the output vector indicate their probability. Therefore, by searching
   * for the element with the highest value, the best prediction can be found. */
  auto begin = output->data.f;
  auto end = std::next(begin, output->dims->data[1]);
  auto max_val = std::max_element(begin, end);

  /* Get the index of the highest value */
  uint8_t status_index = std::distance(output->data.f, max_val);
  *index = status_index;

  /* Pass the probability of the classification */
  *probability = (uint8_t)(*max_val * 100);

#ifdef DEBUG_COLOR_NN__PRINT
  for (size_t i = 0; i < output->dims->data[1]; i++)
  {
    printf("output[%d] = %f\r\n", i, begin[i]);
  }
  printf("status_index: %d\r\n", status_index);
#endif

  return kTfLiteOk;
}

#ifdef DEBUG_COLOR_NN__PROFILER
/* Run the profiler. This gives hints for right-sizing the tensor arena. */
TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;

  /* Load the model */
  const tflite::Model* model = tflite::GetModel(CC_MODEL);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  /* Initialize an operations resolver and add utilized operations */
  static tflite::MicroMutableOpResolver<5> op_resolver;
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED()));

  /* Add activation functions, as required by the model definition */
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax(tflite::Register_SOFTMAX()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());

  /* Create an arena for storing the values calculated during inference. */
  constexpr int kTensorArenaSize = 3000;
  static uint8_t tensor_arena[kTensorArenaSize];

  constexpr int kNumResourceVariables = 24;

  /* Initialize the interpreter */
  tflite::RecordingMicroAllocator* recording_allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter recording_interpreter(
      model, op_resolver, recording_allocator,
      tflite::MicroResourceVariables::Create(recording_allocator, kNumResourceVariables),
      &profiler);

  /* Allocate all tensors */
  TF_LITE_ENSURE_STATUS(recording_interpreter.AllocateTensors());

  /* Fill the input vector with data */
  recording_interpreter.input(0)->data.f[0] = 0.1;
  recording_interpreter.input(0)->data.f[1] = 0.2;
  recording_interpreter.input(0)->data.f[2] = 0.3;
  recording_interpreter.input(0)->data.f[3] = 0.4;
  recording_interpreter.input(0)->data.f[0] = 0.5;
  recording_interpreter.input(0)->data.f[1] = 0.6;
  recording_interpreter.input(0)->data.f[2] = 0.7;
  recording_interpreter.input(0)->data.f[3] = 0.8;
  recording_interpreter.input(0)->data.f[3] = 0.9;

  /* Run inference */
  TF_LITE_ENSURE_STATUS(recording_interpreter.Invoke());

  /* Print debug information */
  MicroPrintf("");
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");
  recording_interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}
#endif

// #include "cy_scb_uart.h"
// #include "cycfg_peripherals.h"
void debug_log_printf(const char* s)
{
//  printf(s);
//	extern void ei_printf(const char *format, ...);
//	ei_printf("%s",s);
//	Cy_SCB_UART_PutString(CYBSP_UART_HW,s);
}

extern "C" int CapClassificationSetup(void)
{
  tflite::InitializeTarget();
  RegisterDebugLogCallback(debug_log_printf);

  #ifndef DEBUG_COLOR_NN__PROFILER
    TF_LITE_ENSURE_STATUS(Setup());
  #else
    TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  #endif

  return 0;
}

extern "C" int CapClassificationPerformInference(CapsenseValues_t* cap_values, int8_t* cap_index, uint8_t* probability)
{
#ifndef DEBUG_COLOR_NN__PROFILER
  TF_LITE_ENSURE_STATUS(PerformInference(cap_values, cap_index, probability));
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}
