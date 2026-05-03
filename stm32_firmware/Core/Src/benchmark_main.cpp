
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "main.h"
#include "lce_micro/lce_ops_micro.h"
#include <cstdio>
#include <cmath>

#include "model_data.h"
#include "person_image_data.h"
#include "no_person_image_data.h"

#ifdef ACCURACY_CHECK_MODE
#include "test_images_data.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

uint32_t g_op_cycles[128] = {0};
const char* g_op_names[128] = {nullptr};

#define TENSOR_ARENA_SIZE (470 * 1024)
uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(64)));

float g_gap_accumulator[512];

// Pointers set once by run_benchmark(); shared with POWER_MEASURE_MODE /
// ACCURACY_CHECK_MODE which always run after run_benchmark().
static const tflite::Model* g_model      = nullptr;
static const float*         g_fc_weights = nullptr;
static const float*         g_fc_biases  = nullptr;
// Resolver and model RAM live as static-locals in run_benchmark() so the
// compiler keeps the same addresses as the original code (preserves cache
// behaviour and the 799 ms latency). Other modes access them via pointers.
static tflite::MicroMutableOpResolver<50>* g_resolver_ptr = nullptr;

void dwt_init() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
uint32_t dwt_get_cycles() { return DWT->CYCCNT; }

void print_float_manual(const char* label, float f) {
    if (std::isnan(f)) { printf("%sNaN", label); return; }
    if (std::isinf(f)) { printf("%sInf", label); return; }
    if (f < 0) { printf("-"); f = -f; }
    int32_t vi = (int32_t)f;
    int32_t vf = (int32_t)((f - vi) * 1000000);
    printf("%s%ld.%06ld", label, (long)vi, (long)vf);
}

extern UART_HandleTypeDef huart1;
int __io_putchar(int ch) {
  HAL_UART_Transmit(&huart1, (uint8_t*)&ch, 1, 10);
  return ch;
}

// Run one full inference.  Exactly one of img_float / img_uint8 must be
// non-null.  img_float: pre-normalised floats in [-1,1].
// img_uint8: raw 0-255 bytes, normalised here as (v/127.5f)-1.0f.
static void infer_one(const float* img_float, const uint8_t* img_uint8,
                      float* p_person, float* p_no_person)
{
    memset(g_gap_accumulator, 0, sizeof(g_gap_accumulator));

    for (int th = 0; th < 4; ++th) {
        for (int tw = 0; tw < 4; ++tw) {
            memset(tensor_arena, 0, TENSOR_ARENA_SIZE);
            tflite::MicroInterpreter interp(g_model, *g_resolver_ptr,
                                            tensor_arena, TENSOR_ARENA_SIZE);
            interp.AllocateTensors();
            float* inp = interp.input(0)->data.f;

            for (int r = 0; r < 32; ++r) {
                int src_off = (th * 32 + r) * 128 * 3 + tw * 32 * 3;
                float* dst  = inp + r * 32 * 3;
                if (img_float) {
                    for (int j = 0; j < 32 * 3; ++j)
                        dst[j] = img_float[src_off + j];
                } else {
                    for (int j = 0; j < 32 * 3; ++j)
                        dst[j] = (img_uint8[src_off + j] / 127.5f) - 1.0f;
                }
            }

            interp.Invoke();
            float* td = interp.output(0)->data.f;
            for (int c = 0; c < 512; ++c) g_gap_accumulator[c] += td[c];
        }
    }
    for (int i = 0; i < 512; ++i) g_gap_accumulator[i] /= 16.0f;

    float scores[2] = {0, 0};
    for (int o = 0; o < 2; ++o) {
        scores[o] = g_fc_biases[o];
        for (int in = 0; in < 512; ++in)
            scores[o] += g_gap_accumulator[in] * g_fc_weights[o * 512 + in];
    }
    float mx   = scores[0] > scores[1] ? scores[0] : scores[1];
    float e0   = expf(scores[0] - mx), e1 = expf(scores[1] - mx);
    float s    = e0 + e1;
    *p_person    = e1 / s;
    *p_no_person = e0 / s;
}

// ---------------------------------------------------------------
void run_benchmark() {
    printf("\r\n========================================\r\n");
    printf("  NAS-BNN 128x128 FULL TILING (Internal)\r\n");
    printf("  Board: STM32H7B3I-DK\r\n");
    printf("  Arena: %d bytes (%d KiB) Internal SRAM\r\n",
           TENSOR_ARENA_SIZE, TENSOR_ARENA_SIZE / 1024);
    printf("  CPU:   %lu MHz\r\n", (unsigned long)(SystemCoreClock / 1000000UL));
    printf("========================================\r\n");

    SCB_EnableDCache();
    dwt_init();

    // Static-locals: keep same addresses as original code to preserve
    // cache-set mapping and the measured 799 ms latency.
    static tflite::MicroMutableOpResolver<50> resolver;
    static bool resolver_init = false;
    if (!resolver_init) {
        lce_micro::RegisterLCECustomOps(resolver);
        resolver.AddAdd(); resolver.AddAveragePool2D(); resolver.AddConcatenation();
        resolver.AddConv2D(); resolver.AddFullyConnected(); resolver.AddPadV2();
        resolver.AddShape(); resolver.AddPack(); resolver.AddPrelu();
        resolver.AddStridedSlice(); resolver.AddReshape(); resolver.AddMul();
        resolver.AddMean(); resolver.AddSoftmax(); resolver.AddQuantize();
        resolver.AddDepthwiseConv2D(); resolver.AddDequantize();
        resolver_init = true;
    }

    static uint8_t model_ram[430000] __attribute__((aligned(16)));
    memcpy(model_ram, g_model_data, g_model_data_len);

    const tflite::Model* m_ram = tflite::GetModel(model_ram);
    auto* subgraph  = const_cast<tflite::Model*>(m_ram)->subgraphs()->Get(0);
    auto* tensors   = subgraph->tensors();
    auto* operators = const_cast<flatbuffers::Vector<
                          flatbuffers::Offset<tflite::Operator>>*>(subgraph->operators());
    uint32_t* op_count_ptr = (uint32_t*)operators;
    auto* buffers = m_ram->buffers();

    for (int i = 0; i < (int)tensors->size(); ++i) {
        auto* t = tensors->Get(i);
        auto* s = t->shape();
        if (s && s->size() == 4) {
            auto* buf_entry = buffers->Get(t->buffer());
            bool is_weight = (buf_entry->data() != nullptr && buf_entry->data()->size() > 0);
            if (!is_weight) {
                int h = s->Get(1); int w = s->Get(2);
                if (h >= 4) const_cast<flatbuffers::Vector<int32_t>*>(s)->Mutate(1, h / 4);
                if (w >= 4) const_cast<flatbuffers::Vector<int32_t>*>(s)->Mutate(2, w / 4);
            }
        }
    }
    const_cast<flatbuffers::Vector<int32_t>*>(subgraph->outputs())->Mutate(0, 157);
    *op_count_ptr = 80;

    auto* w_tensor = tensors->Get(66);
    auto* b_tensor = tensors->Get(67);

    // Publish shared state so infer_one() and other modes can use them
    g_model      = m_ram;
    g_fc_weights = reinterpret_cast<const float*>(
                       buffers->Get(w_tensor->buffer())->data()->data());
    g_fc_biases  = reinterpret_cast<const float*>(
                       buffers->Get(b_tensor->buffer())->data()->data());
    g_resolver_ptr = &resolver;

    // Arena probe (outside timed loop)
    size_t arena_used = 0;
    {
        memset(tensor_arena, 0, TENSOR_ARENA_SIZE);
        tflite::MicroInterpreter probe(m_ram, resolver, tensor_arena, TENSOR_ARENA_SIZE);
        if (probe.AllocateTensors() != kTfLiteOk) {
            printf("ERROR: AllocateTensors failed!\r\n"); return;
        }
        arena_used = probe.arena_used_bytes();
        printf("Arena used: %u / %u bytes (%lu KiB)\r\n",
               (unsigned)arena_used, (unsigned)TENSOR_ARENA_SIZE,
               (unsigned long)(arena_used / 1024));
    }

    const int NUM_WARMUP = 2, NUM_TIMED = 10;
    uint64_t total_cycles = 0;
    uint32_t min_cycles = UINT32_MAX, max_cycles = 0;
    float final_pp = 0.0f, final_pnp = 0.0f;

    for (int run = 0; run < NUM_WARMUP + NUM_TIMED; ++run) {
        memset(g_gap_accumulator, 0, sizeof(g_gap_accumulator));
        uint32_t start = dwt_get_cycles();

        for (int th = 0; th < 4; ++th) {
            for (int tw = 0; tw < 4; ++tw) {
                memset(tensor_arena, 0, TENSOR_ARENA_SIZE);
                tflite::MicroInterpreter interpreter(m_ram, resolver,
                                                     tensor_arena, TENSOR_ARENA_SIZE);
                interpreter.AllocateTensors();
                TfLiteTensor* input = interpreter.input(0);
                float* inp = input->data.f;
                for (int r = 0; r < 32; ++r) {
                    int src_off = (th * 32 + r) * 128 * 3 + tw * 32 * 3;
                    float* dst = inp + r * 32 * 3;
                    for (int j = 0; j < 32 * 3; ++j)
                        dst[j] = person_image_data[src_off + j];
                }
                interpreter.Invoke();
                float* td = interpreter.output(0)->data.f;
                for (int c = 0; c < 512; ++c) g_gap_accumulator[c] += td[c];
            }
        }
        for (int i = 0; i < 512; ++i) g_gap_accumulator[i] /= 16.0f;

        float scores[2] = {0, 0};
        for (int o = 0; o < 2; ++o) {
            scores[o] = g_fc_biases[o];
            for (int in = 0; in < 512; ++in)
                scores[o] += g_gap_accumulator[in] * g_fc_weights[o * 512 + in];
        }
        float mx = scores[0] > scores[1] ? scores[0] : scores[1];
        float e0 = expf(scores[0]-mx), e1 = expf(scores[1]-mx), s = e0+e1;
        final_pp  = e1 / s;
        final_pnp = e0 / s;

        uint32_t cycles = dwt_get_cycles() - start;
        if (run < NUM_WARMUP) {
            printf("[warmup %d] cycles=%lu\r\n", run, (unsigned long)cycles);
        } else {
            total_cycles += cycles;
            if (cycles < min_cycles) min_cycles = cycles;
            if (cycles > max_cycles) max_cycles = cycles;
            uint64_t ms = (uint64_t)cycles * 100000ULL / SystemCoreClock;
            printf("[run %2d] %lu cycles = %lu.%02lu ms\r\n",
                   run - NUM_WARMUP, (unsigned long)cycles,
                   (unsigned long)(ms/100), (unsigned long)(ms%100));
        }
    }

    uint64_t avg = total_cycles / NUM_TIMED;
    uint64_t avg_ms = avg * 100000ULL / SystemCoreClock;
    uint64_t min_ms = (uint64_t)min_cycles * 100000ULL / SystemCoreClock;
    uint64_t max_ms = (uint64_t)max_cycles * 100000ULL / SystemCoreClock;

    printf("\r\n========================================\r\n  RESULTS\r\n");
    printf("========================================\r\n");
    printf("Model Flash:    %lu bytes (%lu KiB)\r\n",
           (unsigned long)g_model_data_len,
           (unsigned long)(g_model_data_len / 1024));
    printf("Arena used:     %u bytes (%lu KiB)\r\n",
           (unsigned)arena_used, (unsigned long)(arena_used / 1024));
    printf("Latency avg:    %lu.%02lu ms\r\n",
           (unsigned long)(avg_ms/100), (unsigned long)(avg_ms%100));
    printf("Latency min:    %lu.%02lu ms\r\n",
           (unsigned long)(min_ms/100), (unsigned long)(min_ms%100));
    printf("Latency max:    %lu.%02lu ms\r\n",
           (unsigned long)((uint64_t)max_cycles*100000ULL/SystemCoreClock/100),
           (unsigned long)((uint64_t)max_cycles*100000ULL/SystemCoreClock%100));
    printf("Throughput:     %lu.%02lu fps\r\n",
           (unsigned long)(100000ULL*100/(avg_ms?avg_ms:1)/100),
           (unsigned long)(100000ULL*100/(avg_ms?avg_ms:1)%100));
    printf("Final scores:   [Person] "); print_float_manual("", final_pp);
    printf("  [No-Person] "); print_float_manual("", final_pnp);
    printf("\r\n========================================\r\n");
}

// ---------------------------------------------------------------
// ACCURACY_CHECK_MODE diagnostic
// ---------------------------------------------------------------
#ifdef ACCURACY_CHECK_MODE
void run_accuracy_check() {
    const int PERSON_IDX = 1;

    printf("\r\n========================================\r\n");
    printf("  ACCURACY CHECK\r\n");
    printf("========================================\r\n");

    // ---- Float sanity check: known no-person image (same format as benchmark) ----
    {
        float pp, pnp;
        infer_one(no_person_image_data, nullptr, &pp, &pnp);
        int pred = (pp >= 0.5f) ? 1 : 0;
        printf("FLOAT sanity [no_person_image_data]: pred=%s p_person=",
               pred==1 ? "P (ERR)" : "NP(OK)");
        print_float_manual("", pp);
        printf("\r\n");

        infer_one(person_image_data, nullptr, &pp, &pnp);
        pred = (pp >= 0.5f) ? 1 : 0;
        printf("FLOAT sanity [person_image_data]:    pred=%s p_person=",
               pred==1 ? "P (OK) " : "NP(ERR)");
        print_float_manual("", pp);
        printf("\r\n");
    }
    printf("----------------------------------------\r\n");

    // ---- Uint8 test images ----
    int correct = 0, tp = 0, fp = 0, fn = 0, tn = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; ++i) {
        float pp, pnp;
        infer_one(nullptr, test_image_data[i], &pp, &pnp);
        int pred  = (pp >= 0.5f) ? 1 : 0;
        int label = test_image_labels[i];
        int ok    = (pred == label);
        correct  += ok;
        if (pred==PERSON_IDX && label==PERSON_IDX) tp++;
        else if (pred==PERSON_IDX && label!=PERSON_IDX) fp++;
        else if (pred!=PERSON_IDX && label==PERSON_IDX) fn++;
        else tn++;

        printf("  [%2d] label=%s pred=%s (%s)  p_person=",
               i,
               label==1 ? "P " : "NP",
               pred ==1 ? "P " : "NP",
               ok ? "OK " : "ERR");
        print_float_manual("", pp);
        printf("\r\n");
    }

    int prec = (tp+fp>0) ? (tp*10000/(tp+fp)) : 0;
    int rec  = (tp+fn>0) ? (tp*10000/(tp+fn)) : 0;
    int f1   = (prec+rec>0) ? (2*prec*rec/(prec+rec)) : 0;
    printf("----------------------------------------\r\n");
    printf("  Correct:   %d / %d\r\n", correct, NUM_TEST_IMAGES);
    printf("  Accuracy:  %d.%02d%%\r\n",
           correct*100/NUM_TEST_IMAGES,
           (correct*10000/NUM_TEST_IMAGES)%100);
    printf("  Precision: %d.%02d%%  Recall: %d.%02d%%  F1: %d.%02d%%\r\n",
           prec/100, prec%100, rec/100, rec%100, f1/100, f1%100);
    printf("========================================\r\n");
}
#endif

// ---------------------------------------------------------------
// POWER_MEASURE_MODE: continuous inference after benchmark
// ---------------------------------------------------------------
#ifdef POWER_MEASURE_MODE
static void run_inference_continuous(void) {
    float pp, pnp;
    while (1) { infer_one(person_image_data, nullptr, &pp, &pnp); (void)pp; (void)pnp; }
}
#endif

// ---------------------------------------------------------------
void benchmark_main(void) {
    run_benchmark();

#ifdef ACCURACY_CHECK_MODE
    run_accuracy_check();
#endif

#ifdef POWER_MEASURE_MODE
    HAL_GPIO_WritePin(GPIOG, GPIO_PIN_11, GPIO_PIN_SET);
    run_inference_continuous();
#else
    while (1) { HAL_GPIO_TogglePin(GPIOG, GPIO_PIN_11); HAL_Delay(500); }
#endif
}

#ifdef __cplusplus
}
#endif
