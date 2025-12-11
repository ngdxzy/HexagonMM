#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>

#include "utils/loadTensor.hpp"
#include "utils/helperFunction.hpp"
#include "utils/errorMetrics.hpp"
#include "test_utils.hpp"

bool test_mul_mat_q8_0_f32(ggml_backend_t& backend, std::string gguf_file);
bool test_mul_mat_f16_f32(ggml_backend_t& backend);

int main(int argc, char** argv) {
    
    bool all_passed = true;
    
    ggml_backend_t backend;
    if (!initialize_backend(backend)) {
        GGML_LOG_ERROR("Failed to initialize backend\n");
        return 1;
    }


    if (!test_mul_mat_f16_f32(backend)) {
        GGML_LOG_ERROR("Failed to test mul mat f16 f32\n");
        all_passed = false;
    }

    if (argc > 1 && !test_mul_mat_q8_0_f32(backend, argv[1])) {
        GGML_LOG_ERROR("Failed to test mul mat q8_0 f32\n");
        all_passed = false;
    }


    // Cleanup
    ggml_backend_free(backend);
    
    GGML_LOG_INFO("\n========================================\n");
    if (all_passed) {
        GGML_LOG_INFO("All tests PASSED! ✓\n");
    } else {
        GGML_LOG_INFO("Some tests FAILED! ✗\n");
    }
    GGML_LOG_INFO("========================================\n");
    
    return 0;
}


bool test_mul_mat_q8_0_f32(ggml_backend_t& backend, std::string gguf_file) {
    const int L = 256;
    const int Do = 1024;
    const int Di = 3072;
    GGML_LOG_INFO("\n=== Testing Matrix Multiplication Q8_0 x F32 ===\n");
    GGUFTensorManager* tensor_manager = new GGUFTensorManager(gguf_file, backend);
    struct ggml_tensor* weight = tensor_manager->get_tensor("blk.0.ffn_down.weight");
    if (!weight) {
        GGML_LOG_ERROR("Failed to load blk.2.ffn_down.weight\n");
    } else {
        GGML_LOG_INFO("blk.2.ffn_down.weight loaded successfully\n");
    }

    GGML_LOG_INFO("Loaded weight tensor with shape: [%d, %d]\n", (int)weight->ne[1], (int)weight->ne[0]);
    struct ggml_init_params ctx_params = {
        .mem_size = ggml_nbytes(weight) + L * Do * sizeof(float) + Di * L * sizeof(float),
        .mem_buffer = nullptr,
        .no_alloc = true
    };
    struct ggml_context* local_ctx = ggml_init(ctx_params);
    
    // create x
    struct ggml_tensor* x = ggml_new_tensor_2d(local_ctx, GGML_TYPE_F32, Di, L);
    if (!x) {
        GGML_LOG_ERROR("Failed to create x tensor\n");
        return false;
    }

    // create w
    std::cout << "Creating w tensor with shape: [" << weight->ne[0] << ", " << weight->ne[1] << "]" << std::endl;
    std::cout << "Weight type: " << weight->type << std::endl;
    std::cout << "Weight data: " << weight->data << std::endl;
    std::cout << "Weight ne: " << weight->ne[0] << ", " << weight->ne[1] << std::endl;
    std::cout << "Weight nb: " << weight->nb[1] << ", " << weight->nb[1] << std::endl;
    struct ggml_tensor* w = ggml_new_tensor_2d(local_ctx, weight->type, weight->ne[0], weight->ne[1]);
    if (!w) {
        GGML_LOG_ERROR("Failed to create w tensor\n");
        return false;
    }

    // pass though mm for graph building
    struct ggml_tensor* y = ggml_mul_mat(local_ctx, w, x);
    if (!y) {
        GGML_LOG_ERROR("Failed to create result tensor\n");
        return false;
    }

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(local_ctx);
    ggml_build_forward_expand(gf, y);
    
    // Allocate buffers on the backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(local_ctx, backend);
    if (!buffer) {
        GGML_LOG_ERROR("Failed to allocate backend buffer\n");
        ggml_free(local_ctx);
        return false;
    }
    
    // copy data
    memcpy(w->data, weight->data, ggml_nbytes(weight));

    // release the tensor manager
    delete tensor_manager;

    GGML_LOG_INFO("Created y tensor with shape: [%d, %d]\n", (int)y->ne[1], (int)y->ne[0]);
    GGML_LOG_INFO("Created w tensor with shape: [%d, %d]\n", (int)w->ne[1], (int)w->ne[0]);
    GGML_LOG_INFO("Created x tensor with shape: [%d, %d]\n", (int)x->ne[1], (int)x->ne[0]);

    // load x
    std::ifstream x_file("../gguf/x.bin", std::ios::binary);
    if (!x_file.is_open()) {
        GGML_LOG_ERROR("Failed to open x file\n");
        ggml_free(local_ctx);
        ggml_backend_buffer_free(buffer);
        return false;
    }
    x_file.read((char*)x->data, Di * L * sizeof(float));
    x_file.close();

    std::cout << "x: " << std::endl;
    print_matrix((float*)x->data, x->ne[1], x->ne[0], 10, 10);

    // load y reference
    float* y_ref_fp32 = (float*)malloc(Do * L * sizeof(float));
    std::ifstream y_file("../gguf/y.bin", std::ios::binary);
    if (!y_file.is_open()) {
        GGML_LOG_ERROR("Failed to open y file\n");
        ggml_free(local_ctx);
        ggml_backend_buffer_free(buffer);
        return false;
    }
    y_file.read((char*)y_ref_fp32, Do * L * sizeof(float));
    y_file.close();

    std::cout << "y reference: " << std::endl;
    print_matrix(y_ref_fp32, L, Do, 10, 10);

    
    // test if the weight is loaded correctly
    {
        // dequant the weight to FP32 array
        float* weight_data_fp32 = (float*)malloc(w->ne[0] * w->ne[1] * sizeof(float));
        std::cout << "Weight data: " << w->data << std::endl;
        std::cout << "Weight shape: " << w->ne[0] << ", " << w->ne[1] << std::endl;
        for (int row = 0; row < w->ne[1]; row++) {
            signed char* q = (signed char*)((uint8_t*)w->data + row * w->nb[1]);
            ggml_fp16_t* scale = (ggml_fp16_t*)(q + w->ne[0]);
            for (int col = 0; col < w->ne[0] / 32; col++) {
                float scale_value = ggml_fp16_to_fp32(scale[col]);
                for (int i = 0; i < 32; i++) {
                    weight_data_fp32[row * w->ne[0] + col * 32 + i] = q[col * 32 + i] * scale_value;
                }
            }
        }

        std::cout << "Dequantized weight tensor:\n";
        print_matrix(weight_data_fp32, w->ne[1], w->ne[0], 10, 10);
        std::ifstream reference_weight("../gguf/w.bin", std::ios::binary);
        if (!reference_weight.is_open()) {
            GGML_LOG_ERROR("Failed to open reference weight file\n");
            ggml_free(local_ctx);
            ggml_backend_buffer_free(buffer);
            return false;
        }
        float* reference_weight_data_fp32 = (float*)malloc(w->ne[0] * w->ne[1] * sizeof(float));
        reference_weight.read((char*)reference_weight_data_fp32, w->ne[0] * w->ne[1] * sizeof(float));
        reference_weight.close();
        std::cout << "Reference weight tensor:\n";
        print_matrix(reference_weight_data_fp32, w->ne[1], w->ne[0], 10, 10);

        ErrorMetrics metrics = calculate_error_metrics(weight_data_fp32, reference_weight_data_fp32, w->ne[0] * w->ne[1]);
        print_error_metrics(metrics);
        if (metrics.cosine_similarity < 0.99) {
            GGML_LOG_ERROR("Cosine similarity is less than 0.99\n");
            ggml_free(local_ctx);
            ggml_backend_buffer_free(buffer);
            return false;
        }
        free(reference_weight_data_fp32);
        free(weight_data_fp32);
    }
    // launch the graph
    std::cout << "Launching graph..." << std::endl;
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(local_ctx);
        return false;
    }
    std::cout << "Graph launched successfully" << std::endl;
    // get the result
    float* y_data = (float*)y->data;
    std::cout << "y: " << std::endl;
    print_matrix(y_data, y->ne[1], y->ne[0], 10, 10);
    // calculate the error metrics
    ErrorMetrics metrics = calculate_error_metrics(y_ref_fp32, y_data, y->ne[0] * y->ne[1]);
    print_error_metrics(metrics);

    ggml_free(local_ctx);
    ggml_backend_buffer_free(buffer);
    return true;
}
// Test matrix multiplication
// Test matrix multiplication
bool test_mul_mat_f16_f32(ggml_backend_t& backend) {
    GGML_LOG_INFO("\n=== Testing Matrix Multiplication F16 x F32 ===\n");
    
    // NOTE: Hexagon backend supports:
    // - Q4_0, Q8_0, MXFP4 types for src0 (weights)
    // - F16 for src0 (requires experimental flag)
    // - F32 for src1 (input) and dst (output)
    // F32 x F32 matmul is NOT supported, so we use F16 x F32
    
    const int m = 256;   // rows of result
    const int n = 512;   // cols of result  
    const int k = 1024;  // shared dimension
    
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,  // Use backend buffers
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        GGML_LOG_ERROR("Failed to create ggml context\n");
        return false;
    }
    
    // Create input matrices
    // For ggml_mul_mat(a, b): result[m,n] = a[k,m] @ b[n,k]
    // a is transposed in the multiplication
    // Use F16 for src0 (a) as Hexagon supports F16 x F32 but not F32 x F32
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);  // [k, m]
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);  // [k, n]
    
    // Apply matrix multiplication
    struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
    
    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    
    // Allocate buffers on the backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        GGML_LOG_ERROR("Failed to allocate backend buffer\n");
        ggml_free(ctx);
        return false;
    }
    
    // Initialize with test data
    ggml_fp16_t* a_data = (ggml_fp16_t*)a->data;
    float* b_data = (float*)b->data;
    srand(123); 
    for (int i = 0; i < k * m; i++) {
        a_data[i] = ggml_fp32_to_fp16(random_float(-4, 4));  // values between -1.0 and 1.0
    }
    for (int i = 0; i < k * n; i++) {
        b_data[i] = random_float(-4, 4);
    }
    
    GGML_LOG_INFO("Matrix A shape: [%d, %d]\n", (int)a->ne[1], (int)a->ne[0]);
    GGML_LOG_INFO("Matrix B shape: [%d, %d]\n", (int)b->ne[1], (int)b->ne[0]);
    
    GGML_LOG_INFO("Computing matrix multiplication on Hexagon backend...\n");
    
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("Failed to compute graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    
    GGML_LOG_INFO("Result shape: [%d, %d]\n", (int)result->ne[1], (int)result->ne[0]);
    print_tensor("Result (first 10)", result);
    
    // CPU reference implementation
    GGML_LOG_INFO("\nComputing CPU reference...\n");
    float* cpu_reference = (float*)malloc(m * n * sizeof(float));
    ggml_fp16_t* a_data_cpu = (ggml_fp16_t*)a->data;
    float* b_data_cpu = (float*)b->data;
    
    // Matrix multiplication: C[n, m] = B @ A^T; (n x k) @ (k x m) = (n x m), or X @ W^T, all row major
    // If everything is viewed as column-major:
    // C^T = A @ B^T; (m x k) @ (k x n) = (m x n)
    // For each output element C[i,j], compute dot product of A[:,i] with B[:,j]
    // A is accessed with [i * k + p], which means A is m x k in row-major
    // B is accessed with [j * k + p], which means B is n x k in row-major
    // C is stored as [j * m + i], which means C is n * m in row-major
    // k is D, m is Do, n is L
    // So: A: [Do, D], B: [L, D], C: [L, Do]
    // During DSP operation, A is put as src0, therefore, it is viewed as [k, m] in column-major
    //                       B is put as src1, therefore, it is viewed as [k, n] in column-major
    // Result is in dst, viewed as [m, n] in column-major
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                // A is stored column-major: A[k,m] means k rows, m columns
                // Element at row p, column i is at index: i*k + p
                ggml_fp16_t a_val_fp16 = a_data_cpu[i * k + p];
                float a_val = ggml_fp16_to_fp32(a_val_fp16);
                float b_val = b_data_cpu[j * k + p];
                sum += a_val * b_val;
            }
            // Result is stored column-major: result[m,n]
            // Element at row i, column j is at index: j*m + i
            cpu_reference[j * m + i] = sum;
        }
    }
    
    // Calculate error metrics
    float* result_data = (float*)result->data;
    ErrorMetrics metrics = calculate_error_metrics(cpu_reference, result_data, m * n);
    print_error_metrics(metrics);
    
    // Verify result: with all 1.0 inputs, result should be k (1024) in each element
    
    free(cpu_reference);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    GGML_LOG_INFO("Matrix multiplication test completed successfully!\n");
    return true;
}