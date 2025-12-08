#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#define GGML_LOG_INFO(...) fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_ERROR(...) fprintf(stderr, "ERROR: " __VA_ARGS__)

// Helper function to print tensor values
void print_tensor(const char* name, struct ggml_tensor* tensor, int max_elements = 10) {
    GGML_LOG_INFO("%s: [", name);
    float* data = (float*)tensor->data;
    int n_elements = ggml_nelements(tensor);
    int print_count = n_elements < max_elements ? n_elements : max_elements;
    
    for (int i = 0; i < print_count; i++) {
        GGML_LOG_INFO("%.4f", data[i]);
        if (i < print_count - 1) GGML_LOG_INFO(", ");
    }
    if (n_elements > max_elements) {
        GGML_LOG_INFO(" ... (%d more)", n_elements - max_elements);
    }
    GGML_LOG_INFO("]\n");
}

// Error metric calculation
struct ErrorMetrics {
    float l1_relative_error;
    float l2_relative_error;
    float cosine_similarity;
    float rms_error;
};


ErrorMetrics calculate_error_metrics(const float* reference, const float* computed, int n_elements) {
    ErrorMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f};
    
    double l1_error = 0.0;
    double l2_error = 0.0;
    double l1_norm_ref = 0.0;
    double l2_norm_ref = 0.0;
    double dot_product = 0.0;
    double norm_ref = 0.0;
    double norm_computed = 0.0;
    double squared_error = 0.0;
    
    for (int i = 0; i < n_elements; i++) {
        double diff = fabs(computed[i] - reference[i]);
        double ref_abs = fabs(reference[i]);
        
        l1_error += diff;
        l2_error += diff * diff;
        l1_norm_ref += ref_abs;
        l2_norm_ref += reference[i] * reference[i];
        
        dot_product += reference[i] * computed[i];
        norm_ref += reference[i] * reference[i];
        norm_computed += computed[i] * computed[i];
        
        squared_error += diff * diff;
    }
    
    // L1 relative error
    metrics.l1_relative_error = (l1_norm_ref > 1e-10) ? (l1_error / l1_norm_ref) : 0.0f;
    
    // L2 relative error
    metrics.l2_relative_error = (l2_norm_ref > 1e-10) ? sqrt(l2_error / l2_norm_ref) : 0.0f;
    
    // Cosine similarity
    double denom = sqrt(norm_ref) * sqrt(norm_computed);
    metrics.cosine_similarity = (denom > 1e-10) ? (dot_product / denom) : 0.0f;
    
    // RMS error
    metrics.rms_error = sqrt(squared_error / n_elements);
    
    return metrics;
}

void print_error_metrics(const ErrorMetrics& metrics) {
    GGML_LOG_INFO("\n--- Error Metrics ---\n");
    GGML_LOG_INFO("L1 Relative Error:    %.6e\n", metrics.l1_relative_error);
    GGML_LOG_INFO("L2 Relative Error:    %.6e\n", metrics.l2_relative_error);
    GGML_LOG_INFO("Cosine Similarity:    %.8f\n", metrics.cosine_similarity);
    GGML_LOG_INFO("RMS Error:            %.6e\n", metrics.rms_error);
    GGML_LOG_INFO("--------------------\n");
}
// Test matrix multiplication
bool test_mul_mat_f16_f32(ggml_backend_t backend) {
    GGML_LOG_INFO("\n=== Testing Matrix Multiplication F16 x F32 ===\n");
    
    // NOTE: Hexagon backend supports:
    // - Q4_0, Q8_0, MXFP4 types for src0 (weights)
    // - F16 for src0 (requires experimental flag)
    // - F32 for src1 (input) and dst (output)
    // F32 x F32 matmul is NOT supported, so we use F16 x F32
    
    const int m = 256;   // rows of result
    const int n = 512;   // cols of result  
    const int k = 256;  // shared dimension
    
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
    for (int i = 0; i < k * m; i++) {
        a_data[i] = ggml_fp32_to_fp16(2 * ((float)rand()/RAND_MAX - 0.5f));  // values between -1.0 and 1.0
    }
    for (int i = 0; i < k * n; i++) {
        b_data[i] = (float)(2 * ((float)rand()/RAND_MAX - 0.5));
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
    // k is D, m is DQ, n is L
    // So: A: [DQ, D], B: [L, D], C: [L, DQ]
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


int main(int argc, char** argv) {
    GGML_LOG_INFO("========================================\n");
    GGML_LOG_INFO("GGML Hexagon Backend Test Suite\n");
    GGML_LOG_INFO("========================================\n\n");
    
    // Initialize Hexagon backend using the registry API
    GGML_LOG_INFO("Initializing Hexagon backend...\n");
    
    ggml_backend_reg_t reg = ggml_backend_hexagon_reg();
    if (!reg) {
        GGML_LOG_ERROR("Failed to get Hexagon backend registry\n");
        return 1;
    }
    
    // Get device count
    size_t device_count = ggml_backend_reg_dev_count(reg);
    GGML_LOG_INFO("Found %zu Hexagon device(s)\n", device_count);
    
    if (device_count == 0) {
        GGML_LOG_ERROR("No Hexagon devices found\n");
        return 1;
    }
    
    // Get the first device
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (!dev) {
        GGML_LOG_ERROR("Failed to get Hexagon device\n");
        return 1;
    }
    
    // Initialize backend from device
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        GGML_LOG_ERROR("Failed to initialize Hexagon backend from device\n");
        return 1;
    }
    
    if (!ggml_backend_is_hexagon(backend)) {
        GGML_LOG_ERROR("Backend is not a Hexagon backend\n");
        ggml_backend_free(backend);
        return 1;
    }
    
    GGML_LOG_INFO("Hexagon backend initialized successfully!\n");
    GGML_LOG_INFO("Backend name: %s\n", ggml_backend_name(backend));
    
    // Run tests
    bool all_passed = true;
    
  
    if (!test_mul_mat_f16_f32(backend)) {
        GGML_LOG_ERROR("Matrix multiplication test F16 x F32 FAILED\n");
        return 1;
    }
    
    return 0;
}
