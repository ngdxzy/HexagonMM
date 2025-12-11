#include "test_utils.hpp"


bool initialize_backend(ggml_backend_t& backend) {
    GGML_LOG_INFO("========================================\n");
    GGML_LOG_INFO("GGML Hexagon Backend Test Suite\n");
    GGML_LOG_INFO("========================================\n\n");
    
    // Initialize Hexagon backend using the registry API
    GGML_LOG_INFO("Initializing Hexagon backend...\n");
    
    ggml_backend_reg_t reg = ggml_backend_hexagon_reg();
    if (!reg) {
        GGML_LOG_ERROR("Failed to get Hexagon backend registry\n");
        return false;
    }
    
    // Get device count
    size_t device_count = ggml_backend_reg_dev_count(reg);
    GGML_LOG_INFO("Found %zu Hexagon device(s)\n", device_count);
    
    if (device_count == 0) {
        GGML_LOG_ERROR("No Hexagon devices found\n");
        return false;
    }
    
    // Get the first device
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (!dev) {
        GGML_LOG_ERROR("Failed to get Hexagon device\n");
        return false;
    }
    
    // Initialize backend from device
    backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        GGML_LOG_ERROR("Failed to initialize Hexagon backend from device\n");
        return false;
    }
    
    if (!ggml_backend_is_hexagon(backend)) {
        GGML_LOG_ERROR("Backend is not a Hexagon backend\n");
        ggml_backend_free(backend);
        return false;
    }
    
    GGML_LOG_INFO("Hexagon backend initialized successfully!\n");
    GGML_LOG_INFO("Backend name: %s\n", ggml_backend_name(backend));

    return true;
}

// Helper function to print tensor values
void print_tensor(const char* name, struct ggml_tensor* tensor, int max_elements) {
    if (tensor->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("Tensor %s is not a float tensor\n", name);
        return;
    }
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
