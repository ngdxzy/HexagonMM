#include "loadTensor.hpp"

// GGUFTensorManager implementation
GGUFTensorManager::GGUFTensorManager(const std::string &filename, ggml_backend_t backend, size_t additional_memory_bytes)
    : filename(filename), backend(backend), additional_memory_bytes(additional_memory_bytes), 
      gguf_ctx(nullptr), ctx_gguf(nullptr), buffer(nullptr), valid(false) {
    valid = load_tensors();
}

GGUFTensorManager::~GGUFTensorManager() {
    // Automatic cleanup - free all resources
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (gguf_ctx) {
        gguf_free(gguf_ctx);
        gguf_ctx = nullptr;
    }
    if (ctx_gguf) {
        ggml_free(ctx_gguf);
        ctx_gguf = nullptr;
    }
}

bool GGUFTensorManager::load_tensors() {
    GGML_LOG_INFO("\n=== Loading GGUF File with Data: %s ===\n", filename.c_str());

    // Step 1: First, load GGUF metadata to get tensor count
    struct ggml_context* temp_ctx = nullptr;
    struct gguf_init_params temp_params = {
        .no_alloc = true,
        .ctx = &temp_ctx
    };
    struct gguf_context* temp_gguf_ctx = gguf_init_from_file(filename.c_str(), temp_params);
    
    if (!temp_gguf_ctx) {
        GGML_LOG_ERROR("Failed to load GGUF file: %s\n", filename.c_str());
        return false;
    }
    
    // Get tensor count to calculate required memory
    int n_tensors = gguf_get_n_tensors(temp_gguf_ctx);
    size_t base_memory = n_tensors * ggml_tensor_overhead();
    size_t total_memory = base_memory + additional_memory_bytes;
    
    GGML_LOG_INFO("GGUF file has %d tensors, allocating %zu bytes for context (base: %zu, additional: %zu)\n", 
                  n_tensors, total_memory, base_memory, additional_memory_bytes);
    
    // Free temporary context
    ggml_free(temp_ctx);
    gguf_free(temp_gguf_ctx);
    temp_ctx = nullptr;
    temp_gguf_ctx = nullptr;
    
    // Step 2: Create a context with the calculated size (including additional memory)
    struct ggml_init_params ctx_params = {
        .mem_size = total_memory,
        .mem_buffer = nullptr,
        .no_alloc = true
    };
    ctx_gguf = ggml_init(ctx_params);
    
    if (!ctx_gguf) {
        GGML_LOG_ERROR("Failed to create GGML context with %zu bytes\n", total_memory);
        return false;
    }
    
    // Step 3: Load GGUF file to get tensor information (dimensions, types, etc.)
    // We need to load it once to get tensor metadata, then we'll recreate tensors in our larger context
    struct ggml_context* temp_ctx_for_info = nullptr;
    struct gguf_init_params info_params = {
        .no_alloc = true,
        .ctx = &temp_ctx_for_info
    };
    struct gguf_context* info_gguf_ctx = gguf_init_from_file(filename.c_str(), info_params);
    
    if (!info_gguf_ctx) {
        GGML_LOG_ERROR("Failed to load GGUF file for tensor info: %s\n", filename.c_str());
        ggml_free(ctx_gguf);
        ctx_gguf = nullptr;
        return false;
    }
    
    // Now we have tensor info in info_gguf_ctx and temp_ctx_for_info
    // We'll use this to manually create tensors in our larger ctx_gguf context
    gguf_ctx = info_gguf_ctx;
    
    // Manually create all tensors in our larger context
    // We can get tensor info from the temporary context that was created
    for (int i = 0; i < n_tensors; i++) {
        const char* tensor_name = gguf_get_tensor_name(gguf_ctx, i);
        enum ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, i);
        
        // Get tensor from temp context to read its dimensions
        struct ggml_tensor* temp_tensor = ggml_get_tensor(temp_ctx_for_info, tensor_name);
        if (!temp_tensor) {
            GGML_LOG_ERROR("Failed to get tensor info for %s\n", tensor_name);
            continue;
        }
        
        // Create the tensor in our larger context with the same dimensions
        struct ggml_tensor* tensor = ggml_new_tensor(ctx_gguf, tensor_type, GGML_MAX_DIMS, temp_tensor->ne);
        if (!tensor) {
            GGML_LOG_ERROR("Failed to create tensor %s in context\n", tensor_name);
            continue;
        }
        
        // Set the tensor name
        ggml_set_name(tensor, tensor_name);
        
        // Add to map
        tensor_map[std::string(tensor_name)] = tensor;
    }
    
    // Free the temporary context (we've copied all tensor metadata to our context)
    ggml_free(temp_ctx_for_info);
    temp_ctx_for_info = nullptr;
    
    GGML_LOG_INFO("Created %zu tensors in context with %zu bytes (base: %zu, additional: %zu)\n",
                  tensor_map.size(), total_memory, base_memory, additional_memory_bytes);
    
    // Step 2: Allocate tensors on backend (Hexagon)
    buffer = ggml_backend_alloc_ctx_tensors(ctx_gguf, backend);
    if (!buffer) {
        GGML_LOG_ERROR("Failed to allocate backend buffer for tensors\n");
        gguf_free(gguf_ctx);
        ggml_free(ctx_gguf);
        gguf_ctx = nullptr;
        ctx_gguf = nullptr;
        return false;
    }
    
    // Step 3: Load tensor data from file
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        GGML_LOG_ERROR("Failed to open file for reading tensor data\n");
        ggml_backend_buffer_free(buffer);
        gguf_free(gguf_ctx);
        ggml_free(ctx_gguf);
        buffer = nullptr;
        gguf_ctx = nullptr;
        ctx_gguf = nullptr;
        return false;
    }
    
    // Get data offset (where tensor data starts in the file)
    size_t data_offset = gguf_get_data_offset(gguf_ctx);
    fseek(file, data_offset, SEEK_SET);
    
    // Read each tensor's data - use the order from GGUF file
    GGML_LOG_INFO("Loading %d tensors...\n", n_tensors);
    
    for (int i = 0; i < n_tensors; i++) {
        const char* tensor_name = gguf_get_tensor_name(gguf_ctx, i);
        
        // Get tensor from our map (we created it earlier)
        auto it = tensor_map.find(std::string(tensor_name));
        if (it != tensor_map.end()) {
            struct ggml_tensor* tensor = it->second;
            size_t tensor_size = ggml_nbytes(tensor);
            
            // Read data into temporary CPU buffer
            void* temp_data = malloc(tensor_size);
            size_t read_size = fread(temp_data, 1, tensor_size, file);
            
            if (read_size != tensor_size) {
                GGML_LOG_ERROR("Failed to read tensor data for %s\n", tensor_name);
                free(temp_data);
                continue;
            }
            
            // Copy data to backend buffer
            ggml_backend_tensor_set(tensor, temp_data, 0, tensor_size);
            
            free(temp_data);
        } else {
            GGML_LOG_ERROR("Tensor %s not found in tensor map\n", tensor_name);
        }
    }
    
    fclose(file);
    
    GGML_LOG_INFO("\n=== All %d tensors loaded successfully ===\n", n_tensors);
    GGML_LOG_INFO("Tensor map size: %zu\n", tensor_map.size());
    
    return true;
}

struct ggml_tensor* GGUFTensorManager::get_tensor(const std::string &tensor_name) const {
    auto it = tensor_map.find(tensor_name);
    if (it != tensor_map.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::string> GGUFTensorManager::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_map.size());
    for (const auto& pair : tensor_map) {
        names.push_back(pair.first);
    }
    return names;
}
