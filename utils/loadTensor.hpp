#pragma once


#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#ifndef GGML_LOG_INFO
#define GGML_LOG_INFO(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef GGML_LOG_ERROR
#define GGML_LOG_ERROR(...) fprintf(stderr, "ERROR: " __VA_ARGS__)
#endif

/**
 * GGUFTensorManager - A RAII class to manage GGUF tensors with automatic cleanup
 * 
 * This class loads all tensors from a GGUF file and manages their lifecycle.
 * All resources (buffer, gguf_ctx, ctx_gguf) are automatically freed in the destructor.
 */
class GGUFTensorManager {
public:
    /**
     * Constructor - Loads all tensors from the GGUF file
     * @param filename Path to the GGUF file
     * @param backend The backend to allocate tensors on
     * @param additional_memory_bytes Additional memory to allocate in the context for creating more tensors (default: 0)
     *                                 This is in addition to the memory needed for the GGUF tensors themselves.
     *                                 Note: This is for tensor metadata overhead, not tensor data storage.
     */
    GGUFTensorManager(const std::string &filename, ggml_backend_t backend, size_t additional_memory_bytes = 0);
    
    /**
     * Destructor - Automatically cleans up all resources
     */
    ~GGUFTensorManager();
    
    // Delete copy constructor and assignment operator to prevent copying
    GGUFTensorManager(const GGUFTensorManager&) = delete;
    GGUFTensorManager& operator=(const GGUFTensorManager&) = delete;
    
    /**
     * Get a specific tensor by name
     * @param tensor_name Name of the tensor to retrieve
     * @return Pointer to the tensor, or nullptr if not found
     */
    struct ggml_tensor* get_tensor(const std::string &tensor_name) const;
    
    /**
     * Get the GGML context (useful for creating new tensors in the same context)
     * @return Pointer to the GGML context
     */
    struct ggml_context* get_context() const { return ctx_gguf; }
    
    /**
     * Get the GGUF context
     * @return Pointer to the GGUF context
     */
    struct gguf_context* get_gguf_context() const { return gguf_ctx; }
    
    /**
     * Check if the manager was successfully initialized
     * @return true if initialization was successful, false otherwise
     */
    bool is_valid() const { return valid; }
    
    /**
     * Get the number of tensors loaded
     * @return Number of tensors
     */
    size_t get_tensor_count() const { return tensor_map.size(); }
    
    /**
     * Get all tensor names
     * @return Vector of tensor names
     */
    std::vector<std::string> get_tensor_names() const;


private:
    std::string filename;
    ggml_backend_t backend;
    size_t additional_memory_bytes;
    struct gguf_context* gguf_ctx;
    struct ggml_context* ctx_gguf;
    ggml_backend_buffer_t buffer;
    std::unordered_map<std::string, struct ggml_tensor*> tensor_map;
    bool valid;
    
    bool load_tensors();
};