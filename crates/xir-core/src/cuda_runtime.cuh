// Minimal synchronous host transport. Each parameter gets a distinct allocation.
static int xir_error(char* error, size_t capacity, const char* message) {
    if (error && capacity) std::snprintf(error, capacity, "%s", message);
    return 1;
}

#define XIR_CUDA(expression) do { \
    const cudaError_t status = (expression); \
    if (status != cudaSuccess) { \
        if (error && error_capacity) std::snprintf(error, error_capacity, "%s: %s", #expression, cudaGetErrorString(status)); \
        return 2; \
    } \
} while (false)

struct XirDeviceBuffer {
    void* pointer = nullptr;
    ~XirDeviceBuffer() { if (pointer) cudaFree(pointer); }
    XirDeviceBuffer() = default;
    XirDeviceBuffer(const XirDeviceBuffer&) = delete;
    XirDeviceBuffer& operator=(const XirDeviceBuffer&) = delete;
};
