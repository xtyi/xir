// Basic CUDA pointer examples corresponding to XIR pointer/address-space types.

#include <cuda_runtime.h>

// A and B are ordinary C pointers passed through the kernel ABI. In CUDA they
// conventionally point into global memory allocated by cudaMalloc.
__global__ void add_global(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float* a_ptr = A + i;  // pointer arithmetic in global memory
        float* c_ptr = C + i;
        *c_ptr = *a_ptr + B[i];
    }
}

// shared_buf is a pointer into the block's shared memory. The storage is
// declared with __shared__; the pointer itself remains ordinary C syntax.
__global__ void add_shared(const float* input, float* output, int n) {
    __shared__ float shared_buf[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    shared_buf[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    if (i < n) {
        float* out_ptr = output + i;
        *out_ptr = shared_buf[tid] + 1.0f;
    }
}

// A pointer can also be stored in a local variable or passed to a device
// function. The compiler decides whether private scalar state stays in a
// register; spilling is a backend implementation detail.
__device__ float load_and_scale(const float* ptr, float scale) {
    return (*ptr) * scale;
}

__global__ void scale_global(const float* input, float* output, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = load_and_scale(input + i, scale);
    }
}

