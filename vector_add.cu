#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// CUDA error checking macro
#define cudaCheckError(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
            exit(code);
    }
}

// CUDA Kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }
}

// CPU version of vector addition
void vectorAddCPU(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int N = 10000 * 100; // 1,000,000 elements to see clear speedup
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);
    float *h_c_gpu = (float *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = sinf(i) * sinf(i);
        h_b[i] = cosf(i) * cosf(i);
    }

    // ----------- CPU version -----------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms\n";

    // ----------- GPU version -----------
    float *d_a, *d_b, *d_c;
    cudaCheckError(cudaMalloc(&d_a, size));
    cudaCheckError(cudaMalloc(&d_b, size));
    cudaCheckError(cudaMalloc(&d_c, size));

    cudaCheckError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int THREADS_PER_BLOCK = 256;
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAdd<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaCheckError(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_duration.count() << " ms\n";

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5)
        {
            std::cerr << "Mismatch at index " << i << ": " << h_c_cpu[i] << " != " << h_c_gpu[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success)
    {
        std::cout << "Vector addition completed successfully!\n";
        double speedup = cpu_duration.count() / gpu_duration.count();
        std::cout << "Speedup: " << speedup << "X" << std::endl;
    }

    // Free memory
    cudaCheckError(cudaFree(d_a));
    cudaCheckError(cudaFree(d_b));
    cudaCheckError(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}
