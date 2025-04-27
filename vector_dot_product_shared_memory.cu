#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <chrono>

#define THREADS_PER_BLOCK 1024

// CUDA kernel for partial dot product using shared memory
__global__ void vectorDotProduct_partial(const float *a, const float *b, float *partial_c, int n)
{
    __shared__ float cache[THREADS_PER_BLOCK];

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes its own contribution
    float temp = 0.0f;
    if (gidx < n)
    {
        temp = a[gidx] * b[gidx];
    }
    cache[tid] = temp;

    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Write block's result to partial_c
    if (tid == 0)
    {
        partial_c[blockIdx.x] = cache[0];
    }
}

// CPU version for comparison
float vectorDotProduct_cpu(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

int main()
{
    // Set vector size
    int n = 1 << 24; // ~16 million elements

    size_t vector_size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(vector_size);
    float *h_b = (float *)malloc(vector_size);

    // Initialize input vectors with random floats
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_a;
    float *d_b;
    float *d_partial_c;
    cudaMalloc((void **)&d_a, vector_size);
    cudaMalloc((void **)&d_b, vector_size);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc((void **)&d_partial_c, blocks * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vector_size, cudaMemcpyHostToDevice);

    // ----------------- CPU Timing -----------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float result_cpu = vectorDotProduct_cpu(h_a, h_b, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // ----------------- GPU Timing -----------------
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    vectorDotProduct_partial<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial_c, n);
    cudaEventRecord(stop_gpu);

    // Copy partial results back
    float *h_partial_c = (float *)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial_c, d_partial_c, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    // Final summation on host
    float result_gpu = 0.0f;
    for (int i = 0; i < blocks; ++i)
    {
        result_gpu += h_partial_c[i];
    }

    // ----------------- Verification -----------------
    if (fabs(result_cpu - result_gpu) < 1e-2)
    {
        std::cout << "Results match!" << std::endl;
    }
    else
    {
        std::cout << "Results do NOT match!" << std::endl;
        std::cout << "CPU: " << result_cpu << " GPU: " << result_gpu << std::endl;
    }

    // ----------------- Speedup -----------------
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "X" << std::endl;

    // Free memory
    free(h_a);
    free(h_b);
    free(h_partial_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_c);

    return 0;
}
