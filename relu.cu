#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define N 10000000

// CUDA Kernel
__global__ void relu_kernel(const float *input, float *output, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        output[index] = fmaxf(0.0f, input[index]);
    }
}

// CPU version of the ReLU function (sequential)
void relu_cpu(const float *input, float *output, int n)
{
    for (int i = 0; i < n; i++)
    {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

int main()
{
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output_cpu = (float *)malloc(size);
    float *h_output_gpu = (float *)malloc(size);

    // Fill in the array with random values between -10 and +10
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++)
    {
        h_input[i] = ((float)(rand() % 2000) - 1000.0f) / 100.0f;
    }

    // --- Sequential (CPU) execution ---
    clock_t start_cpu = clock(); // Start timing for CPU execution
    relu_cpu(h_input, h_output_cpu, N);
    clock_t end_cpu = clock(); // End timing for CPU execution
    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    // --- Parallel (GPU) execution ---
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Configure kernel launch
    int THREADS_PER_BLOCK = 1024;
    int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    clock_t start_gpu = clock(); // Start timing for GPU execution
    relu_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaDeviceSynchronize();   // Ensure kernel finishes before measuring end time
    clock_t end_gpu = clock(); // End timing for GPU execution
    double gpu_time = double(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    // Calculate speedup
    double speedup = cpu_time / gpu_time;

    // Print results
    std::cout << "CPU time: " << cpu_time << " seconds\n";
    std::cout << "GPU time: " << gpu_time << " seconds\n";
    std::cout << "Speedup: " << speedup << "X\n";

    return 0;
}
