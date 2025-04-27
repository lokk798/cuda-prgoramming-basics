#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <chrono>

#define THREADS_PER_BLOCK 1024

// CUDA Kernel for Matrix-Vector Multiplication
__global__ void matVecMul_kernel(const float *M, const float *x, float *y, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col)
        {
            sum += M[row * cols + col] * x[col];
        }
        y[row] = sum;
    }
}

// CPU version for verification and timing
void matVecMul_cpu(const float *M, const float *x, float *y, int rows, int cols)
{
    for (int row = 0; row < rows; ++row)
    {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col)
        {
            sum += M[row * cols + col] * x[col];
        }
        y[row] = sum;
    }
}

int main()
{
    // Set matrix size
    int rows = 4096; // Example size, you can change
    int cols = 4096;

    size_t matrix_size = rows * cols * sizeof(float);
    size_t vector_size = cols * sizeof(float);
    size_t output_size = rows * sizeof(float);

    // Allocate host memory
    float *h_M = (float *)malloc(matrix_size);
    float *h_x = (float *)malloc(vector_size);
    float *h_y_cpu = (float *)malloc(output_size);
    float *h_y_gpu = (float *)malloc(output_size);

    // Initialize matrix and vector
    for (int i = 0; i < rows * cols; ++i)
    {
        h_M[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < cols; ++i)
    {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_M;
    float *d_x;
    float *d_y;
    cudaMalloc((void **)&d_M, matrix_size);
    cudaMalloc((void **)&d_x, vector_size);
    cudaMalloc((void **)&d_y, output_size);

    // Copy inputs to device
    cudaMemcpy(d_M, h_M, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, vector_size, cudaMemcpyHostToDevice);

    // ----------------- CPU Timing -----------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matVecMul_cpu(h_M, h_x, h_y_cpu, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // ----------------- GPU Timing -----------------
    int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    matVecMul_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_M, d_x, d_y, rows, cols);
    cudaEventRecord(stop_gpu);

    // Copy result back
    cudaMemcpy(h_y_gpu, d_y, output_size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    // ----------------- Verification -----------------
    bool match = true;
    for (int i = 0; i < rows; ++i)
    {
        if (fabs(h_y_cpu[i] - h_y_gpu[i]) > 1e-4)
        {
            match = false;
            std::cout << "Mismatch at index " << i << ": CPU = " << h_y_cpu[i]
                      << ", GPU = " << h_y_gpu[i] << std::endl;
            break;
        }
    }

    if (match)
    {
        std::cout << "Results match!" << std::endl;
    }
    else
    {
        std::cout << "Results do NOT match!" << std::endl;
    }

    // ----------------- Speedup -----------------
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "X" << std::endl;

    // Free memory
    free(h_M);
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);
    cudaFree(d_M);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
