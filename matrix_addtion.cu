#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

#define THREADS_X 32
#define THREADS_Y 32

__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int width = 1 << 12;  // 4096
    const int height = 1 << 12; // 4096
    size_t size = width * height * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < width * height; i++)
    {
        h_A[i] = (float)(rand()) / RAND_MAX;
        h_B[i] = (float)(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS_X, THREADS_Y);
    dim3 numBlocks((width + THREADS_X - 1) / THREADS_X, (height + THREADS_Y - 1) / THREADS_Y);

    // CUDA timing
    auto start_cuda = std::chrono::high_resolution_clock::now();

    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width, height);
    cudaDeviceSynchronize();

    auto end_cuda = std::chrono::high_resolution_clock::now();
    double cuda_time = std::chrono::duration<double>(end_cuda - start_cuda).count();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();

    float *h_C_CPU = (float *)malloc(size);
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int idx = row * width + col;
            h_C_CPU[idx] = h_A[idx] + h_B[idx];
        }
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // Verification
    bool correct = true;
    for (int i = 0; i < width * height; i++)
    {
        if (abs(h_C[i] - h_C_CPU[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    if (correct)
    {
        printf("Matrix addition is correct!\n");
    }
    else
    {
        printf("Matrix addition is WRONG!\n");
    }

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << cuda_time << " ms" << std::endl;

    printf("Speedup: %.2fx\n", cpu_time / cuda_time);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
