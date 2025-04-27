#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h> // For FLT_MAX
#include <math.h>
#include <chrono>

#define THREADS_PER_BLOCK 1024

__global__ void argmax_partial_kernel(const float *input, int n, float *max_vals_partial, int *max_idxs_partial)
{
    __shared__ float s_vals[THREADS_PER_BLOCK];
    __shared__ int s_idxs[THREADS_PER_BLOCK];

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (gidx < n)
    {
        s_vals[tid] = input[gidx];
        s_idxs[tid] = gidx;
    }
    else
    {
        s_vals[tid] = -FLT_MAX; // Very small number
        s_idxs[tid] = -1;       // Invalid index
    }

    __syncthreads();

    // Reduction to find the maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            if (s_vals[tid + stride] > s_vals[tid])
            {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the result for this block
    if (tid == 0)
    {
        max_vals_partial[blockIdx.x] = s_vals[0];
        max_idxs_partial[blockIdx.x] = s_idxs[0];
    }
}

void argmax_cpu(const float *input, int n, float *max_val, int *max_idx)
{
    *max_val = -FLT_MAX;
    *max_idx = -1;
    for (int i = 0; i < n; i++)
    {
        if (input[i] > *max_val)
        {
            *max_val = input[i];
            *max_idx = i;
        }
    }
}

int main()
{
    const int n = 1 << 20; // 1 million elements

    size_t size = n * sizeof(float);

    // Host allocations
    float *h_input = (float *)malloc(size);
    float *h_max_vals_partial;
    int *h_max_idxs_partial;
    float max_val_cpu;
    int max_idx_cpu;

    // Initialize input
    for (int i = 0; i < n; i++)
    {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU Argmax
    auto start_cpu = std::chrono::high_resolution_clock::now();
    argmax_cpu(h_input, n, &max_val_cpu, &max_idx_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Device allocations
    float *d_input;
    float *d_max_vals_partial;
    int *d_max_idxs_partial;

    cudaMalloc((void **)&d_input, size);

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc((void **)&d_max_vals_partial, num_blocks * sizeof(float));
    cudaMalloc((void **)&d_max_idxs_partial, num_blocks * sizeof(int));

    h_max_vals_partial = (float *)malloc(num_blocks * sizeof(float));
    h_max_idxs_partial = (int *)malloc(num_blocks * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // GPU Argmax
    cudaDeviceSynchronize();
    auto start_gpu = std::chrono::high_resolution_clock::now();
    argmax_partial_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, n, d_max_vals_partial, d_max_idxs_partial);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Copy partial results back
    cudaMemcpy(h_max_vals_partial, d_max_vals_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_idxs_partial, d_max_idxs_partial, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    float final_max = -FLT_MAX;
    int final_idx = -1;
    for (int i = 0; i < num_blocks; i++)
    {
        if (h_max_vals_partial[i] > final_max)
        {
            final_max = h_max_vals_partial[i];
            final_idx = h_max_idxs_partial[i];
        }
    }

    // Verification
    if (fabs(final_max - max_val_cpu) < 1e-3 && final_idx == max_idx_cpu)
    {
        printf("Results verified: CPU and GPU match.\n");
    }
    else
    {
        printf("Mismatch!\n");
        printf("CPU: max = %f at idx = %d\n", max_val_cpu, max_idx_cpu);
        printf("GPU: max = %f at idx = %d\n", final_max, final_idx);
    }

    printf("CPU time: %.5f seconds\n", cpu_time.count());
    printf("GPU time: %.5f seconds\n", gpu_time.count());
    printf("Speedup: %.2fX\n", cpu_time.count() / gpu_time.count());

    // Free memory
    free(h_input);
    free(h_max_vals_partial);
    free(h_max_idxs_partial);
    cudaFree(d_input);
    cudaFree(d_max_vals_partial);
    cudaFree(d_max_idxs_partial);

    return 0;
}
