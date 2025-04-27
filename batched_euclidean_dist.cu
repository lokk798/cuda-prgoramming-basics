#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define THREADS_PER_BLOCK 1024

__global__ void euclideanDist_kernel(const float *batchA, const float *batchB, float *distances, int num_vectors, int dim)
{
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx < num_vectors)
    {
        float sum_sq = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            int index = vec_idx * dim + j;
            float diff = batchA[index] - batchB[index];
            sum_sq += diff * diff;
        }
        distances[vec_idx] = sqrtf(sum_sq);
    }
}

void euclideanDist_cpu(const float *batchA, const float *batchB, float *distances, int num_vectors, int dim)
{
    for (int i = 0; i < num_vectors; i++)
    {
        float sum_sq = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            int index = i * dim + j;
            float diff = batchA[index] - batchB[index];
            sum_sq += diff * diff;
        }
        distances[i] = sqrtf(sum_sq);
    }
}

int main()
{
    const int num_vectors = 1 << 20; // 1 million vectors
    const int dim = 128;             // 128-dimensional vectors

    size_t size = num_vectors * dim * sizeof(float);
    size_t dist_size = num_vectors * sizeof(float);

    // Allocate host memory
    float *h_batchA = (float *)malloc(size);
    float *h_batchB = (float *)malloc(size);
    float *h_distances_cpu = (float *)malloc(dist_size);
    float *h_distances_gpu = (float *)malloc(dist_size);

    // Initialize host data
    for (int i = 0; i < num_vectors * dim; i++)
    {
        h_batchA[i] = static_cast<float>(rand()) / RAND_MAX;
        h_batchB[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU computation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    euclideanDist_cpu(h_batchA, h_batchB, h_distances_cpu, num_vectors, dim);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Allocate device memory
    float *d_batchA, *d_batchB, *d_distances;
    cudaMalloc((void **)&d_batchA, size);
    cudaMalloc((void **)&d_batchB, size);
    cudaMalloc((void **)&d_distances, dist_size);

    // Copy data to device
    cudaMemcpy(d_batchA, h_batchA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_batchB, h_batchB, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int num_blocks = (num_vectors + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaDeviceSynchronize();
    auto start_gpu = std::chrono::high_resolution_clock::now();
    euclideanDist_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_batchA, d_batchB, d_distances, num_vectors, dim);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Copy result back
    cudaMemcpy(h_distances_gpu, d_distances, dist_size, cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < num_vectors; i++)
    {
        if (fabs(h_distances_cpu[i] - h_distances_gpu[i]) > 1e-3)
        {
            printf("Mismatch at %d: CPU %f, GPU %f\n", i, h_distances_cpu[i], h_distances_gpu[i]);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("Results verified: CPU and GPU match.\n");
    }

    printf("CPU time: %.5f seconds\n", cpu_time.count());
    printf("GPU time: %.5f seconds\n", gpu_time.count());
    printf("Speedup: %.2fX\n", cpu_time.count() / gpu_time.count());

    // Free memory
    free(h_batchA);
    free(h_batchB);
    free(h_distances_cpu);
    free(h_distances_gpu);
    cudaFree(d_batchA);
    cudaFree(d_batchB);
    cudaFree(d_distances);

    return 0;
}
