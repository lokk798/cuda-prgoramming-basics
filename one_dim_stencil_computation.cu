#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define BLOCK_SIZE 1024
#define RADIUS 3

__global__ void stencil_1d(const float *in, float *out, int n)
{
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int lidx = threadIdx.x + RADIUS;

    // Load center elements
    if (gidx < n)
        temp[lidx] = in[gidx];
    else
        temp[lidx] = 0.0f;

    // Load left halo
    if (threadIdx.x < RADIUS)
    {
        int left_idx = gidx - RADIUS;
        if (left_idx >= 0)
            temp[lidx - RADIUS] = in[left_idx];
        else
            temp[lidx - RADIUS] = 0.0f;
    }

    // Load right halo
    if (threadIdx.x >= blockDim.x - RADIUS)
    {
        int right_idx = gidx + RADIUS;
        if (right_idx < n)
            temp[lidx + RADIUS] = in[right_idx];
        else
            temp[lidx + RADIUS] = 0.0f;
    }

    __syncthreads();

    if (gidx < n)
    {
        float result = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; ++offset)
        {
            result += temp[lidx + offset];
        }
        out[gidx] = result;
    }
}

void stencilCPU(const float *in, float *out, int n)
{
    for (int i = 0; i < n; ++i)
    {
        float sum = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; ++offset)
        {
            int idx = i + offset;
            if (idx >= 0 && idx < n)
                sum += in[idx];
        }
        out[i] = sum;
    }
}

int main()
{
    int n = 1 << 24; // ~16 million elements
    size_t size = n * sizeof(float);

    float *h_in = (float *)malloc(size);
    float *h_out_cpu = (float *)malloc(size);
    float *h_out_gpu = (float *)malloc(size);

    for (int i = 0; i < n; ++i)
    {
        h_in[i] = 1.0f;
    }

    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    stencilCPU(h_in, h_out_cpu, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();

    // GPU timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    stencil_1d<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++)
    {
        if (abs(h_out_cpu[i] - h_out_gpu[i]) > 1e-5)
        {
            correct = false;
            printf("Mismatch at index %d: CPU=%f GPU=%f\n", i, h_out_cpu[i], h_out_gpu[i]);
            break;
        }
    }

    printf("Stencil Computation %s\n", correct ? "PASSED" : "FAILED");
    printf("CPU Time = %f seconds\n", cpu_time);
    printf("GPU Time = %f seconds\n", gpu_time);
    printf("Speedup = %.2fx\n", cpu_time / gpu_time);

    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
