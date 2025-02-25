#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cuda_fp16.h>
#include "utils.h"
using namespace std;

template <typename T, int vec_size>
__global__ void threads_unroll_add_kernel(const T *in_dense, const T *in_bc, T *out, const size_t n,
                                          const size_t stride0, const size_t stride1) {
    const int block_work_size = blockDim.x * vec_size;
    auto index = blockIdx.x * block_work_size + threadIdx.x;
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
        if (index < n) {
            out[index] = in_dense[index] + in_bc[index / stride0 * stride1 + index % stride1];
        }
        index += blockDim.x;
    }
}

template <typename T, int vec_size>
float threads_unroll_add(const T *in_dense, const T *in_bc, T *out, const size_t n,
                         const size_t stride0, const size_t stride1) {
    const int block_size = 1024;
    const int block_work_size = block_size * vec_size;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    threads_unroll_add_kernel<T, vec_size><<<numBlocks, threadsPerBlock>>>(
        in_dense, in_bc, out, n, stride0, stride1);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

template <typename T, int vec_size>
void test_threads_unroll_add(size_t left, size_t mid, size_t right) {
    auto problem_size = left * mid * right;
    auto bc_size = left * 1 * right;

    auto in_dense_cpu = new T[problem_size];
    auto in_bc_cpu = new T[bc_size];
    auto out_cpu = new T[problem_size];

    for (int i = 0; i < problem_size; i++)
        in_dense_cpu[i] = (T)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    for (int i = 0; i < bc_size; i++)
        in_bc_cpu[i] = (T)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

    T *in_dense_cuda, *in_bc_cuda, *out_cuda;
    cudaMalloc(&in_dense_cuda, problem_size * sizeof(T));
    cudaMalloc(&in_bc_cuda, bc_size * sizeof(T));
    cudaMalloc(&out_cuda, problem_size * sizeof(T));

    cudaMemcpy(in_dense_cuda, in_dense_cpu, problem_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(in_bc_cuda, in_bc_cpu, bc_size * sizeof(T), cudaMemcpyHostToDevice);

    float timems;
    for (int i = 0; i < 300; i++)
        timems = threads_unroll_add<T, vec_size>(in_dense_cuda, in_bc_cuda, out_cuda, problem_size, mid * right, right);

    float total_GBytes = (2 * problem_size + bc_size) * sizeof(T) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    cudaMemcpy(out_cpu, out_cuda, problem_size * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < problem_size; i++) {
        auto bc_idx = i / (mid * right) * right + i % right;
        auto diff = (float)(out_cpu[i] - in_dense_cpu[i] - in_bc_cpu[bc_idx]);
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    cudaFree(in_dense_cuda);
    cudaFree(in_bc_cuda);
    cudaFree(out_cuda);

    delete[] in_dense_cpu;
    delete[] in_bc_cpu;
    delete[] out_cpu;
}

int main() {
    std::cout << "threads unroll broadcast add test ...\n";

    std::cout << "float4: ";
    test_threads_unroll_add<float, 4>(2, 12, 2048 * 2048);

    std::cout << "half4: ";
    test_threads_unroll_add<half, 4>(2, 12, 2048 * 2048);
}
