#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cuda_fp16.h>
#include "utils.h"
using namespace std;

template <typename T, int vec_size>
__global__ void threads_unroll_copy_kernel(const T *in, T *out, const size_t n) {
    const int block_work_size = blockDim.x * vec_size;
    auto index = blockIdx.x * block_work_size + threadIdx.x;
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
        if (index < n) {
            out[index] = in[index];
        }
        index += blockDim.x;
    }
}

template <typename T, int vec_size>
float threads_unroll_copy(const T *in, T *out, size_t n) {
    const int block_size = 1024;
    const int block_work_size = block_size * vec_size;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    threads_unroll_copy_kernel<T, vec_size><<<numBlocks, threadsPerBlock>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

template <typename T, int vec_size>
void test_threads_unroll_copy(size_t n) {
    auto in_cpu = new T[n];
    auto out_cpu = new T[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = (T)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

    T *in_cuda, *out_cuda;
    cudaMalloc(&in_cuda, n * sizeof(T));
    cudaMalloc(&out_cuda, n * sizeof(T));

    cudaMemcpy(in_cuda, in_cpu, n * sizeof(T), cudaMemcpyHostToDevice);

    float timems;
    for (int i = 0; i < 300; i++)
        timems = threads_unroll_copy<T, vec_size>(in_cuda, out_cuda, n);

    float total_GBytes = (n + n) * sizeof(T) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    cudaMemcpy(out_cpu, out_cuda, n * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        auto diff = (float)(out_cpu[i] - in_cpu[i]);
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    cudaFree(in_cuda);
    cudaFree(out_cuda);
    delete in_cpu;
    delete out_cpu;
}

int main() {
    std::cout << "1GB threads unroll copy test ...\n";

    std::cout << "float1: ";
    test_threads_unroll_copy<float, 1>(1024 * 1024 * 256 + 2);
    std::cout << "float2: ";
    test_threads_unroll_copy<float, 2>(1024 * 1024 * 256 + 2);
    std::cout << "float4: ";
    test_threads_unroll_copy<float, 4>(1024 * 1024 * 256 + 2);
    std::cout << "float8: ";
    test_threads_unroll_copy<float, 8>(1024 * 1024 * 256 + 2);
    std::cout << "float16: ";
    test_threads_unroll_copy<float, 16>(1024 * 1024 * 256 + 2);

    std::cout << "half1: ";
    test_threads_unroll_copy<half, 1>(1024 * 1024 * 256 + 2);
    std::cout << "half2: ";
    test_threads_unroll_copy<half, 2>(1024 * 1024 * 256 + 2);
    std::cout << "half4: ";
    test_threads_unroll_copy<half, 4>(1024 * 1024 * 256 + 2);
    std::cout << "half8: ";
    test_threads_unroll_copy<half, 8>(1024 * 1024 * 256 + 2);
    std::cout << "half16: ";
    test_threads_unroll_copy<half, 16>(1024 * 1024 * 256 + 2);
}
