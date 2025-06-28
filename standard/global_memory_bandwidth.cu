#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <cuda_fp16.h>

#include "utils.h"
using namespace std;

template <typename T, int vec_size>
__global__ void threads_copy_kernel(const T *in, T *out, const size_t n) {
    const int block_work_size = blockDim.x * vec_size;
    auto index = blockIdx.x * block_work_size + threadIdx.x * vec_size;
    auto remaining = n - index;
    if (remaining < vec_size) {
        for (auto i = index; i < n; i++) {
            out[i] = in[i];
        }
    } else {
        using vec_t = aligned_array<T, vec_size>;
        auto in_vec = reinterpret_cast<vec_t *>(const_cast<T *>(&in[index]));
        auto out_vec = reinterpret_cast<vec_t *>(&out[index]);
        *out_vec = *in_vec;
    }
}

template <typename T, int vec_size>
float threads_copy(const T *in, T *out, size_t n) {
    const int block_size = 256;
    const int block_work_size = block_size * vec_size;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    threads_copy_kernel<T, vec_size><<<numBlocks, threadsPerBlock>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

template <int vec_size, typename scalar_t>
void test_threads_copy(size_t n) {
    auto in_cpu = new scalar_t[n];
    auto out_cpu = new scalar_t[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    scalar_t *in_cuda, *out_cuda;
    cudaMalloc(&in_cuda, n * sizeof(scalar_t));
    cudaMalloc(&out_cuda, n * sizeof(scalar_t));

    cudaMemcpy(in_cuda, in_cpu, n * sizeof(scalar_t), cudaMemcpyHostToDevice);

    float timems;
    for (int i = 0; i < 2; i++)
        timems = threads_copy<scalar_t, vec_size>(in_cuda, out_cuda, n);
    std::cout << "timems:" << timems << "\n";

    float total_GBytes = (n + n) * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    cudaMemcpy(out_cpu, out_cuda, n * sizeof(scalar_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        auto diff = (float)out_cpu[i] - (float)in_cpu[i];
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
    std::cout << "1GB threads copy test ...\n";
    std::cout << "float1: ";
    test_threads_copy<1, float>(1024 * 1024 * 256 + 2);
    std::cout << "float2: ";
    test_threads_copy<2, float>(1024 * 1024 * 256 + 2);
    std::cout << "float4: ";
    test_threads_copy<4, float>(1024 * 1024 * 256 + 2);
    std::cout << "float8: ";
    test_threads_copy<8, float>(1024 * 1024 * 256 + 2);
    std::cout << "half1: ";
    test_threads_copy<1, __half>((1024 * 1024 * 256 + 2) * 2);
}
