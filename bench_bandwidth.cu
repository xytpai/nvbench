#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
using namespace std;

#define FLOAT_N 4
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
    T val[vec_size];
};
typedef aligned_array<float, FLOAT_N> floatn;

template <typename T>
__global__ void threads_copy_kernel(
    const T *in, T *out,
    const size_t n) {
    const int block_work_size = blockDim.x * FLOAT_N;
    auto index = blockIdx.x * block_work_size + threadIdx.x * FLOAT_N;
    auto remaining = n - index;
    if (remaining < FLOAT_N) {
        for(int i=index;i<n;i++) {
            out[i] = in[i];
        }
    } else {
        auto in_vec = reinterpret_cast<floatn*>(const_cast<T*>(&in[index]));
        auto out_vec = reinterpret_cast<floatn*>(&out[index]);
        *out_vec = *in_vec;
    }
}

template <typename T>
float threads_copy(const T *in, T *out, size_t n) {
    const int block_size = 256;
    const int block_work_size = block_size * FLOAT_N;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    threads_copy_kernel<T><<<numBlocks, threadsPerBlock>>>(in, out, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int test_thcopyf4() {
    const size_t n = 1024*1024*1024 + 2; // 4GB
    auto in_cpu = new float[n];
    auto out_cpu = new float[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    float *in_cuda, *out_cuda;
    cudaMalloc(&in_cuda, n * sizeof(float));
    cudaMalloc(&out_cuda, n * sizeof(float));

    cudaMemcpy(in_cuda, in_cpu, n * sizeof(float), cudaMemcpyHostToDevice);

    auto timems = threads_copy(in_cuda, out_cuda, n);

    float total_GBytes = (n + n) * sizeof(float) / 1024.0 / 1024 / 1024;
    std::cout << total_GBytes / (timems/1000.0) << " gbps\n";

    cudaMemcpy(out_cpu, out_cuda, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        auto diff = out_cpu[i] - in_cpu[i];
        diff = diff > 0 ? diff : -diff;
        // std::cout<<diff<<"\n";
        if (diff > 0.1)
            return 1;
    }
    std::cout << "ok\n";

    cudaFree(in_cuda);
    cudaFree(out_cuda);
    delete in_cpu;
    delete out_cpu;
    return 0;
}

int main()
{
    std::cout << "4GB f4 thread copy test ...\n";
    for(int i=0;i<3;i++)
        test_thcopyf4();
}
