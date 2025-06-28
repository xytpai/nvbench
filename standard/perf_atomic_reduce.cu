#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
    T val[vec_size];
};

template <typename T>
__device__ T warp_reduce(T val) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        // Warp Shuffle
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    }
    return val;
}

template <typename T>
__device__ void block_reduce(T &val, T *shared) {
    val = warp_reduce<T>(val);
    auto warp_id = threadIdx.x / 32;
    auto warp_tid = threadIdx.x % 32;
    if (warp_id == 0) {
        shared[warp_tid] = val;
    }
    __syncthreads();
    if (warp_id == 0 && warp_tid == 0) {
        for (int i = 1; i < blockDim.x / 32; ++i) {
            val += shared[i];
        }
    }
}

template <typename T, int vec_size, int loops>
__global__ void reduce_kernel(T *in_, T *out, size_t reduce_size) {
    using vec_t = aligned_array<T, vec_size>;

    auto block_work_size = loops * blockDim.x * vec_size;
    auto block_offset = blockIdx.x * block_work_size;
    auto in = in_ + block_offset;

    __shared__ T shared[32];
    T acc = 0;

    auto remaining = reduce_size - block_offset;
    remaining = remaining > block_work_size ? block_work_size : remaining;

    // Thread Reduce
    for (int i = threadIdx.x * vec_size; i < remaining; i += blockDim.x * vec_size) {
        auto input_vec = *reinterpret_cast<vec_t *>(&in[i]);
        for (int j = 0; j < vec_size; ++j) {
            acc += input_vec.val[j];
        }
    }

    block_reduce<T>(acc, shared);

    if (threadIdx.x == 0) {
        atomicAdd(out, acc);
    }
}

template <typename T, int vec_size>
float reduce_gpu(T *in, T *out, size_t reduce_size) {
    const int block_size = 256;
    const int loops = 4;
    auto block_work_size = block_size * vec_size * loops;

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((reduce_size + block_work_size - 1) / block_work_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock>>>(in, out, reduce_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

template <int vec_size, typename scalar_t>
void test_reduce_gpu(size_t n) {
    auto in_cpu = new scalar_t[n];
    auto out_cpu = new scalar_t[1];
    for (size_t i = 0; i < n; i++) {
        in_cpu[i] = i / scalar_t(n);
    }
    scalar_t *in_cuda, *out_cuda;
    cudaMalloc(&in_cuda, n * sizeof(scalar_t));
    cudaMalloc(&out_cuda, n * sizeof(1));
    cudaMemcpy(in_cuda, in_cpu, n * sizeof(scalar_t), cudaMemcpyHostToDevice);

    float timems;
    for (int i = 0; i < 4; i++) {
        cudaMemset(out_cuda, 0, sizeof(scalar_t));
        timems = reduce_gpu<scalar_t, vec_size>(in_cuda, out_cuda, n);
    }
    std::cout << "timems:" << timems << "\n";

    float total_GBytes = (n + 1) * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    cudaMemcpy(out_cpu, out_cuda, 1 * sizeof(scalar_t), cudaMemcpyDeviceToHost);

    scalar_t ref = n / (scalar_t)2;
    std::cout << "ref: " << ref << ", out_cuda: " << out_cpu[0] << "\n";

    cudaFree(in_cuda);
    cudaFree(out_cuda);
    delete in_cpu;
    delete out_cpu;
}

int main() {
    std::cout << "1GB reduce test ...\n";
    std::cout << "float4: ";
    test_reduce_gpu<4, float>(1024 * 1024 * 256);
}
