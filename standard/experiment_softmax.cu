#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace reduce_utils {

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
    T val[vec_size];
};

template <typename T, typename func_t>
__device__ T warp_reduce(T val, func_t fn) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        // Warp Shuffle
        val = fn(val, __shfl_down_sync(0xffffffff, val, offset, 32));
    }
    return val;
}

template <typename T, typename func_t>
__device__ void block_reduce(T &val, T *shared, func_t fn) {
    val = warp_reduce<T, func_t>(val, fn);
    auto warp_id = threadIdx.x / 32;
    auto warp_tid = threadIdx.x % 32;
    if (warp_id == 0) {
        shared[warp_tid] = val;
    }
    __syncthreads();
    if (warp_id == 0 && warp_tid == 0) {
        for (int i = 1; i < blockDim.x / 32; ++i) {
            val = fn(val, shared[i]);
        }
    }
}

template <typename T, int vec_size, typename func_t>
__device__ T thread_reduce(T *block_start, int n, func_t fn, T ident) {
    using vec_t = aligned_array<T, vec_size>;
    T acc = ident;
    for (int i = threadIdx.x * vec_size; i < n; i += blockDim.x * vec_size) {
        auto input_vec = *reinterpret_cast<vec_t *>(&block_start[i]);
        for (int j = 0; j < vec_size; ++j) {
            acc = fn(acc, input_vec.val[j]);
        }
    }
    return acc;
}

} // namespace reduce_utils

template <typename T, int vec_size, int loops>
__global__ void softmax_kernel(T *in, T *out, size_t n, int *semaphores, T *out_max, T *out_sum, int *stg) {
    auto block_work_size = loops * blockDim.x * vec_size;
    auto block_offset = blockIdx.x * block_work_size;
    auto in_block = in + block_offset;
    auto out_block = out + block_offset;

    __shared__ T shared[32];
    auto remaining = ::min((int)(n - block_offset), block_work_size);

    auto max_fn = [](T a, T b) { return ::max(a, b); };
    T maxval = reduce_utils::thread_reduce<T, vec_size>(in_block, remaining, max_fn, -1e20);
    reduce_utils::block_reduce<T>(maxval, shared, max_fn);

    if (gridDim.x > 1) {
        // if (threadIdx.x == 0) {
        //     out_max[blockIdx.x] = maxval;
        // }
        // reduce_utils::sync_blocks(semaphores, stg);

        // auto nbr_remaining = ::min((int)(gridDim.x - block_offset), block_work_size);
        // maxval = reduce_utils::thread_reduce<T, vec_size>(out_max + block_offset, nbr_remaining, max_fn, -1e20);
        // reduce_utils::block_reduce<T>(maxval, shared, max_fn);
    }

    if (threadIdx.x == 0) {
        shared[0] = maxval;
    }
    __syncthreads();
    maxval = shared[0];

    auto sum_fn = [](T a, T b) { return a + b; };
    auto sum_exp_fn = [maxval](T a, T b) { return a + ::exp(b - maxval); };
    T sumval = reduce_utils::thread_reduce<T, vec_size>(in_block, remaining, sum_exp_fn, 0);
    reduce_utils::block_reduce<T>(sumval, shared, sum_fn);

    if (gridDim.x > 1) {
        // if (threadIdx.x == 0) {
        //     atomicAdd(out_sum, sumval);
        // }
        // reduce_utils::sync_blocks(semaphores, stg);
    } else {
        if (threadIdx.x == 0) {
            shared[0] = sumval;
        }
        __syncthreads();
        sumval = shared[0];
    }

    using vec_t = reduce_utils::aligned_array<T, vec_size>;
    for (int i = threadIdx.x * vec_size; i < remaining; i += blockDim.x * vec_size) {
        auto input_vec = *reinterpret_cast<vec_t *>(&in_block[i]);
        for (int j = 0; j < vec_size; ++j) {
            input_vec.val[j] = ::exp(input_vec.val[j] - maxval) / sumval;
        }
        *reinterpret_cast<vec_t *>(&out_block[i]) = input_vec;
    }
}

template <typename T, int vec_size>
float softmax_gpu(T *in, T *out, size_t n) {
    const int block_size = 256;
    const int loops = 4;
    auto block_work_size = block_size * vec_size * loops;
    auto nblocks = (n + block_work_size - 1) / block_work_size;

    int *semaphores, *stg;
    T *out_max, *out_sum;
    cudaMalloc(&semaphores, sizeof(int));
    cudaMalloc(&stg, sizeof(int));
    cudaMalloc(&out_max, nblocks * sizeof(T));
    cudaMalloc(&out_sum, sizeof(T));
    cudaMemset(semaphores, 0, sizeof(int));
    cudaMemset(stg, 0, sizeof(int));
    cudaMemset(out_sum, 0, sizeof(T));

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(nblocks);
    std::cout << "nblocks: " << nblocks << "\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    softmax_kernel<T, vec_size, loops><<<numBlocks, threadsPerBlock>>>(
        in, out, n, semaphores, out_max, out_sum, stg);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(semaphores);
    cudaFree(stg);
    cudaFree(out_max);
    cudaFree(out_sum);

    return ms;
}

template <typename scalar_t>
void softmax_ref(scalar_t *in, scalar_t *out, size_t n) {
    scalar_t maxval = -1e20;
    for (size_t i = 0; i < n; i++) {
        maxval = std::max(maxval, in[i]);
    }
    scalar_t sumval = 0;
    for (size_t i = 0; i < n; i++) {
        sumval += ::exp(in[i] - maxval);
    }
    for (size_t i = 0; i < n; i++) {
        out[i] = ::exp(in[i] - maxval) / sumval;
    }
}

template <int vec_size, typename scalar_t>
void test_softmax_gpu(size_t n) {
    auto in_cpu = new scalar_t[n];
    auto out_cpu = new scalar_t[n];
    auto out_ref = new scalar_t[n];
    for (size_t i = 0; i < n; i++) {
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    softmax_ref(in_cpu, out_ref, n);

    scalar_t *in_cuda, *out_cuda;
    cudaMalloc(&in_cuda, n * sizeof(scalar_t));
    cudaMalloc(&out_cuda, n * sizeof(scalar_t));
    cudaMemcpy(in_cuda, in_cpu, n * sizeof(scalar_t), cudaMemcpyHostToDevice);

    float timems;
    for (int i = 0; i < 4; i++) {
        timems = softmax_gpu<scalar_t, vec_size>(in_cuda, out_cuda, n);
    }
    std::cout << "timems:" << timems << "\n";

    float total_GBytes = (n + n) * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    cudaMemcpy(out_cpu, out_cuda, n * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        auto diff = (float)out_cpu[i] - (float)out_ref[i];
        if (i < 10)
            std::cout << out_cpu[i] << ", " << out_ref[i] << "\n";
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
    delete out_ref;
}

int main() {
    std::cout << "1GB softmax test ...\n";
    std::cout << "float4: ";
    test_softmax_gpu<4, float>(1024 * 4);
}
