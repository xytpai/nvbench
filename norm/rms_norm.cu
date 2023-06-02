#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <vector>
#include <math.h>
#include <assert.h>

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    }
    return val;
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T val, T *shared, const int tid) {
    const int w_tid = tid & 31;
    const int wid = tid >> 5;
    val = warp_reduce_sum(val);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0) {
        val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    return shared[0];
}

template <typename scalar_t, int BLOCK_LOOPS, int BLOCK_THREADS, int VEC_SIZE>
__global__ void rms_norm_cuda_kernel(
    scalar_t *output,
    const scalar_t *input,
    const scalar_t *weight_,
    const int norm_size) {
    constexpr float eps = 1e-5;
    using acc_t = float;
    using vec_t = aligned_array<scalar_t, VEC_SIZE>;
    __shared__ acc_t data[32];
    auto batch_offset = blockIdx.x * norm_size;
    auto input_b = const_cast<scalar_t *>(input + batch_offset);
    auto output_b = output + batch_offset;
    auto weight = const_cast<scalar_t *>(weight_);
    auto tid = threadIdx.x;

    acc_t sum = 0;
    scalar_t input_regs[BLOCK_LOOPS][VEC_SIZE];
    scalar_t weight_regs[BLOCK_LOOPS][VEC_SIZE];
#pragma unroll
    for (int i = 0; i < BLOCK_LOOPS; i++) {
        int idx = (i * BLOCK_THREADS + tid) * VEC_SIZE;
        int remaining = norm_size - idx;
        if (remaining >= VEC_SIZE) {
            auto vec_i = *reinterpret_cast<vec_t *>(input_b + idx);
            auto vec_w = *reinterpret_cast<vec_t *>(weight + idx);
#pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                sum += vec_i.val[j] * vec_i.val[j];
                input_regs[i][j] = vec_i.val[j];
                weight_regs[i][j] = vec_w.val[j];
            }
        } else {
            for (int j = 0; j < remaining; j++) {
                scalar_t temp = input_b[idx + j];
                sum += temp * temp;
                input_regs[i][j] = temp;
                weight_regs[i][j] = weight[idx + j];
            }
        }
    }
    sum = block_reduce_sum<acc_t>(sum, data, tid);
    auto mq = ::rsqrt(sum / (acc_t)(norm_size) + eps);
#pragma unroll
    for (int i = 0; i < BLOCK_LOOPS; i++) {
        int idx = (i * BLOCK_THREADS + tid) * VEC_SIZE;
        int remaining = norm_size - idx;
        if (remaining >= VEC_SIZE) {
            vec_t vec;
#pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                vec.val[j] = input_regs[i][j] * mq * weight_regs[i][j];
            }
            *reinterpret_cast<vec_t *>(output_b + idx) = vec;
        } else {
            for (int j = 0; j < remaining; j++) {
                output_b[idx + j] = input_regs[i][j] * mq * weight_regs[i][j];
            }
        }
    }
}

template <typename scalar_t, int BLOCK_LOOPS, int BLOCK_THREADS, int VEC_SIZE = 4>
float rms_norm_cuda_impl(
    scalar_t *output,
    const scalar_t *input,
    const scalar_t *weight,
    const int batch_size,
    const int norm_size) {
    dim3 block(BLOCK_THREADS);
    dim3 grid(batch_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rms_norm_cuda_kernel<scalar_t, BLOCK_LOOPS, BLOCK_THREADS, VEC_SIZE><<<grid, block>>>(output, input, weight, norm_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

inline uint64_t next_power2(uint64_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

template <typename scalar_t>
float rms_norm_cuda(
    scalar_t *output,
    const scalar_t *input,
    const scalar_t *weight,
    const int batch_size,
    const int norm_size) {
    switch (next_power2(norm_size)) {
    case 16384:
        return rms_norm_cuda_impl<scalar_t, 2, 1024, 8>(output, input, weight, batch_size, norm_size);
    case 8192:
        return rms_norm_cuda_impl<scalar_t, 2, 1024>(output, input, weight, batch_size, norm_size);
    case 4096:
        return rms_norm_cuda_impl<scalar_t, 2, 512>(output, input, weight, batch_size, norm_size);
    case 2048:
        return rms_norm_cuda_impl<scalar_t, 2, 256>(output, input, weight, batch_size, norm_size);
    case 1024:
    case 512:
    case 256:
        return rms_norm_cuda_impl<scalar_t, 1, 256>(output, input, weight, batch_size, norm_size);
    default:
        std::cout << "kernel not ready\n";
        return 0.0;
    }
}

template <typename scalar_t>
void rms_norm_cpu(
    scalar_t *output,
    const scalar_t *input,
    const scalar_t *weight,
    const int batch_size,
    const int norm_size) {
    constexpr float eps = 1e-5;
    for (int b = 0; b < batch_size; b++) {
        auto input_b = input + b * norm_size;
        auto output_b = output + b * norm_size;
        float sum = 0.0f;
        for (int n = 0; n < norm_size; n++) {
            sum += input_b[n] * input_b[n];
        }
        auto r = 1.0f / std::sqrt(sum / (float)norm_size + eps);
        for (int n = 0; n < norm_size; n++) {
            output_b[n] = input_b[n] * r * weight[n];
        }
    }
}

struct rms_norm_sizes {
    int batch_size, norm_size;
    rms_norm_sizes(int batch_size_, int norm_size_) :
        batch_size(batch_size_), norm_size(norm_size_) {
    }
};

int main() {
    std::cout << "rms_norm\n";
    using scalar_t = float;

    std::vector<rms_norm_sizes> sizes;
    sizes.push_back(rms_norm_sizes(16 * 1024, 256));
    sizes.push_back(rms_norm_sizes(16 * 1024, 512));
    sizes.push_back(rms_norm_sizes(16 * 1024, 1024));
    sizes.push_back(rms_norm_sizes(16 * 1024, 2048));
    sizes.push_back(rms_norm_sizes(16 * 1024, 4096));
    sizes.push_back(rms_norm_sizes(16 * 1024, 8192));
    sizes.push_back(rms_norm_sizes(16 * 1024, 16384));

    for (auto size : sizes) {
        int batch_size = size.batch_size;
        int norm_size = size.norm_size;

        std::cout << "batch_size=" << batch_size << ", norm_size=" << norm_size << "\n";

        auto input_cpu = new scalar_t[batch_size * norm_size];
        auto weight_cpu = new scalar_t[norm_size];
        auto out_cpu = new scalar_t[batch_size * norm_size];

        for (int i = 0; i < batch_size * norm_size; i++)
            input_cpu[i] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);
        for (int i = 0; i < norm_size; i++)
            weight_cpu[i] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);

        scalar_t *input_cuda, *weight_cuda, *out_cuda;
        cudaMalloc(&input_cuda, batch_size * norm_size * sizeof(scalar_t));
        cudaMalloc(&weight_cuda, norm_size * sizeof(scalar_t));
        cudaMalloc(&out_cuda, batch_size * norm_size * sizeof(scalar_t));
        cudaMemcpy(input_cuda, input_cpu, batch_size * norm_size * sizeof(scalar_t), cudaMemcpyHostToDevice);
        cudaMemcpy(weight_cuda, weight_cpu, norm_size * sizeof(scalar_t), cudaMemcpyHostToDevice);

        rms_norm_cpu<scalar_t>(out_cpu, input_cpu, weight_cpu, batch_size, norm_size);
        auto timems = rms_norm_cuda<scalar_t>(out_cuda, input_cuda, weight_cuda, batch_size, norm_size);

        double total_gbytes = ((double)batch_size * norm_size * 2 + norm_size) * sizeof(scalar_t) / 1000.0 / 1000 / 1000;
        std::cout << timems << " ms, " << total_gbytes / (timems / 1000.0) << " gbps\n";

        auto out_cuda_ = new scalar_t[batch_size * norm_size];
        cudaMemcpy(out_cuda_, out_cuda, batch_size * norm_size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        auto maxdiff = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < batch_size * norm_size; i++) {
            // if (i < 100)
            //     std::cout << out_cuda_[i] << " " << out_cpu[i] << std::endl;
            auto diff = std::abs(out_cuda_[i] - out_cpu[i]);
            maxdiff = std::max(maxdiff, diff);
        }
        std::cout << "maxdiff: " << maxdiff << std::endl;

        cudaFree(input_cuda);
        cudaFree(weight_cuda);
        cudaFree(out_cuda);
        delete[] input_cpu;
        delete[] weight_cpu;
        delete[] out_cpu;
        delete[] out_cuda_;
    }
    return 0;
}
