#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
    __host__ __device__ aligned_array() {
    }
    __host__ __device__ aligned_array(scalar_t v) {
#pragma unroll
        for (int i = 0; i < vec_size; i++) val[i] = v;
    }
};

template <typename scalar_t = float, int BLOCK_M, int BLOCK_N, int BLOCK_K = 16>
__global__ void gemm_cuda_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    using vec_t = aligned_array<scalar_t, 4>;
    static_assert(BLOCK_M % 32 == 0);
    static_assert(BLOCK_N % 64 == 0);
    static_assert(BLOCK_K % 16 == 0);

    constexpr int TILE_M = BLOCK_M / 32;
    constexpr int TILE_N = BLOCK_N / 64;

    auto tid = threadIdx.x;
    auto w_tid = tid % 32;
    auto wid = tid / 32;

    auto block_a_begin = (blockIdx.x * BLOCK_M) * k;
    auto block_a_end = block_a_begin + k;
    auto block_a_step = BLOCK_K;
    auto block_b_begin = blockIdx.y * BLOCK_N;
    auto block_b_step = BLOCK_K * n;

    __shared__ vec_t As[2][BLOCK_K * BLOCK_M / 4];
    __shared__ vec_t Bs[2][BLOCK_K * BLOCK_N / 4];

    vec_t c_reg[TILE_M * TILE_N][2] = {{vec_t((scalar_t)(0.0))}};

    constexpr int LDG_A_X_WORK_SIZE = BLOCK_K / 4;
    constexpr int LDG_B_X_WORK_SIZE = BLOCK_N / 4;
    auto ldg_a_vec_idx = tid % LDG_A_X_WORK_SIZE;
    auto ldg_b_vec_idx = tid % LDG_B_X_WORK_SIZE;

    int write_stage_idx = 1;
    int read_stage_idx = 0;

    for (int a_begin = block_a_begin, b_begin = block_b_begin;
         a_begin < block_a_end; a_begin += block_a_step, b_begin += block_b_step) {
        // load data block to register
        constexpr int LDG_REG_A_COUNT = BLOCK_K * BLOCK_M / 4 / 256;
        constexpr int LDG_REG_B_COUNT = BLOCK_K * BLOCK_N / 4 / 256;
        vec_t ldg_a_reg[LDG_REG_A_COUNT];
        vec_t ldg_b_reg[LDG_REG_B_COUNT];
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            ldg_a_reg[i] = reinterpret_cast<vec_t *>(const_cast<scalar_t *>(a) + a_begin
                                                     + ((256 * i + tid) / LDG_A_X_WORK_SIZE) * k)[ldg_a_vec_idx];
        }
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            ldg_b_reg[i] = reinterpret_cast<vec_t *>(const_cast<scalar_t *>(b) + b_begin
                                                     + ((256 * i + tid) / LDG_B_X_WORK_SIZE) * n)[ldg_b_vec_idx];
        }

        // transpose to shared local memory
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            auto y = (256 * i + tid) / LDG_A_X_WORK_SIZE;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                As[write_stage_idx][(ldg_a_vec_idx * 4 + j) * (BLOCK_M / 4) + y / 4].val[y % 4] = ldg_a_reg[i].val[j];
            }
        }
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            Bs[write_stage_idx][256 * i + tid] = ldg_b_reg[i];
        }
        read_stage_idx ^= 1;
        write_stage_idx ^= 1;
        __syncthreads();

        // As: BLOCK_K, BLOCK_M/4, 4
        // Bs: BLOCK_K, BLOCK_N/4, 4
    }

    // write back
    int out_block_offset_y = blockIdx.x * BLOCK_M;
    int out_block_offset_x = blockIdx.y * BLOCK_N;
#pragma unroll
    for (int i = 0; i < TILE_M; i++) {
        int out_tile_offset_y = out_block_offset_y + i * 32;
#pragma unroll
        for (int j = 0; j < TILE_N; j++) {
            int out_vec_offset_x = out_block_offset_x + j * 64 + wid % 4 * 16 + w_tid % 4 * 4;
#pragma unroll
            for (int vi = 0; vi < 2; vi++) {
                int out_vec_offset_y = out_tile_offset_y + wid / 4 * 16 + w_tid / 4 * 2 + vi;
                if (out_vec_offset_y < m && out_vec_offset_x < n) {
                    auto offset = out_vec_offset_y * n + out_vec_offset_x;
                    vec_t out_ = *reinterpret_cast<vec_t *>(out + offset);
                    for (int vec_i = 0; vec_i < 4; vec_i++) {
                        out_.val[vec_i] = alpha * c_reg[i * TILE_N + j][vi].val[vec_i] + beta * out_.val[vec_i];
                    }
                    *reinterpret_cast<vec_t *>(out + offset) = out_;
                }
            }
        }
    }
}

template <typename scalar_t = float, int BLOCK_M = 64, int BLOCK_N = 64>
float gemm_cuda(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    assert(k % 4 == 0);
    assert(n % 4 == 0);

    dim3 block(256);
    dim3 grid((m + BLOCK_M - 1) / BLOCK_M, (n + BLOCK_N - 1) / BLOCK_N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemm_cuda_kernel<scalar_t, BLOCK_M, BLOCK_N><<<grid, block>>>(out, a, b, m, n, k, alpha, beta);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

template <typename scalar_t = float>
__global__ void gemm_cuda_ref_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    auto mi = blockIdx.x * 32 + threadIdx.x;
    auto ni = blockIdx.y * 32 + threadIdx.y;
    if (mi < m && ni < n) {
        float acc = 0.f;
        for (int ki = 0; ki < k; ki++) {
            acc += a[mi * k + ki] * b[ki * n + ni];
        }
        out[mi * n + ni] = alpha * acc + beta * out[mi * n + ni];
    }
}

template <typename scalar_t = float>
void gemm_cuda_ref(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    dim3 block(32, 32);
    dim3 grid((m + 32 - 1) / 32, (n + 32 - 1) / 32);
    gemm_cuda_ref_kernel<scalar_t><<<grid, block>>>(out, a, b, m, n, k, alpha, beta);
    cudaDeviceSynchronize();
}

struct gemm_sizes {
    int m, n, k;
    float alpha, beta;
    gemm_sizes(int m_, int n_, int k_, float a, float b) :
        m(m_), n(n_), k(k_), alpha(a), beta(b) {
    }
};

int main() {
    using scalar_t = float;

    std::vector<gemm_sizes> sizes;
    sizes.push_back(gemm_sizes(1024, 1024, 1024, 0.5, 0.5));
    sizes.push_back(gemm_sizes(2048, 2048, 2048, 0.5, 0.5));
    sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));

    for (auto size : sizes) {
        int m = size.m;
        int n = size.n;
        int k = size.k;
        auto alpha = size.alpha;
        auto beta = size.beta;

        std::cout << "m=" << m << ", n=" << n << ", k=" << k
                  << ", alpha=" << alpha << ", beta=" << beta << "\n";

        auto a_cpu = new scalar_t[m * k];
        auto b_cpu = new scalar_t[k * n];
        auto out_cpu = new scalar_t[m * n];
        for (int i = 0; i < m * k; i++)
            a_cpu[i] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);
        for (int i = 0; i < k * n; i++)
            b_cpu[i] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);
        for (int i = 0; i < m * n; i++)
            out_cpu[i] = static_cast<scalar_t>(rand()) / static_cast<scalar_t>(RAND_MAX);

        scalar_t *a_cuda, *b_cuda, *out_cuda;
        cudaMalloc(&a_cuda, m * k * sizeof(scalar_t));
        cudaMalloc(&b_cuda, k * n * sizeof(scalar_t));
        cudaMalloc(&out_cuda, m * n * sizeof(scalar_t));
        cudaMemcpy(a_cuda, a_cpu, m * k * sizeof(scalar_t), cudaMemcpyHostToDevice);
        cudaMemcpy(b_cuda, b_cpu, k * n * sizeof(scalar_t), cudaMemcpyHostToDevice);
        cudaMemcpy(out_cuda, out_cpu, m * n * sizeof(scalar_t), cudaMemcpyHostToDevice);

        scalar_t *a_cuda_ref, *b_cuda_ref, *out_cuda_ref;
        cudaMalloc(&a_cuda_ref, m * k * sizeof(scalar_t));
        cudaMalloc(&b_cuda_ref, k * n * sizeof(scalar_t));
        cudaMalloc(&out_cuda_ref, m * n * sizeof(scalar_t));
        cudaMemcpy(a_cuda_ref, a_cpu, m * k * sizeof(scalar_t), cudaMemcpyHostToDevice);
        cudaMemcpy(b_cuda_ref, b_cpu, k * n * sizeof(scalar_t), cudaMemcpyHostToDevice);
        cudaMemcpy(out_cuda_ref, out_cpu, m * n * sizeof(scalar_t), cudaMemcpyHostToDevice);

        gemm_cuda_ref<scalar_t>(out_cuda_ref, a_cuda_ref, b_cuda_ref, m, n, k, alpha, beta);
        auto timems = gemm_cuda<scalar_t>(out_cuda, a_cuda, b_cuda, m, n, k, alpha, beta);

        double total_gbytes = ((double)m * k + k * n + m * n + m * n) * sizeof(scalar_t) / 1000.0 / 1000 / 1000;
        std::cout << total_gbytes / (timems / 1000.0) << " gbps, ";

        double tflops = ((double)2 * m * n * k) / (timems / 1000) * 1e-12;
        std::cout << tflops << " tflops\n";

        auto out_cuda_ref_ = new scalar_t[m * n];
        auto out_cuda_ = new scalar_t[m * n];
        cudaMemcpy(out_cuda_ref_, out_cuda_ref, m * n * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_cuda_, out_cuda, m * n * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        auto maxdiff = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < m * n; i++) {
            auto diff = std::abs(out_cuda_[i] - out_cuda_ref_[i]);
            maxdiff = std::max(maxdiff, diff);
        }
        std::cout << "maxdiff: " << maxdiff << std::endl;

        cudaFree(a_cuda);
        cudaFree(b_cuda);
        cudaFree(out_cuda);
        cudaFree(a_cuda_ref);
        cudaFree(b_cuda_ref);
        cudaFree(out_cuda_ref);
        delete[] a_cpu;
        delete[] b_cpu;
        delete[] out_cpu;
        delete[] out_cuda_;
    }
    return 0;
}
