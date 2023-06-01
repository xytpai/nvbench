#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

template <typename scalar_t, int VEC_A, int VEC_B>
struct mma_reg_t {
    using vec_a_t = aligned_array<scalar_t, VEC_A>;
    using vec_b_t = aligned_array<scalar_t, VEC_B>;
    union {
        vec_a_t a_vec;
        scalar_t a[VEC_A];
    };
    union {
        vec_b_t b_vec;
        scalar_t b[VEC_B];
    };
};

template <typename scalar_t, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TH_M, int TH_N>
__device__ void block_mma(scalar_t (*o)[TH_M * TH_N], scalar_t *a, scalar_t *b) {
    static_assert(BLOCK_M % 64 == 0);
    static_assert(BLOCK_N % 64 == 0);
    static_assert(BLOCK_K % 16 == 0);

    constexpr int MMA_BLOCK_M = 16 * TH_M;
    constexpr int MMA_BLOCK_N = 16 * TH_N;

    constexpr int LANE_M = BLOCK_M / MMA_BLOCK_M;
    constexpr int LANE_N = BLOCK_N / MMA_BLOCK_N;

    auto tid = threadIdx.x;

#pragma unroll
    for (int lm = 0; lm < LANE_M; lm++) {
        auto offset_lane_m = lm * MMA_BLOCK_M;
#pragma unroll
        for (int ln = 0; ln < LANE_N; ln++) {
            auto offset_lane_n = ln * MMA_BLOCK_N;
            auto offset_th_m = offset_lane_m + tid / 16 * TH_M;
            auto offset_th_n = offset_lane_n + tid % 16 * TH_N;
            using vec_a_t = aligned_array<scalar_t, TH_M>;
            using vec_b_t = aligned_array<scalar_t, TH_N>;
            mma_reg_t<scalar_t, TH_M, TH_N> reg;
            for (int k = 0; k < BLOCK_K; k++) {
                reg.a_vec = *reinterpret_cast<vec_a_t *>(a + k * BLOCK_M + offset_th_m);
                reg.b_vec = *reinterpret_cast<vec_b_t *>(b + k * BLOCK_N + offset_th_n);
                for (int i = 0; i < TH_M; i++) {
                    for (int j = 0; j < TH_N; j++) {
                        o[lm * LANE_N + ln][i * TH_N + j] += reg.a[i] * reg.b[j];
                    }
                }
            }
        }
    }
}

template <typename scalar_t = float, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TH_M, int TH_N, bool DOUBLE_BUFFER>
__global__ void gemm_cuda_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    static_assert(BLOCK_M >= 64);
    static_assert(BLOCK_M % TH_M == 0);
    static_assert(BLOCK_N >= 64);
    static_assert(BLOCK_N % TH_N == 0);

    constexpr int VEC_SIZE = TH_N;
    constexpr int LANE_M = BLOCK_M / 16 / TH_M;
    constexpr int LANE_N = BLOCK_N / 16 / TH_N;
    using vec_t = aligned_array<scalar_t, VEC_SIZE>;

    auto tid = threadIdx.x;

    auto block_a_begin = (blockIdx.y * BLOCK_M) * k;
    auto block_a_end = block_a_begin + k;
    constexpr int block_a_step = BLOCK_K;
    auto block_b_begin = blockIdx.x * BLOCK_N;
    auto block_b_step = BLOCK_K * n;

    extern __shared__ scalar_t slm[];
    auto as = reinterpret_cast<scalar_t *>(slm);
    scalar_t *bs;
    constexpr int BLOCK_KM_SIZE = BLOCK_K * BLOCK_M;
    constexpr int BLOCK_KN_SIZE = BLOCK_K * BLOCK_N;
    if constexpr (DOUBLE_BUFFER) {
        bs = as + BLOCK_KM_SIZE * 2;
    } else {
        bs = as + BLOCK_KM_SIZE;
    }

    scalar_t o_reg[LANE_M * LANE_N][TH_M * TH_N];
    for (int i = 0; i < LANE_M * LANE_N; i++) {
        for (int j = 0; j < TH_M * TH_N; j++) {
            o_reg[i][j] = 0;
        }
    }

    constexpr int LDG_A_X_WORK_SIZE = BLOCK_K / VEC_SIZE;
    constexpr int LDG_B_X_WORK_SIZE = BLOCK_N / VEC_SIZE;
    auto ldg_a_vec_idx = tid % LDG_A_X_WORK_SIZE;
    auto ldg_b_vec_idx = tid % LDG_B_X_WORK_SIZE;

    int write_stage_idx = 0;
    int read_stage_idx = DOUBLE_BUFFER ? 1 : 0;

    for (int a_begin = block_a_begin, b_begin = block_b_begin;
         a_begin < block_a_end; a_begin += block_a_step, b_begin += block_b_step) {
        // load data block to register
        constexpr int LDG_REG_A_COUNT = BLOCK_KM_SIZE / VEC_SIZE / 256;
        constexpr int LDG_REG_B_COUNT = BLOCK_KN_SIZE / VEC_SIZE / 256;
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
        auto bs_vec = reinterpret_cast<vec_t *>(bs + write_stage_idx * BLOCK_KN_SIZE);
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            auto y = (256 * i + tid) / LDG_A_X_WORK_SIZE;
#pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                as[write_stage_idx * BLOCK_KM_SIZE + (ldg_a_vec_idx * VEC_SIZE + j) * BLOCK_M + y] = ldg_a_reg[i].val[j];
            }
        }
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            bs_vec[256 * i + tid] = ldg_b_reg[i];
        }
        if constexpr (DOUBLE_BUFFER) {
            read_stage_idx ^= 1;
            write_stage_idx ^= 1;
        }
        __syncthreads();
        block_mma<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K, TH_M, TH_N>(o_reg, as + read_stage_idx * BLOCK_KM_SIZE, bs + read_stage_idx * BLOCK_KN_SIZE);
        if constexpr (!DOUBLE_BUFFER) {
            __syncthreads();
        }
    }

    // write back
    int out_block_offset_y = blockIdx.y * BLOCK_M;
    int out_block_offset_x = blockIdx.x * BLOCK_N;
#pragma unroll
    for (int lm = 0; lm < LANE_M; lm++) {
        auto lane_y = out_block_offset_y + lm * 16 * TH_M + tid / 16 * TH_M;
#pragma unroll
        for (int ln = 0; ln < LANE_N; ln++) {
            auto th_x_base = out_block_offset_x + ln * 16 * TH_N + tid % 16 * TH_N;
#pragma unroll
            for (int i = 0; i < TH_M; i++) {
                auto th_y = lane_y + i;
                if (th_y < m && th_x_base < n) {
                    auto vec = *reinterpret_cast<vec_t *>(out + th_y * n + th_x_base);
#pragma unroll
                    for (int j = 0; j < TH_N; j++) {
                        vec.val[j] = alpha * o_reg[lm * LANE_N + ln][i * TH_N + j] + beta * vec.val[j];
                    }
                    *reinterpret_cast<vec_t *>(out + th_y * n + th_x_base) = vec;
                }
            }
        }
    }
}

template <typename scalar_t = float, int BLOCK_M = 128, int BLOCK_N = 64, int BLOCK_K = 16, int TH_M = 8, int TH_N = 4, bool DOUBLE_BUFFER = true>
float gemm_cuda_impl(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    assert(k % 4 == 0);
    assert(n % 4 == 0);

    dim3 block(256);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    auto slm_size = DOUBLE_BUFFER ? (BLOCK_M + BLOCK_N) * BLOCK_K * 2 : (BLOCK_M + BLOCK_N) * BLOCK_K;
    gemm_cuda_kernel<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K, TH_M, TH_N, DOUBLE_BUFFER><<<grid, block, slm_size * sizeof(scalar_t)>>>(out, a, b, m, n, k, alpha, beta);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

template <typename scalar_t>
float gemm_cuda(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    if (m <= 1024 && n <= 1024) {
        return gemm_cuda_impl<scalar_t, 128, 64, 16, 8, 4, true>(out, a, b, m, n, k, alpha, beta);
    } else {
        return gemm_cuda_impl<scalar_t, 128, 128, 16, 4, 4, false>(out, a, b, m, n, k, alpha, beta);
    }
}

template <typename scalar_t = float>
__global__ void gemm_cuda_ref_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    auto mi = blockIdx.y * 32 + threadIdx.y;
    auto ni = blockIdx.x * 32 + threadIdx.x;
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
    dim3 grid((n + 32 - 1) / 32, (m + 32 - 1) / 32);
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
    std::cout << "sgemm_simt\n";
    using scalar_t = float;

    std::vector<gemm_sizes> sizes;
    sizes.push_back(gemm_sizes(1024, 1024, 1024, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1028, 1028, 1028, 0.5, 0.5));
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
        std::cout << timems << " ms, " << total_gbytes / (timems / 1000.0) << " gbps, ";

        double tflops = ((double)2 * m * n * k) / (timems / 1000) * 1e-12;
        std::cout << tflops << " tflops\n";

        auto out_cuda_ref_ = new scalar_t[m * n];
        auto out_cuda_ = new scalar_t[m * n];
        cudaMemcpy(out_cuda_ref_, out_cuda_ref, m * n * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_cuda_, out_cuda, m * n * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        auto maxdiff = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < m * n; i++) {
            // if (i < 100)
            //     std::cout << out_cuda_[i] << " " << out_cuda_ref_[i] << "\n";
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
