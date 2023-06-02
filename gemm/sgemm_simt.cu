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

template <typename scalar_t, int BLOCK_K, int BLOCK_M_WARPS, int BLOCK_N_WARPS,
          int WARP_M_LANES, int WARP_N_LANES, int WARP_M_THREADS, int WARP_N_THREADS, int VEC_M, int VEC_N>
__device__ void block_mma(scalar_t (*o)[VEC_M * VEC_N], scalar_t *a, scalar_t *b, int wid, int w_tid) {
    constexpr int LANE_M = WARP_M_THREADS * VEC_M;
    constexpr int LANE_N = WARP_N_THREADS * VEC_N;
    constexpr int WARP_M = WARP_M_LANES * LANE_M;
    constexpr int WARP_N = WARP_N_LANES * LANE_N;
    constexpr int BLOCK_M = BLOCK_M_WARPS * WARP_M;
    constexpr int BLOCK_N = BLOCK_N_WARPS * WARP_N;
    using a_vec_t = aligned_array<scalar_t, VEC_M>;
    using b_vec_t = aligned_array<scalar_t, VEC_N>;

    auto warp_offset_y = wid / BLOCK_N_WARPS * WARP_M;
    auto warp_offset_x = wid % BLOCK_N_WARPS * WARP_N;

#pragma unroll
    for (int k = 0; k < BLOCK_K; k++) {
        
#pragma unroll
        for (int lm = 0; lm < WARP_M_LANES; lm++) {
#pragma unroll
            for (int ln = 0; ln < WARP_N_LANES; ln++) {
                auto lane_offset_y = warp_offset_y + lm * LANE_M;
                auto lane_offset_x = warp_offset_x + ln * LANE_N;
                auto th_offset_y = lane_offset_y + w_tid / WARP_N_THREADS * VEC_M;
                auto th_offset_x = lane_offset_x + w_tid % WARP_N_THREADS * VEC_N;
                mma_reg_t<scalar_t, VEC_M, VEC_N> reg;
                reg.a_vec = *reinterpret_cast<a_vec_t *>(a + k * BLOCK_M + th_offset_y);
                reg.b_vec = *reinterpret_cast<b_vec_t *>(b + k * BLOCK_N + th_offset_x);
#pragma unroll
                for (int i = 0; i < VEC_M; i++) {
#pragma unroll
                    for (int j = 0; j < VEC_N; j++) {
                        o[lm * WARP_N_LANES + ln][i * VEC_N + j] += reg.a[i] * reg.b[j];
                    }
                }
            }
        }
    }
}

template <typename scalar_t,
          int BLOCK_K,
          int BLOCK_M_WARPS, int BLOCK_N_WARPS,
          int WARP_M_LANES, int WARP_N_LANES,
          int WARP_M_THREADS, int WARP_N_THREADS,
          int VEC_M, int VEC_N,
          bool DOUBLE_BUFFER>
__global__ __launch_bounds__(256) void gemm_cuda_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    static_assert(BLOCK_M_WARPS * BLOCK_N_WARPS == 8);
    static_assert(WARP_M_THREADS * WARP_N_THREADS == 32);
    constexpr int WARP_M = WARP_M_LANES * WARP_M_THREADS * VEC_M;
    constexpr int WARP_N = WARP_N_LANES * WARP_N_THREADS * VEC_N;
    constexpr int BLOCK_M = BLOCK_M_WARPS * WARP_M;
    constexpr int BLOCK_N = BLOCK_N_WARPS * WARP_N;

    // get idx
    auto tid = threadIdx.x;
    auto wid = tid >> 5;
    auto w_tid = tid & 31;
    auto block_y = blockIdx.y;
    auto block_x = blockIdx.x;

    // get slm
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

    // init o_reg
    scalar_t o_reg[WARP_M_LANES * WARP_N_LANES][VEC_M * VEC_N] = {{(scalar_t)0}};

    constexpr int LDG_VEC_SIZE = 4;
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;
    constexpr int LDG_A_X_THREADS = BLOCK_K / LDG_VEC_SIZE;
    constexpr int LDG_B_X_THREADS = BLOCK_N / LDG_VEC_SIZE;
    auto ldg_a_vec_idx = tid % LDG_A_X_THREADS;
    auto ldg_b_vec_idx = tid % LDG_B_X_THREADS;
    constexpr int LDG_REG_A_COUNT = BLOCK_KM_SIZE / LDG_VEC_SIZE / 256;
    constexpr int LDG_REG_B_COUNT = BLOCK_KN_SIZE / LDG_VEC_SIZE / 256;
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);

    int write_stage_idx = 0;
    int read_stage_idx = DOUBLE_BUFFER ? 1 : 0;

    for (
        int a_begin = block_y * BLOCK_M * k, b_begin = block_x * BLOCK_N;
        a_begin < block_y * BLOCK_M * k + k;
        a_begin += BLOCK_K, b_begin += BLOCK_K * n) {
        {
            // load data block to register
            ldg_vec_t ldg_a_reg[LDG_REG_A_COUNT];
            ldg_vec_t ldg_b_reg[LDG_REG_B_COUNT];
#pragma unroll
            for (int i = 0; i < LDG_REG_A_COUNT; i++)
                ldg_a_reg[i] = reinterpret_cast<ldg_vec_t *>(const_cast<scalar_t *>(a) + a_begin
                                                             + ((256 * i + tid) / LDG_A_X_THREADS) * k)[ldg_a_vec_idx];
#pragma unroll
            for (int i = 0; i < LDG_REG_B_COUNT; i++)
                ldg_b_reg[i] = reinterpret_cast<ldg_vec_t *>(const_cast<scalar_t *>(b) + b_begin
                                                             + ((256 * i + tid) / LDG_B_X_THREADS) * n)[ldg_b_vec_idx];

            // transpose to shared local memory
            auto bs_vec = reinterpret_cast<ldg_vec_t *>(bs + write_stage_idx * BLOCK_KN_SIZE);
#pragma unroll
            for (int i = 0; i < LDG_REG_A_COUNT; i++) {
                auto y = (256 * i + tid) / LDG_A_X_THREADS;
#pragma unroll
                for (int j = 0; j < LDG_VEC_SIZE; j++) {
                    as[write_stage_idx * BLOCK_KM_SIZE + (ldg_a_vec_idx * LDG_VEC_SIZE + j) * BLOCK_M + y] = ldg_a_reg[i].val[j];
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
        }

        {
            block_mma<scalar_t, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS,
                      WARP_M_LANES, WARP_N_LANES, WARP_M_THREADS, WARP_N_THREADS, VEC_M, VEC_N>(
                o_reg, as + read_stage_idx * BLOCK_KM_SIZE, bs + read_stage_idx * BLOCK_KN_SIZE,
                wid, w_tid);
        }

        if constexpr (!DOUBLE_BUFFER) {
            __syncthreads();
        }
    }

    { // write back
        using stg_vec_t = aligned_array<scalar_t, VEC_N>;
        auto out_warp_offset_y = block_y * BLOCK_M + wid / BLOCK_N_WARPS * WARP_M;
        auto out_warp_offset_x = block_x * BLOCK_N + wid % BLOCK_N_WARPS * WARP_N;
        constexpr int WARP_LANE_STEP_M = WARP_M / WARP_M_LANES;
        constexpr int WARP_LANE_STEP_N = WARP_N / WARP_N_LANES;
#pragma unroll
        for (int lm = 0; lm < WARP_M_LANES; lm++) {
#pragma unroll
            for (int ln = 0; ln < WARP_N_LANES; ln++) {
                auto out_th_offset_y = out_warp_offset_y + lm * WARP_LANE_STEP_M + w_tid / WARP_N_THREADS * VEC_M;
                auto out_th_offset_x = out_warp_offset_x + ln * WARP_LANE_STEP_N + w_tid % WARP_N_THREADS * VEC_N;
#pragma unroll
                for (int i = 0; i < VEC_M; i++) {
                    auto y = out_th_offset_y + i;
                    if (y < m && out_th_offset_x < n) {
                        auto vec = *reinterpret_cast<stg_vec_t *>(out + y * n + out_th_offset_x);
#pragma unroll
                        for (int j = 0; j < VEC_N; j++) {
                            vec.val[j] = alpha * o_reg[lm * WARP_N_LANES + ln][i * VEC_N + j] + beta * vec.val[j];
                        }
                        *reinterpret_cast<stg_vec_t *>(out + y * n + out_th_offset_x) = vec;
                    }
                }
            }
        }
    }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N, bool DOUBLE_BUFFER>
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

    if constexpr (BLOCK_M == 64 && BLOCK_N == 64) {
        constexpr int BLOCK_K = 32;
        auto slm_size = DOUBLE_BUFFER ? (BLOCK_M + BLOCK_N) * BLOCK_K * 2 : (BLOCK_M + BLOCK_N) * BLOCK_K;
        gemm_cuda_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                         /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_M_LANES*/ 1, /*WARP_N_LANES*/ 1,
                         /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 4,
                         DOUBLE_BUFFER><<<grid, block, slm_size * sizeof(scalar_t)>>>(out, a, b, m, n, k, alpha, beta);
    }
    if constexpr (BLOCK_M == 128 && BLOCK_N == 64) {
        constexpr int BLOCK_K = 16;
        auto slm_size = DOUBLE_BUFFER ? (BLOCK_M + BLOCK_N) * BLOCK_K * 2 : (BLOCK_M + BLOCK_N) * BLOCK_K;
        gemm_cuda_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                         /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_M_LANES*/ 2, /*WARP_N_LANES*/ 2,
                         /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 2,
                         DOUBLE_BUFFER><<<grid, block, slm_size * sizeof(scalar_t)>>>(out, a, b, m, n, k, alpha, beta);
    } else if constexpr (BLOCK_M == 128 && BLOCK_N == 128) {
        constexpr int BLOCK_K = 16;
        auto slm_size = DOUBLE_BUFFER ? (BLOCK_M + BLOCK_N) * BLOCK_K * 2 : (BLOCK_M + BLOCK_N) * BLOCK_K;
        gemm_cuda_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                         /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_M_LANES*/ 2, /*WARP_N_LANES*/ 2,
                         /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 4,
                         DOUBLE_BUFFER><<<grid, block, slm_size * sizeof(scalar_t)>>>(out, a, b, m, n, k, alpha, beta);
    } else if constexpr (BLOCK_M == 256 && BLOCK_N == 128) {
        constexpr int BLOCK_K = 16;
        auto slm_size = DOUBLE_BUFFER ? (BLOCK_M + BLOCK_N) * BLOCK_K * 2 : (BLOCK_M + BLOCK_N) * BLOCK_K;
        gemm_cuda_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                         /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_M_LANES*/ 4, /*WARP_N_LANES*/ 2,
                         /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 4,
                         DOUBLE_BUFFER><<<grid, block, slm_size * sizeof(scalar_t)>>>(out, a, b, m, n, k, alpha, beta);
    }

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
    auto min_size = std::min(m, n);
    if (min_size <= 512) {
        return gemm_cuda_impl<scalar_t, 64, 64, true>(out, a, b, m, n, k, alpha, beta);
    } else if (min_size <= 1024) {
        return gemm_cuda_impl<scalar_t, 128, 64, true>(out, a, b, m, n, k, alpha, beta);
    } else if (min_size <= 4096) {
        return gemm_cuda_impl<scalar_t, 128, 128, true>(out, a, b, m, n, k, alpha, beta);
    } else {
        return gemm_cuda_impl<scalar_t, 256, 128, true>(out, a, b, m, n, k, alpha, beta);
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
    sizes.push_back(gemm_sizes(512, 512, 512, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1024, 1024, 1024, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1028, 1028, 1028, 0.5, 0.5));
    sizes.push_back(gemm_sizes(2048, 2048, 2048, 0.5, 0.5));
    sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
    sizes.push_back(gemm_sizes(8192, 8192, 8192, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1<<14, 1<<14, 1<<14, 0.5, 0.5));

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
