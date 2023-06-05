#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>
#include <mma.h>
#include <cuda_fp16.h>

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

template <typename scalar_t,
          int BLOCK_K,
          int BLOCK_M_WARPS, int BLOCK_N_WARPS,
          int WARP_M_LANES, int WARP_N_LANES,
          int WARP_M_THREADS, int WARP_N_THREADS,
          int VEC_M, int VEC_N>
__global__ __launch_bounds__(256) void gemm_cuda_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    static_assert(BLOCK_M_WARPS * BLOCK_N_WARPS == 8);
    static_assert(WARP_M_THREADS * WARP_N_THREADS == 32);
    constexpr int LANE_M = WARP_M_THREADS * VEC_M;
    constexpr int LANE_N = WARP_N_THREADS * VEC_N;
    static_assert(LANE_M == 16);
    static_assert(LANE_N == 16);
    static_assert(BLOCK_K == 32);
    constexpr int WARP_M = WARP_M_LANES * LANE_M;
    constexpr int WARP_N = WARP_N_LANES * LANE_N;
    constexpr int BLOCK_M = BLOCK_M_WARPS * WARP_M;
    constexpr int BLOCK_N = BLOCK_N_WARPS * WARP_N;
    constexpr int BLOCK_KM_SIZE = BLOCK_K * BLOCK_M;
    constexpr int BLOCK_KN_SIZE = BLOCK_K * BLOCK_N;

    // get idx
    auto tid = threadIdx.x;
    auto wid = tid >> 5;
    // auto w_tid = tid & 31;
    auto block_y = blockIdx.y;
    auto block_x = blockIdx.z * gridDim.x + blockIdx.x;

    // get slm
    __shared__ scalar_t as[2][BLOCK_KM_SIZE];
    __shared__ scalar_t bs[2][BLOCK_KN_SIZE];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, scalar_t, nvcuda::wmma::row_major> a_frag[2][WARP_M_LANES];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, scalar_t, nvcuda::wmma::row_major> b_frag[2][WARP_N_LANES];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> o_frag[WARP_M_LANES][WARP_N_LANES];
#pragma unroll
    for (int i = 0; i < WARP_M_LANES; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N_LANES; j++) {
            nvcuda::wmma::fill_fragment(o_frag[i][j], 0.0);
        }
    }

    constexpr int LDG_VEC_SIZE = 8;
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;
    constexpr int LDG_A_X_THREADS = BLOCK_K / LDG_VEC_SIZE;
    constexpr int LDG_B_X_THREADS = BLOCK_N / LDG_VEC_SIZE;
    auto ldg_a_vec_idx = tid % LDG_A_X_THREADS;
    auto ldg_b_vec_idx = tid % LDG_B_X_THREADS;
    constexpr int LDG_REG_A_COUNT = BLOCK_KM_SIZE / LDG_VEC_SIZE / 256;
    constexpr int LDG_REG_B_COUNT = BLOCK_KN_SIZE / LDG_VEC_SIZE / 256;
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);

    int write_stage_idx = 0;
    int read_stage_idx = 1;

    for (
        int a_begin = block_y * BLOCK_M * k, b_begin = block_x * BLOCK_N;
        a_begin < block_y * BLOCK_M * k + k;
        a_begin += BLOCK_K, b_begin += BLOCK_K * n) {
        {
            ldg_vec_t ldg_a_reg[LDG_REG_A_COUNT];
            ldg_vec_t ldg_b_reg[LDG_REG_B_COUNT];
#pragma unroll
            for (int i = 0; i < LDG_REG_A_COUNT; i++) {
                auto idx = 256 * i + tid;
                ldg_a_reg[i] = reinterpret_cast<ldg_vec_t *>(const_cast<scalar_t *>(a) + a_begin + (idx / LDG_A_X_THREADS) * k)[ldg_a_vec_idx];
            }
#pragma unroll
            for (int i = 0; i < LDG_REG_B_COUNT; i++) {
                auto idx = 256 * i + tid;
                ldg_b_reg[i] = reinterpret_cast<ldg_vec_t *>(const_cast<scalar_t *>(b) + b_begin + (idx / LDG_B_X_THREADS) * n)[ldg_b_vec_idx];
            }
            auto as_vec = reinterpret_cast<ldg_vec_t *>(as[write_stage_idx]);
            auto bs_vec = reinterpret_cast<ldg_vec_t *>(bs[write_stage_idx]);
#pragma unroll
            for (int i = 0; i < LDG_REG_A_COUNT; i++) {
                as_vec[256 * i + tid] = ldg_a_reg[i];
            }
#pragma unroll
            for (int i = 0; i < LDG_REG_A_COUNT; i++) {
                bs_vec[256 * i + tid] = ldg_b_reg[i];
            }
            read_stage_idx ^= 1;
            write_stage_idx ^= 1;
            __syncthreads();
        }

        {
            auto a_ptr = as[read_stage_idx];
            auto b_ptr = bs[read_stage_idx];
            auto warp_y = wid / BLOCK_N_WARPS * WARP_M;
            auto warp_x = wid % BLOCK_N_WARPS * WARP_N;

#pragma unroll
            for (int i = 0; i < WARP_M_LANES; i++) {
                int lane_y = warp_y + i * LANE_M;
                nvcuda::wmma::load_matrix_sync(a_frag[0][i], a_ptr + lane_y * BLOCK_K, BLOCK_K);
                nvcuda::wmma::load_matrix_sync(a_frag[1][i], a_ptr + lane_y * BLOCK_K + 16, BLOCK_K);
            }
#pragma unroll
            for (int j = 0; j < WARP_N_LANES; j++) {
                int lane_x = warp_x + j * LANE_N;
                nvcuda::wmma::load_matrix_sync(b_frag[0][j], b_ptr + lane_x, BLOCK_N);
                nvcuda::wmma::load_matrix_sync(b_frag[1][j], b_ptr + lane_x + 16 * BLOCK_N, BLOCK_N);
            }
#pragma unroll
            for (int i = 0; i < WARP_M_LANES; i++) {
#pragma unroll
                for (int j = 0; j < WARP_N_LANES; j++) {
                    nvcuda::wmma::mma_sync(o_frag[i][j], a_frag[0][i], b_frag[0][j], o_frag[i][j]);
                    nvcuda::wmma::mma_sync(o_frag[i][j], a_frag[1][i], b_frag[1][j], o_frag[i][j]);
                }
            }
        }
    }

    { // write back
        auto out_warp_y = block_y * BLOCK_M + wid / BLOCK_N_WARPS * WARP_M;
        auto out_warp_x = block_x * BLOCK_N + wid % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (int i = 0; i < WARP_M_LANES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_N_LANES; j++) {
                auto out_lane_y = out_warp_y + i * LANE_M;
                auto out_lane_x = out_warp_x + j * LANE_N;
                auto out_offset = out_lane_y * n + out_lane_x;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, scalar_t> c_frag;
                nvcuda::wmma::load_matrix_sync(c_frag, out + out_offset, n, nvcuda::wmma::mem_row_major);
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    c_frag.x[k] = alpha * (scalar_t)o_frag[i][j].x[k] + beta * c_frag.x[k];
                }
                nvcuda::wmma::store_matrix_sync(out + out_offset, c_frag, n, nvcuda::wmma::mem_row_major);
            }
        }
    }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
float gemm_cuda_impl(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    assert(k % 4 == 0);
    assert(n % 4 == 0);
    int m_blocks = (m + BLOCK_M - 1) / BLOCK_M;
    int n_blocks = (n + BLOCK_N - 1) / BLOCK_N;
    int split_num = (n_blocks + 4096 - 1) / 4096;
    dim3 block(256);
    dim3 grid((n_blocks + split_num - 1) / split_num, m_blocks, split_num);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if constexpr (BLOCK_M == 256 && BLOCK_N == 128) {
        constexpr int BLOCK_K = 32;
        gemm_cuda_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                         /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_M_LANES*/ 4, /*WARP_N_LANES*/ 4,
                         /*WARP_M_THREADS*/ 8, /*WARP_N_THREADS*/ 4, /*VEC_M*/ 2, /*VEC_N*/ 4><<<grid, block>>>(out, a, b, m, n, k, alpha, beta);
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
    // if (min_size <= 512) {
    //     return gemm_cuda_impl<scalar_t, 64, 64, true>(out, a, b, m, n, k, alpha, beta);
    // } else if (min_size <= 1024) {
    //     return gemm_cuda_impl<scalar_t, 128, 64, true>(out, a, b, m, n, k, alpha, beta);
    // } else if (min_size <= 4096) {
    //     return gemm_cuda_impl<scalar_t, 128, 128, true>(out, a, b, m, n, k, alpha, beta);
    // } else {
    //     return gemm_cuda_impl<scalar_t, 256, 128, true>(out, a, b, m, n, k, alpha, beta);
    // }
    return gemm_cuda_impl<scalar_t, 256, 128>(out, a, b, m, n, k, alpha, beta);
}

template <typename scalar_t>
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
            acc += (float)a[mi * k + ki] * (float)b[ki * n + ni];
        }
        auto r = (float)alpha * acc;
        out[mi * n + ni] = r + (float)beta * (float)out[mi * n + ni];
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
    std::cout << "gemm_wmma_fp16\n";
    using scalar_t = half;

    std::vector<gemm_sizes> sizes;
    // sizes.push_back(gemm_sizes(512, 512, 512, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1024, 1024, 1024, 0.5, 0.5));
    // sizes.push_back(gemm_sizes(1028, 1028, 1028, 0.5, 0.5));
    sizes.push_back(gemm_sizes(2048, 2048, 2048, 0.5, 0.5));
    sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
    sizes.push_back(gemm_sizes(8192, 8192, 8192, 0.5, 0.5));
    // sizes.push_back(gemm_sizes(1<<14, 1<<14, 1<<14, 0.5, 0.5));

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
            a_cpu[i] = static_cast<scalar_t>((float)rand() / RAND_MAX);
        for (int i = 0; i < k * n; i++)
            b_cpu[i] = static_cast<scalar_t>((float)rand() / RAND_MAX);
        for (int i = 0; i < m * n; i++)
            out_cpu[i] = static_cast<scalar_t>((float)rand() / RAND_MAX);

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
        auto maxdiff = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < m * n; i++) {
            // if (i < 10)

            auto diff = std::abs((float)out_cuda_[i] - (float)out_cuda_ref_[i]);
            // if (i < 256+256+1)
            // std::cout << (float)out_cuda_[i] << " " << (float)out_cuda_ref_[i] << " " << diff << "\n";
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
        delete[] out_cuda_ref_;
    }
    return 0;
}