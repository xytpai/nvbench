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

template <typename scalar_t, int size>
struct array {
    scalar_t val[size];
};

template <typename scalar_t, int dim0, int dim1>
struct array2d {
    scalar_t val[dim0][dim1];
};

template <typename scalar_t, typename acc_t>
struct mma_16x8x16 {
    using FragmentA = aligned_array<scalar_t, 8>;
    using FragmentB = aligned_array<scalar_t, 4>;
    using FragmentC = aligned_array<acc_t, 4>;
    __forceinline__ __device__ void operator()(
        FragmentC &d,
        FragmentA const &a,
        FragmentB const &b,
        FragmentC const &c) const {
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
        float const *C = reinterpret_cast<float const *>(&c);
        float *D = reinterpret_cast<float *>(&d);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%10,%11,%12,%13};\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    }
};

template <typename scalar_t, typename acc_t, int M, int N, int K = 16>
struct mma_worker {
    static_assert(M % 16 == 0);
    static_assert(N % 8 == 0);
    static_assert(K == 16);
    enum {
        A_LANES = M / 16,
        B_LANES = N / 8,
    };
    using FragmentA = array<aligned_array<scalar_t, 8>, A_LANES>;
    using FragmentB = array<aligned_array<scalar_t, 4>, B_LANES>;
    using FragmentC = array2d<aligned_array<acc_t, 4>, A_LANES, B_LANES>;

    __forceinline__ __device__ void fill_fragment_c(FragmentC &c, acc_t val) {
#pragma unroll
        for (int i = 0; i < A_LANES; i++) {
#pragma unroll
            for (int j = 0; j < B_LANES; j++) {
                c.val[i][j].val[0] = val;
                c.val[i][j].val[1] = val;
                c.val[i][j].val[2] = val;
                c.val[i][j].val[3] = val;
            }
        }
    }

    __forceinline__ __device__ void load_matrix_a(FragmentA &a, scalar_t *ptr, int stride, int w_tid) {
        auto x = w_tid % 4 * 2;
#pragma unroll
        for (int i = 0; i < A_LANES; i++) {
            auto y = w_tid / 4 * A_LANES + i;
            a.val[i].val[0] = ptr[y * stride + x + 0];
            a.val[i].val[1] = ptr[y * stride + x + 1];
            a.val[i].val[2] = ptr[(y + 8 * A_LANES) * stride + x + 0];
            a.val[i].val[3] = ptr[(y + 8 * A_LANES) * stride + x + 1];
            a.val[i].val[4] = ptr[y * stride + 8 + x + 0];
            a.val[i].val[5] = ptr[y * stride + 8 + x + 1];
            a.val[i].val[6] = ptr[(y + 8 * A_LANES) * stride + 8 + x + 0];
            a.val[i].val[7] = ptr[(y + 8 * A_LANES) * stride + 8 + x + 1];
        }
    }

    __forceinline__ __device__ void load_matrix_b(FragmentB &b, scalar_t *ptr, int stride, int w_tid) {
        auto y = w_tid % 4 * 2;
#pragma unroll
        for (int i = 0; i < B_LANES; i++) {
            auto x = w_tid / 4 * B_LANES + i;
            b.val[i].val[0] = ptr[y * stride + x];
            b.val[i].val[1] = ptr[(y + 1) * stride + x];
            b.val[i].val[2] = ptr[(8 + y) * stride + x];
            b.val[i].val[3] = ptr[(8 + y + 1) * stride + x];
        }
    }

    __forceinline__ __device__ void operator()(
        FragmentC &d,
        FragmentA const &a,
        FragmentB const &b,
        FragmentC const &c) const {
        mma_16x8x16<scalar_t, acc_t> mma;
#pragma unroll
        for (int i = 0; i < A_LANES; i++) {
#pragma unroll
            for (int j = 0; j < B_LANES; j++) {
                mma(d.val[i][j], a.val[i], b.val[j], c.val[i][j]);
            }
        }
    }

    __forceinline__ __device__ void store_matrix(scalar_t *ptr, FragmentC &c, int stride, int w_tid, scalar_t alpha, scalar_t beta) {
        auto y = w_tid / 4 * A_LANES;
        auto x = w_tid % 4 * 2 * B_LANES;
        using vec_t = aligned_array<scalar_t, B_LANES * 2>;
#pragma unroll
        for (int i = 0; i < A_LANES; i++) {
            auto vec0 = *reinterpret_cast<vec_t *>(&ptr[(y + i) * stride + x]);
            auto vec1 = *reinterpret_cast<vec_t *>(&ptr[(y + i + 8 * A_LANES) * stride + x]);
#pragma unroll
            for (int j = 0; j < B_LANES; j++) {
                vec0.val[j * 2] = alpha * (scalar_t)c.val[i][j].val[0] + beta * vec0.val[j * 2];
                vec0.val[j * 2 + 1] = alpha * (scalar_t)c.val[i][j].val[1] + beta * vec0.val[j * 2 + 1];
                vec1.val[j * 2] = alpha * (scalar_t)c.val[i][j].val[2] + beta * vec1.val[j * 2];
                vec1.val[j * 2 + 1] = alpha * (scalar_t)c.val[i][j].val[3] + beta * vec1.val[j * 2 + 1];
            }
            *reinterpret_cast<vec_t *>(&ptr[(y + i) * stride + x]) = vec0;
            *reinterpret_cast<vec_t *>(&ptr[(y + i + 8 * A_LANES) * stride + x]) = vec1;
        }
    }
};

template <typename scalar_t,
          int BLOCK_M_LANES, int BLOCK_N_LANES,
          int LANE_M_WARPS, int LANE_N_WARPS,
          int WARP_M_THREADS, int WARP_N_THREADS,
          int VEC_M, int VEC_N,
          int PAD = 8>
__global__ __launch_bounds__(256) void gemm_cuda_kernel(
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ a,
    const scalar_t *__restrict__ b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    constexpr int BLOCK_K = 32;
    static_assert(LANE_M_WARPS * LANE_N_WARPS == 8);
    static_assert(WARP_M_THREADS * WARP_N_THREADS == 32);
    constexpr int WARP_M = WARP_M_THREADS * VEC_M;
    constexpr int WARP_N = WARP_N_THREADS * VEC_N;
    static_assert(WARP_M % 16 == 0);
    static_assert(WARP_N % 8 == 0);
    constexpr int LANE_M = LANE_M_WARPS * WARP_M;
    constexpr int LANE_N = LANE_N_WARPS * WARP_N;
    constexpr int BLOCK_M = BLOCK_M_LANES * LANE_M;
    constexpr int BLOCK_N = BLOCK_N_LANES * LANE_N;

    // idx
    auto tid = threadIdx.x;
    auto wid = tid >> 5;
    auto w_tid = tid & 31;
    auto block_y = blockIdx.y;
    auto block_x = blockIdx.z * gridDim.x + blockIdx.x;

    // slm
    __shared__ scalar_t as[2][BLOCK_M * (BLOCK_K + PAD)];
    __shared__ scalar_t bs[2][BLOCK_K * (BLOCK_N + PAD)];

    using mma_t = mma_worker<scalar_t, float, WARP_M, WARP_N>;
    mma_t mma;
    typename mma_t::FragmentA a_frag[2][BLOCK_M_LANES];
    typename mma_t::FragmentB b_frag[2][BLOCK_N_LANES];
    typename mma_t::FragmentC o_frag[BLOCK_M_LANES][BLOCK_N_LANES];

#pragma unroll
    for (int i = 0; i < BLOCK_M_LANES; i++) {
#pragma unroll
        for (int j = 0; j < BLOCK_N_LANES; j++) {
            mma.fill_fragment_c(o_frag[i][j], 0.0);
        }
    }

    constexpr int LDG_VEC_SIZE = 8;
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;
    constexpr int LDG_A_X_THREADS = BLOCK_K / LDG_VEC_SIZE;
    constexpr int LDG_B_X_THREADS = BLOCK_N / LDG_VEC_SIZE;
    auto ldg_a_vec_idx = tid % LDG_A_X_THREADS;
    auto ldg_b_vec_idx = tid % LDG_B_X_THREADS;
    constexpr int LDG_REG_A_COUNT = BLOCK_M * BLOCK_K / LDG_VEC_SIZE / 256;
    constexpr int LDG_REG_B_COUNT = BLOCK_K * BLOCK_N / LDG_VEC_SIZE / 256;
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
                int y = (256 * i + tid) / LDG_A_X_THREADS;
                as_vec[y * ((BLOCK_K + PAD) / LDG_VEC_SIZE) + ldg_a_vec_idx] = ldg_a_reg[i];
            }
#pragma unroll
            for (int i = 0; i < LDG_REG_B_COUNT; i++) {
                int y = (256 * i + tid) / LDG_B_X_THREADS;
                bs_vec[y * ((BLOCK_N + PAD) / LDG_VEC_SIZE) + ldg_b_vec_idx] = ldg_b_reg[i];
            }
            read_stage_idx ^= 1;
            write_stage_idx ^= 1;
            __syncthreads();
        }

        {
            auto a_ptr = as[read_stage_idx];
            auto b_ptr = bs[read_stage_idx];
            auto warp_y = wid / LANE_N_WARPS * WARP_M;
            auto warp_x = wid % LANE_N_WARPS * WARP_N;

#pragma unroll
            for (int i = 0; i < BLOCK_M_LANES; i++) {
                auto y = i * LANE_M + warp_y;
                mma.load_matrix_a(a_frag[0][i], a_ptr + y * (BLOCK_K + PAD), BLOCK_K + PAD, w_tid);
                mma.load_matrix_a(a_frag[1][i], a_ptr + y * (BLOCK_K + PAD) + 16, BLOCK_K + PAD, w_tid);
            }
#pragma unroll
            for (int j = 0; j < BLOCK_N_LANES; j++) {
                auto x = j * LANE_N + warp_x;
                mma.load_matrix_b(b_frag[0][j], b_ptr + x, BLOCK_N + PAD, w_tid);
                mma.load_matrix_b(b_frag[1][j], b_ptr + x + 16 * (BLOCK_N + PAD), BLOCK_N + PAD, w_tid);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_M_LANES; i++) {
#pragma unroll
                for (int j = 0; j < BLOCK_N_LANES; j++) {
                    mma(o_frag[i][j], a_frag[0][i], b_frag[0][j], o_frag[i][j]);
                    mma(o_frag[i][j], a_frag[1][i], b_frag[1][j], o_frag[i][j]);
                }
            }
        }
    }

    { // write back
        auto out_warp_y = block_y * BLOCK_M + wid / LANE_N_WARPS * WARP_M;
        auto out_warp_x = block_x * BLOCK_N + wid % LANE_N_WARPS * WARP_N;
#pragma unroll
        for (int i = 0; i < BLOCK_M_LANES; i++) {
#pragma unroll
            for (int j = 0; j < BLOCK_N_LANES; j++) {
                auto y = out_warp_y + i * LANE_M;
                auto x = out_warp_x + j * LANE_N;
                if (y < m && x < n) {
                    auto out_offset = y * n + x;
                    mma.store_matrix(out + out_offset, o_frag[i][j], n, w_tid, alpha, beta);
                    __syncthreads();
                }
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
    assert(m % 16 == 0);
    assert(n % 16 == 0);
    assert(k % 32 == 0);
    int m_blocks = (m + BLOCK_M - 1) / BLOCK_M;
    int n_blocks = (n + BLOCK_N - 1) / BLOCK_N;
    int split_num = (n_blocks + 4096 - 1) / 4096;
    dim3 block(256);
    dim3 grid((n_blocks + split_num - 1) / split_num, m_blocks, split_num);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemm_cuda_kernel<scalar_t, /*BLOCK_M_LANES*/ BLOCK_M / 32, /*BLOCK_N_LANES*/ BLOCK_N / 64, /*LANE_M_WARPS*/ 2, /*LANE_N_WARPS*/ 4,
                     /*WARP_M_THREADS*/ 8, /*WARP_N_THREADS*/ 4, /*VEC_M*/ 2, /*VEC_N*/ 4><<<grid, block>>>(out, a, b, m, n, k, alpha, beta);

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
    // auto min_size = std::min(m, n);
    // if (min_size <= 1024) {
    //     return gemm_cuda_impl<scalar_t, 64, 64>(out, a, b, m, n, k, alpha, beta);
    // } else if (min_size <= 2048) {
    //     return gemm_cuda_impl<scalar_t, 64, 128>(out, a, b, m, n, k, alpha, beta);
    // } else if (min_size <= 4096) {
    //     return gemm_cuda_impl<scalar_t, 64, 128>(out, a, b, m, n, k, alpha, beta);
    // } else {
    //     return gemm_cuda_impl<scalar_t, 128, 128>(out, a, b, m, n, k, alpha, beta);
    // }
    return gemm_cuda_impl<scalar_t, 128, 128>(out, a, b, m, n, k, alpha, beta);
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
    std::cout << "hgemm_mma\n";
    using scalar_t = half;

    std::vector<gemm_sizes> sizes;
    sizes.push_back(gemm_sizes(512, 512, 512, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1024, 1024, 1024, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1024 + 16, 1024 + 16, 64, 0.5, 0.5));
    sizes.push_back(gemm_sizes(2048, 2048, 2048, 0.5, 0.5));
    sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
    sizes.push_back(gemm_sizes(8192, 8192, 8192, 0.5, 0.5));
    // sizes.push_back(gemm_sizes(1 << 14, 1 << 14, 1 << 14, 0.5, 0.5));

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
