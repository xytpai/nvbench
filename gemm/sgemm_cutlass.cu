#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>
#include "cutlass/gemm/device/gemm.h"

template <typename scalar_t>
float gemm_cuda(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;
    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {m, n, k},
        {a, k},
        {b, n},
        {out, n},
        {out, n},
        {alpha, beta}
    );
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemm_op(args);
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
    std::cout << "sgemm_cutlass\n";
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
