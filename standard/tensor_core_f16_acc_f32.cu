#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <mma.h>
#include <cuda_fp16.h>
using namespace std;

template <typename scalar_t, int LOOP>
__global__ void wmma_loop_kernel(scalar_t *input, float *output, int stride_in_elem) {
    // scalar_t: __half or __bfloat16
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = index / 32;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, scalar_t, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, scalar_t, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> o_frag;
    nvcuda::wmma::fill_fragment(o_frag, 0.0);
    nvcuda::wmma::load_matrix_sync(a_frag, input + wid * 16, stride_in_elem);
    nvcuda::wmma::load_matrix_sync(b_frag, input + wid * 16, stride_in_elem);
    for (int i = 0; i < LOOP; i++) {
        nvcuda::wmma::mma_sync(o_frag, a_frag, b_frag, o_frag);
    }
    nvcuda::wmma::store_matrix_sync(output + wid * 16, o_frag, stride_in_elem, nvcuda::wmma::mem_row_major);
}

template <int LOOP, int num_blocks>
float wmma_test() {
    dim3 threadsPerBlock(256);
    dim3 numBlocks(num_blocks);
    constexpr int n = 16 * 256 * num_blocks;
    auto input = new __half[n];
    auto output = new float[n];
    __half *dinput;
    float *doutput;
    cudaMalloc(&dinput, n * sizeof(__half));
    cudaMemcpy(dinput, input, n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMalloc(&doutput, n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    wmma_loop_kernel<__half, LOOP><<<numBlocks, threadsPerBlock>>>(dinput, doutput, 256 * num_blocks);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(output, doutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
    delete[] input;
    delete[] output;
    return ms;
}

int main() {
    constexpr int LOOP = 1000000;
    constexpr int num_blocks = 4096;
    constexpr int warps_per_block = 256 / 32;
    for (int i = 0; i < 3; i++) {
        auto timems = wmma_test<LOOP, num_blocks>();
        auto tflops =
            ((double)2 * 16 * 16 * 16) * LOOP * num_blocks * warps_per_block / (timems / 1000) * 1e-12;
        std::cout << tflops << " TFLOPS" << std::endl;
    }
    return 0;
}
