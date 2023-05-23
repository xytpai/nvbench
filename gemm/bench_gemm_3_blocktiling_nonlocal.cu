#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
using namespace std;

void matmul(const float *a, int ah, int aw, const float *b, int bw, float *c, float alpha, float beta) {
    for (int i = 0; i < ah; i++) {
        for (int j = 0; j < bw; j++) {
            float sum = 0;
            for (int k = 0; k < aw; k++)
                sum += a[i * aw + k] * b[k * bw + j];
            c[i * bw + j] = alpha * sum + beta * c[i * bw + j];
        }
    }
}

__global__ void matmul_kernel(
    const float *A, int Ah, int Aw,
    const float *B, int Bw,
    float *C,
    float alpha, float beta) {
    const int BM = 64;
    const int BK = 16;
    auto tx = threadIdx.x % 32;
    auto ty = threadIdx.x / 32;
    // auto x = blockIdx.x * BLOCKSIZE + tx;
    // auto y = blockIdx.y * BLOCKSIZE + ty;

    auto block_y_begin = (blockIdx.y * BM)*Aw;
    auto block_y_end = block_y_begin + Aw;
    auto block_y_step = BK;
    auto block_x_begin = blockIdx.x * BM;
    auto block_x_step = BK * Bw;

    auto innerAy = threadIdx.x / BK;
    auto innerAx = threadIdx.x % BK;
    auto innerBy = threadIdx.x / BM;
    auto innerBx = threadIdx.x % BM;

    float tmp[4] = {0.0};
    for (int a_bg = block_y_begin, b_bg = block_x_begin; a_bg < block_y_end; a_bg += block_y_step, b_bg += block_x_step) {
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BM];
        As[innerAy * BK + innerAx] = A[a_bg + innerAy * Aw + innerAx];
        Bs[innerBy * BM + innerBx] = B[b_bg + innerBy * Bw + innerBx];
        __syncthreads();
#pragma unroll
        for (int m=0; m<2; m++) {
#pragma unroll
            for(int n=0;n<2;n++) {
                auto As_ = As + m * 32 * BK;
                auto Bs_ = Bs + n * 32;
                for (int k = 0; k < BK; ++k) {
                    tmp[m*2+n] += As_[ty * BK + k] * Bs_[k * BM + tx];
                }
            }
        }    
        __syncthreads();
    }
#pragma unroll
    for (int m=0; m<2; m++) {
#pragma unroll
        for(int n=0;n<2;n++) {
            auto x = blockIdx.x * 64 + n * 32 + tx;
            auto y = blockIdx.y * 64 + m * 32 + ty;
            C[y * Bw + x] = alpha * tmp[m*2+n] + beta * C[y * Bw + x];
        }
    }
}

float matmul_cu(const float *a, int ah, int aw, const float *b, int bw, float *c, float alpha, float beta) {
    dim3 threadsPerBlock(32 * 32);
    dim3 numBlocks(ah / 64, bw / 64);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(a, ah, aw, b, bw, c, alpha, beta);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << milliseconds << " ms" << std::endl;
    return milliseconds;
}

int main() {
    const int ah = 1024;
    const int aw = 1024;
    const int bw = 1024;
    const float alpha = 0.5;
    const float beta = 0.5;

    auto ref_a = new float[ah * aw];
    auto ref_b = new float[aw * bw];
    auto ref_c = new float[ah * bw];
    for (int i = 0; i < ah * aw; i++)
        ref_a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (int i = 0; i < aw * bw; i++)
        ref_b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (int i = 0; i < ah * bw; i++)
        ref_c[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    float *a, *b, *c;
    cudaMalloc(&a, ah * aw * sizeof(float));
    cudaMalloc(&b, aw * bw * sizeof(float));
    cudaMalloc(&c, ah * bw * sizeof(float));
    cudaMemcpy(a, ref_a, ah * aw * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, ref_b, aw * bw * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c, ref_c, ah * bw * sizeof(float), cudaMemcpyHostToDevice);

    matmul(ref_a, ah, aw, ref_b, bw, ref_c, alpha, beta);
    auto timems = matmul_cu(a, ah, aw, b, bw, c, alpha, beta);
    float total_GBytes = (ah * aw + aw * bw + ah * bw + ah * bw) * sizeof(float) / 1024.0 / 1024 / 1024;
    std::cout << total_GBytes / (timems/1000.0) << " gbps\n";

    auto out_c = new float[ah * bw];
    cudaMemcpy(out_c, c, ah * bw * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < ah * bw; i++) {
        auto diff = out_c[i] - ref_c[i];
        diff = diff > 0 ? diff : -diff;
        // std::cout<<diff<<"\n";
        if (diff > 0.1)
            return 1;
    }
    std::cout << "ok\n";

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    delete ref_a;
    delete ref_b;
    delete ref_c;
    delete out_c;
    return 0;
}
