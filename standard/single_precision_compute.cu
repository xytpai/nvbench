#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
using namespace std;

template <int LOOP>
__global__ void fmad_loop_kernel(float *x) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float a = x[index], b = -1.0f;
    for (int i = 0; i < LOOP; i++) {
        for (int j = 0; j < LOOP; j++) {
            a = a * b + b;
        }
    }
    x[index] = a;
}

template <int LOOP, int block_size, int num_blocks>
float fmad_test() {
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(num_blocks);
    constexpr int n = block_size * num_blocks;
    auto x = new float[n];
    float *dx;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fmad_loop_kernel<LOOP><<<numBlocks, threadsPerBlock>>>(dx);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(x, dx, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dx);
    delete[] x;
    return ms;
}

int main() {
    constexpr int LOOP = 10000;
    constexpr int block_size = 256;
    constexpr int num_blocks = 2048;
    for (int i = 0; i < 3; i++) {
        auto timems = fmad_test<LOOP, block_size, num_blocks>();
        auto tflops =
            2.0 * LOOP * LOOP * num_blocks * block_size / (timems / 1000) * 1e-12;
        auto arithmetic_intensity = 2.0f * LOOP * LOOP / (sizeof(float) * 2);
        std::cout << "arithmetic_intensity: " << arithmetic_intensity << " FLOP/Byte.  |  COMPUTE:";
        std::cout << tflops << " TFLOPS" << std::endl;
    }
    return 0;
}
