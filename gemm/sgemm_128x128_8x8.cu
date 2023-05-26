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

__device__ inline float4 fma_f4(const float alpha, const float4 &a, const float beta, const float4 &b) {
    float4 res;
    res.x = alpha * a.x + beta * b.x;
    res.y = alpha * a.y + beta * b.y;
    res.z = alpha * a.z + beta * b.z;
    res.w = alpha * a.w + beta * b.w;
    return res;
}

template<int BM=128, int BN=128, int BK=16, int TM=4, int TN=4, int SQRT_BSIZE=16>
__global__ void matmul_kernel(
    const float *A, int Ah, int Aw,
    const float *B, int Bw,
    float *C,
    float alpha, float beta) {
    auto lid = threadIdx.x;

    auto block_y_begin = (blockIdx.y * BM)*Aw;
    auto block_y_end = block_y_begin + Aw;
    auto block_y_step = BK;
    auto block_x_begin = blockIdx.x * BN;
    auto block_x_step = BK * Bw;

    constexpr int LDG_REG_A_COUNT = BM * BK / 4 / SQRT_BSIZE / SQRT_BSIZE;
    constexpr int LDG_REG_B_COUNT = BN * BK / 4 / SQRT_BSIZE / SQRT_BSIZE;
    float4 ldg_a_reg[LDG_REG_A_COUNT];
    float4 ldg_b_reg[LDG_REG_B_COUNT];
    constexpr int LDG_A_X_CT = BK / 4;
    constexpr int LDG_B_X_CT = BN / 4;
    auto lid_mod_LDG_A_X_CT = lid % LDG_A_X_CT;
    auto lid_mod_LDG_B_X_CT = lid % LDG_B_X_CT;

    float4 a_reg[LDG_REG_A_COUNT];
    float4 b_reg[LDG_REG_B_COUNT];
    float4 c_reg[LDG_REG_A_COUNT * LDG_REG_B_COUNT][4] = {{make_float4(0.f, 0.f, 0.f, 0.f)}};

    for (int a_begin = block_y_begin, b_begin = block_x_begin; 
        a_begin < block_y_end; a_begin += block_y_step, b_begin += block_x_step) 
    {
        __shared__ float4 As[BK * BM / 4];
        __shared__ float4 Bs[BK * BN / 4];
#pragma unroll
        for(int i=0; i<LDG_REG_A_COUNT; i++) {
            ldg_a_reg[i] = reinterpret_cast<float4*>(const_cast<float*>(A) + a_begin + ((blockDim.x * i + lid) / LDG_A_X_CT) * Aw)[lid_mod_LDG_A_X_CT];
        }
#pragma unroll
        for(int i=0; i<LDG_REG_B_COUNT; i++) {
            ldg_b_reg[i] = reinterpret_cast<float4*>(const_cast<float*>(B) + b_begin + ((blockDim.x * i + lid) / LDG_B_X_CT) * Bw)[lid_mod_LDG_B_X_CT];
        }
#pragma unroll
        for(int i=0; i<LDG_REG_A_COUNT; i++) {
            auto y = (blockDim.x * i + lid) / LDG_A_X_CT;
#pragma unroll
            for(int j=0; j<4; j++) {
                reinterpret_cast<float*>(&As[(lid_mod_LDG_A_X_CT * 4 + j) * (BM/4) + y/4].x)[y%4] = reinterpret_cast<float*>(&ldg_a_reg[i].x)[j];
            }
        }
#pragma unroll
        for(int i=0; i<LDG_REG_B_COUNT; i++) {
            Bs[blockDim.x * i + lid] = ldg_b_reg[i];
        }
        __syncthreads();

#pragma unroll
        for(int k=0; k<BK; k++) {
#pragma unroll
            for(int ia=0; ia<LDG_REG_A_COUNT; ia++) {
                a_reg[ia] = As[k * BM / 4 + lid / 16 + ia * 16];
            }
#pragma unroll
            for(int ib=0; ib<LDG_REG_B_COUNT; ib++) {
                b_reg[ib] = Bs[k * BN / 4 + lid % 16 + ib * 16];
            }

            // mma
#pragma unroll
            for(int ia=0; ia<LDG_REG_A_COUNT; ia++) {
#pragma unroll
                for(int ib=0; ib<LDG_REG_B_COUNT; ib++) {
                    auto fa = reinterpret_cast<float*>(&a_reg[ia]);
                    auto fb = reinterpret_cast<float*>(&b_reg[ib]);
                    auto fc = reinterpret_cast<float*>(&c_reg[ia * LDG_REG_B_COUNT + ib]);
                    #pragma unroll
                    for(int m=0; m<4; m++) {
                        #pragma unroll
                        for(int n=0; n<4; n++) {
                            fc[m*4+n] += fa[m] * fb[n];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // write back
#pragma unroll
    for(int ia=0; ia<LDG_REG_A_COUNT; ia++) {
#pragma unroll
        for(int ib=0; ib<LDG_REG_B_COUNT; ib++) {
            auto y_base = (lid / 16 + ia * 16) * 4;
            auto x = (lid % 16 + ib * 16) * 4;
            for(int i=0; i<4; i++) {
                auto y = y_base + i;
                auto ptr = reinterpret_cast<float4*>(&C[(blockIdx.y * BM + y) * Bw + (blockIdx.x * BN + x)]);
                *ptr = fma_f4(alpha, c_reg[ia * LDG_REG_B_COUNT + ib][i], beta, *ptr);
            }
        }
    }
}

float matmul_cu(const float *a, int ah, int aw, const float *b, int bw, float *c, float alpha, float beta) {
    dim3 threadsPerBlock(16 * 16);
    dim3 numBlocks(ah / 128, bw / 128);
    
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
