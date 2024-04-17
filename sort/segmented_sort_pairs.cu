#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include <vector>
#include "block_radix_sort.h"
#include "utils.h"

template <typename method_t, typename key_t, typename value_t>
__global__ void segmented_sort_pairs_cuda_kernel(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int num_elements, key_t padding_key) {
    int b = blockIdx.x;
    int b_offset = b * num_elements;
    int lid = threadIdx.x;
    extern __shared__ char shared[];
    auto method = method_t(lid, shared);

    key_t keys[method_t::REG_LEN];
    value_t values[method_t::REG_LEN];

#pragma unroll
    for (int ITEM = 0; ITEM < method_t::KEYS_PER_THREAD; ++ITEM) {
        int offset = lid + method_t::BLOCK_THREADS * ITEM;
        if (offset < num_elements) {
            keys[ITEM] = keys_in[b_offset + offset];
            values[ITEM] = values_in[b_offset + offset];
        } else {
            keys[ITEM] = padding_key;
        }
    }

    method.sort_blocked(keys, values, 0, sizeof(key_t) * 8);

#pragma unroll
    for (int ITEM = 0; ITEM < method_t::KEYS_PER_THREAD; ++ITEM) {
        int offset = lid + method_t::BLOCK_THREADS * ITEM;
        if (offset < num_elements) {
            keys_out[b_offset + offset] = keys[ITEM];
            values_out[b_offset + offset] = values[ITEM];
        }
    }
}

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_sort_pairs_cuda(const key_t *keys_in, key_t *keys_out,
                               const value_t *values_in, value_t *values_out,
                               int num_segments, int num_elements) {
    dim3 block(BLOCK_THREADS);
    dim3 grid(num_segments);
    using method_t = sort_impl::BlockRadixSort<key_t, BLOCK_THREADS, KEYS_PER_ITEM, IS_DESCENDING, value_t>;
    auto padding_key = method_t::IS_DESCENDING ? std::numeric_limits<key_t>::lowest() : std::numeric_limits<key_t>::max();
    segmented_sort_pairs_cuda_kernel<method_t, key_t, value_t><<<grid, block, method_t::LocalMemorySize()>>>(keys_in, keys_out, values_in, values_out, num_elements, padding_key);
}

template <typename key_t, typename value_t, bool IS_DESCENDING>
inline void segmented_sort_pairs_(const key_t *keys_in, key_t *keys_out,
                                  const value_t *values_in, value_t *values_out,
                                  int num_segments, int num_elements) {
    if (num_elements > 4096) {
        std::cout << "num_elements should shorter than 4096\n";
    } else if (num_elements > 2048) {
        segmented_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 1024>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 1024) {
        segmented_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 512>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 512) {
        segmented_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 256>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 256) {
        segmented_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 128>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else {
        segmented_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 64>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    }
}

template <typename key_t, typename value_t>
void segmented_sort_pairs(const key_t *keys_in, key_t *keys_out,
                          const value_t *values_in, value_t *values_out,
                          int num_segments, int num_elements, bool descending) {
    if (descending)
        segmented_sort_pairs_<key_t, value_t, true>(
            keys_in, keys_out, values_in, values_out,
            num_segments, num_elements);
    else
        segmented_sort_pairs_<key_t, value_t, false>(
            keys_in, keys_out, values_in, values_out,
            num_segments, num_elements);
}

int main() {
    float max_gbps = 0;
    for (int it = 0; it < 40; it++) {
        using key_t = float;
        using value_t = int;
        int num_segments = randint_scalar(5, 60);
        if (it == 20) num_segments = 3333;
        int num_elements = randint_scalar(10, 4096);
        if (it == 10) num_elements = 4096;
        bool is_descending = randint_scalar(0, 2) > 0;

        std::cout << "testing sort pairs num_segments[" << num_segments
                  << "] num_elements[" << num_elements << "] is_descending[" << is_descending << "]\n";
        int total_size = num_segments * num_elements;
        auto key = new key_t[total_size];
        auto value = new value_t[total_size];
        auto key_out = new key_t[total_size];
        auto value_out = new value_t[total_size];
        fill_rand<key_t>(key, total_size, -10000.0, 10000.0);
        for (int i = 0; i < num_segments; i++) {
            for (int j = 0; j < num_elements; j++) {
                value[i * num_elements + j] = j;
            }
        }

        key_t *key_dev;
        value_t *value_dev;

        cudaMalloc(&key_dev, total_size * sizeof(key_t));
        cudaMalloc(&value_dev, total_size * sizeof(value_t));
        cudaMemcpy(key_dev, key, total_size * sizeof(key_t), cudaMemcpyHostToDevice);
        cudaMemcpy(value_dev, value, total_size * sizeof(value_t), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        segmented_sort_pairs(key_dev, key_dev, value_dev, value_dev, num_segments, num_elements, is_descending);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float total_GBytes = (total_size + total_size) * (sizeof(key_t) + sizeof(value_t)) / 1000.0 / 1000.0;
        auto gbps = total_GBytes / (milliseconds);
        max_gbps = std::max(gbps, max_gbps);
        std::cout << gbps << " GBPS\n";

        cudaMemcpy(key_out, key_dev, total_size * sizeof(key_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(value_out, value_dev, total_size * sizeof(value_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_segments; i++) {
            auto key_begin = key + i * num_elements;
            auto value_begin = value + i * num_elements;
            using pair_t = std::pair<key_t, value_t>;
            std::vector<pair_t> v;
            for (int j = 0; j < num_elements; j++)
                v.push_back(std::make_pair(key_begin[j], value_begin[j]));
            if (is_descending)
                std::stable_sort(v.begin(), v.end(), [](pair_t a, pair_t b) { return a.first > b.first; });
            else
                std::stable_sort(v.begin(), v.end(), [](pair_t a, pair_t b) { return a.first < b.first; });
            for (int j = 0; j < num_elements; j++) {
                key_begin[j] = v[j].first;
                value_begin[j] = v[j].second;
            }
        }
        std::cout << "testing key...\n";
        if (!all_close(key_out, key, total_size))
            return 1;
        std::cout << "testing value...\n";
        if (!all_close(value_out, value, total_size))
            return 1;
        delete[] key;
        delete[] value;
        delete[] key_out;
        delete[] value_out;
        cudaFree(key_dev);
        cudaFree(value_dev);
    }

    std::cout << "ok, max_gbps:" << max_gbps << "\n";
    return 0;
}
