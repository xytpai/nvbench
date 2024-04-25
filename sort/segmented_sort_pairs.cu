#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include "sorting_common.h"
#include "sorting_radixsort.h"
#include "utils.h"

// =================== block sort ===================

template <typename method_t, typename key_t, typename value_t>
__global__ void segmented_block_sort_pairs_cuda_kernel(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int num_elements) {
    int seg_idx = blockIdx.x;
    int seg_offset = seg_idx * num_elements;
    int lid = threadIdx.x;
    extern __shared__ char shared[];
    auto method = method_t(lid, shared);
    method.load_keys(keys_in + seg_offset, num_elements);
    method.load_values(values_in + seg_offset, num_elements);
    int begin_bit = 0;
    int end_bit = 8 * sizeof(key_t);
    while (true) {
        method.rank_keys(begin_bit, end_bit);
        method.exchange_keys();
        method.exchange_values();
        begin_bit += method_t::RADIX_BITS;
        if (begin_bit >= end_bit) break;
    }
    method.store_keys(keys_out + seg_offset, num_elements);
    method.store_values(values_out + seg_offset, num_elements);
}

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_block_sort_pairs_cuda(const key_t *keys_in, key_t *keys_out,
                                     const value_t *values_in, value_t *values_out,
                                     int num_segments, int num_elements) {
    dim3 block(BLOCK_THREADS);
    dim3 grid(num_segments);
    using method_t = sorting_impl::BlockRadixSort<key_t, BLOCK_THREADS, KEYS_PER_ITEM, IS_DESCENDING, value_t>;
    segmented_block_sort_pairs_cuda_kernel<method_t, key_t, value_t><<<grid, block, method_t::LocalMemorySize()>>>(keys_in, keys_out, values_in, values_out, num_elements);
}

// =================== upsweep ===================

template <typename method_t, typename key_t, typename value_t>
__global__ void segmented_tile_sort_pairs_upsweep_kernel(
    const key_t *keys_in, int *counts,
    int num_elements,
    int begin_bit, int end_bit) {
    int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
    int seg_idx = blockIdx.x / num_tiles;
    int tile_idx = blockIdx.x % num_tiles;
    int lid = threadIdx.x;
    auto keys_in_seg = keys_in + seg_idx * num_elements;
    auto counts_seg = counts + seg_idx * method_t::RADIX_BUCKETS * num_tiles;
    int tile_offset = tile_idx * method_t::PROCESSING_LENGTH;
    int tile_end = tile_offset + method_t::PROCESSING_LENGTH;
    tile_end = tile_end > num_elements ? num_elements : tile_end;
    extern __shared__ char shared[];
    auto method = method_t(keys_in_seg, lid, tile_idx, begin_bit, end_bit, num_tiles, counts_seg, shared);
    method.run(tile_offset, tile_end);
}

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_tile_sort_pairs_upsweep_cuda(const key_t *keys_in, int *counts,
                                            int num_segments, int num_elements, int begin_bit, int end_bit) {
    using method_t = sorting_impl::RadixSortUpsweep<key_t, BLOCK_THREADS, KEYS_PER_ITEM, IS_DESCENDING, value_t>;
    int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
    dim3 block(BLOCK_THREADS);
    dim3 grid(num_segments * num_tiles);
    segmented_tile_sort_pairs_upsweep_kernel<method_t, key_t, value_t><<<grid, block, method_t::LocalMemorySize()>>>(
        keys_in, counts, num_elements, begin_bit, end_bit);
}

// =================== scan bins ===================

template <typename method_t, int RADIX_BUCKETS>
__global__ void segmented_tile_sort_pairs_scan_kernel(
    int *counts, int num_tiles) {
    int seg_idx = blockIdx.x;
    int lid = threadIdx.x;
    extern __shared__ char shared[];
    auto counts_seg = counts + seg_idx * RADIX_BUCKETS * num_tiles;
    auto method = method_t(counts_seg, reinterpret_cast<int *>(shared), lid);
    method.run(num_tiles * RADIX_BUCKETS);
}

template <int KEYS_PER_ITEM, int BLOCK_THREADS, int RADIX_BUCKETS>
void segmented_tile_sort_pairs_scan_cuda(int *counts, int num_tiles, int num_segments) {
    using method_t = sorting_impl::RadixSortScanBins<BLOCK_THREADS, KEYS_PER_ITEM>;
    dim3 block(BLOCK_THREADS);
    dim3 grid(num_segments);
    segmented_tile_sort_pairs_scan_kernel<method_t, RADIX_BUCKETS><<<grid, block, method_t::LocalMemorySize()>>>(counts, num_tiles);
}

// =================== downsweep ===================

template <typename method_t, typename key_t, typename value_t>
__global__ void segmented_tile_sort_pairs_downsweep_kernel(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int num_elements,
    int begin_bit, int end_bit, int *counts) {
    int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
    int seg_idx = blockIdx.x / num_tiles;
    int tile_idx = blockIdx.x % num_tiles;
    int seg_offset = seg_idx * num_elements;
    int lid = threadIdx.x;
    int tile_offset = tile_idx * method_t::PROCESSING_LENGTH;
    auto counts_seg = counts + seg_idx * method_t::RADIX_BUCKETS * num_tiles;
    extern __shared__ char shared[];
    auto method = method_t(lid, shared);
    method.load_keys(keys_in + seg_offset, num_elements, tile_offset);
    method.load_values(values_in + seg_offset, num_elements, tile_offset);
    method.load_bin_offsets(counts_seg, tile_idx, num_tiles);
    method.rank_keys(begin_bit, end_bit);
    method.exchange_and_store_keys(keys_out + seg_offset, num_elements);
    method.exchange_and_store_values(values_out + seg_offset, num_elements);
}

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_tile_sort_pairs_downsweep_cuda(const key_t *keys_in, key_t *keys_out,
                                              const value_t *values_in, value_t *values_out,
                                              int num_segments, int num_elements, int begin_bit, int end_bit, int *counts) {
    using method_t = sorting_impl::BlockRadixSort<key_t, BLOCK_THREADS, KEYS_PER_ITEM, IS_DESCENDING, value_t>;
    int num_tiles = (num_elements + method_t::PROCESSING_LENGTH - 1) / method_t::PROCESSING_LENGTH;
    dim3 block(BLOCK_THREADS);
    dim3 grid(num_segments * num_tiles);
    segmented_tile_sort_pairs_downsweep_kernel<method_t, key_t, value_t><<<grid, block, method_t::LocalMemorySize()>>>(
        keys_in, keys_out, values_in, values_out, num_elements, begin_bit, end_bit, counts);
}

// ==================== tile sort ===================

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_tile_sort_pairs_cuda(const key_t *keys_in, key_t *keys_out,
                                    const value_t *values_in, value_t *values_out,
                                    int num_segments, int num_elements) {
    constexpr int TILE_PROCESSING_LENGTH = BLOCK_THREADS * KEYS_PER_ITEM;
    int num_tiles = (num_elements + TILE_PROCESSING_LENGTH - 1) / TILE_PROCESSING_LENGTH;
    constexpr int end_bit = sizeof(key_t) * 8;
    constexpr int RADIX_BITS = 4;
    constexpr int RADIX_BUCKETS = 16;
    int begin_bit = 0;
    int *counts;
    key_t *keys_temp;
    value_t *values_temp;
    cudaMalloc(&counts, num_segments * RADIX_BUCKETS * num_tiles * sizeof(int));
    cudaMalloc(&keys_temp, num_segments * num_elements * sizeof(key_t));
    cudaMalloc(&values_temp, num_segments * num_elements * sizeof(value_t));
    cudaDeviceSynchronize();
    key_t *keys_in_ = const_cast<key_t *>(keys_in);
    key_t *keys_out_ = keys_temp;
    value_t *values_in_ = const_cast<value_t *>(values_in);
    value_t *values_out_ = values_temp;
    while (true) {
        segmented_tile_sort_pairs_upsweep_cuda<key_t, value_t, IS_DESCENDING, KEYS_PER_ITEM, BLOCK_THREADS>(
            keys_in_, counts, num_segments, num_elements, begin_bit, end_bit);
        cudaDeviceSynchronize();
        segmented_tile_sort_pairs_scan_cuda<KEYS_PER_ITEM, BLOCK_THREADS, RADIX_BUCKETS>(counts, num_tiles, num_segments);
        cudaDeviceSynchronize();
        segmented_tile_sort_pairs_downsweep_cuda<key_t, value_t, IS_DESCENDING, KEYS_PER_ITEM, BLOCK_THREADS>(
            keys_in_, keys_out_, values_in_, values_out_, num_segments, num_elements, begin_bit, end_bit, counts);
        cudaDeviceSynchronize();
        if (begin_bit == 0) {
            keys_in_ = keys_temp;
            keys_out_ = keys_out;
            values_in_ = values_temp;
            values_out_ = values_out;
        } else {
            std::swap(keys_in_, keys_out_);
            std::swap(values_in_, values_out_);
        }
        begin_bit += RADIX_BITS;
        if (begin_bit >= end_bit) break;
    }
    cudaFree(counts);
    cudaFree(keys_temp);
    cudaFree(values_temp);
}

template <typename key_t, typename value_t, bool IS_DESCENDING>
inline void segmented_sort_pairs_(const key_t *keys_in, key_t *keys_out,
                                  const value_t *values_in, value_t *values_out,
                                  int num_segments, int num_elements) {
    if (num_elements > 4096) {
        std::cout << "Using tile sort\n";
        segmented_tile_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 512>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 2048) {
        segmented_block_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 1024>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 1024) {
        segmented_block_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 512>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 512) {
        segmented_block_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 256>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 256) {
        segmented_block_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 128>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else {
        segmented_block_sort_pairs_cuda<key_t, value_t, IS_DESCENDING, 4, 64>(
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
    std::srand(1337);
    for (int it = 0; it < 40; it++) {
        using key_t = float;
        using value_t = int;
        int num_segments = randint_scalar(4, 20);
        int num_elements = randint_scalar(10, 140960);
        if (it == 10) num_elements = 123;
        else if (it == 11) {
            num_segments = 1;
            num_elements = 33554432; // 2**25
        }
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
        std::cout << milliseconds << "\n";

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

    std::cout << "ok\n";
    return 0;
}
