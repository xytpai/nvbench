#pragma once

#include "sorting_common.h"

namespace sorting_impl {

template <
    typename KeyT,
    int BLOCK_THREADS_,
    int KEYS_PER_THREAD_,
    bool IS_DESCENDING_ = false,
    typename ValueT = NullType,
    typename DigitT = uint16_t,   // Covering BLOCK_THREADS * KEYS_PER_THREAD.
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform packed prefix sum.
    int RADIX_BITS_ = 4>
class BlockRadixSort {
public:
    using KeyTraitsT = typename KeyTraits<KeyT>::Type;

    enum {
        BLOCK_THREADS = BLOCK_THREADS_,
        KEYS_PER_THREAD = KEYS_PER_THREAD_,
        IS_DESCENDING = IS_DESCENDING_,
        RADIX_BITS = RADIX_BITS_,

        PROCESSING_LENGTH = BLOCK_THREADS * KEYS_PER_THREAD,
        RADIX_BUCKETS = 1 << RADIX_BITS,
        KEYS_ONLY = std::is_same<ValueT, NullType>::value,
        PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
        COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,
        LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE,
        DIGIT_BITS = sizeof(DigitT) << 3,
        KEY_TRAITS_TYPE_MASK = 1l << ((sizeof(KeyTraitsT) << 3) - 1),
    };

    static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
    static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");
    static_assert(
        ((1l << (sizeof(DigitT) << 3)) - 1)
            >= (BLOCK_THREADS * KEYS_PER_THREAD),
        " ");

private:
    union RankT {
        CounterT counters[COUNTER_LANES][BLOCK_THREADS];
        CounterT counters_flat[COUNTER_LANES * BLOCK_THREADS];
        DigitT buckets[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
    };

    union LocalStorage {
        RankT rank_storage;
        struct {
            KeyTraitsT exchange_ukeys[PROCESSING_LENGTH];
            int relative_bin_offsets[RADIX_BUCKETS];
        };
        ValueT exchange_values[PROCESSING_LENGTH];
    };

    LocalStorage &local_storage_;
    int lid_;
    int bin_offset_;

    int ranks_[KEYS_PER_THREAD];
    KeyTraitsT ukeys_[KEYS_PER_THREAD];
    ValueT values_[KEYS_PER_THREAD];
    int relative_bin_offsets_[KEYS_PER_THREAD];
    int begin_bit_;
    int pass_bits_;
    bool enable_bin_offsets_ = false;

public:
    static HOST_DEVICE int LocalMemorySize() {
        return sizeof(LocalStorage);
    }

    DEVICE inline void load_bin_offsets(int *counts, int block_id, int num_blocks) {
        int bin_idx = lid_;
        if (lid_ < RADIX_BUCKETS) {
            if (IS_DESCENDING)
                bin_idx = RADIX_BUCKETS - bin_idx - 1;
            bin_offset_ = counts[block_id + bin_idx * num_blocks];
        }
        enable_bin_offsets_ = true;
        __syncthreads();
    }

    DEVICE inline BlockRadixSort(int lid, char *local_ptr) :
        lid_(lid),
        local_storage_(reinterpret_cast<LocalStorage &>(*local_ptr)) {
    }

    DEVICE inline void load_keys(const KeyT *keys_block_in, int num_elements, int block_offset = 0) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = block_offset + lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < num_elements) {
                ukeys_[ITEM] = KeyTraits<KeyT>::convert(keys_block_in[offset]);
            } else {
                KeyTraitsT padding_key;
                if (IS_DESCENDING) {
                    padding_key = 0;
                } else {
                    padding_key = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
                    padding_key = padding_key ^ (padding_key - 1);
                }
                ukeys_[ITEM] = padding_key;
            }
        }
    }

    DEVICE inline void load_values(const ValueT *values_block_in, int num_elements, int block_offset = 0) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = block_offset + lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < num_elements) {
                values_[ITEM] = values_block_in[offset];
            }
        }
    }

    DEVICE inline void store_keys(KeyT *keys_block_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_ukeys[lid_ * KEYS_PER_THREAD + ITEM] = ukeys_[ITEM];
        }
        __syncthreads();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            if (offset < num_elements) {
                keys_block_out[offset] = KeyTraits<KeyT>::deconvert(local_storage_.exchange_ukeys[offset]);
            }
        }
        __syncthreads();
    }

    DEVICE inline void store_values(ValueT *values_block_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_values[lid_ * KEYS_PER_THREAD + ITEM] = values_[ITEM];
        }
        __syncthreads();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            if (offset < num_elements) {
                values_block_out[offset] = local_storage_.exchange_values[offset];
            }
        }
        __syncthreads();
    }

    DEVICE inline void exchange_and_store_keys(KeyT *keys_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_ukeys[ranks_[ITEM]] = ukeys_[ITEM];
        }
        __syncthreads();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            auto ukey = local_storage_.exchange_ukeys[offset];
            relative_bin_offsets_[ITEM] = local_storage_.relative_bin_offsets[extract_digit(ukey)];
            offset += relative_bin_offsets_[ITEM];
            if (offset < num_elements) {
                keys_out[offset] = KeyTraits<KeyT>::deconvert(ukey);
            }
        }
        __syncthreads();
    }

    DEVICE inline void exchange_and_store_values(ValueT *values_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_values[ranks_[ITEM]] = values_[ITEM];
        }
        __syncthreads();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            auto value = local_storage_.exchange_values[offset];
            offset += relative_bin_offsets_[ITEM];
            if (offset < num_elements) {
                values_out[offset] = value;
            }
        }
        __syncthreads();
    }

    DEVICE inline void exchange_keys() {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_ukeys[ranks_[ITEM]] = ukeys_[ITEM];
        }
        __syncthreads();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            ukeys_[ITEM] = local_storage_.exchange_ukeys[offset];
        }
        __syncthreads();
    }

    DEVICE inline void exchange_values() {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_values[ranks_[ITEM]] = values_[ITEM];
        }
        __syncthreads();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            values_[ITEM] = local_storage_.exchange_values[offset];
        }
        __syncthreads();
    }

    DEVICE inline DigitT extract_digit(KeyTraitsT key) {
        return ((key >> begin_bit_) & ((1 << pass_bits_) - 1));
    }

    DEVICE inline void rank_keys(int begin_bit, int end_bit) {
        begin_bit_ = begin_bit;
        pass_bits_ = end_bit - begin_bit_;
        pass_bits_ = RADIX_BITS < pass_bits_ ? RADIX_BITS : pass_bits_;
        DigitT *digit_counters[KEYS_PER_THREAD];

        // reset buckets
#pragma unroll
        for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
            local_storage_.rank_storage.counters[ITEM][lid_] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto digit = extract_digit(ukeys_[ITEM]);
            auto sub_counter = digit >> LOG_COUNTER_LANES;
            auto counter_lane = digit & (COUNTER_LANES - 1);
            if (IS_DESCENDING) {
                sub_counter = PACKING_RATIO - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }
            digit_counters[ITEM] =
                &local_storage_.rank_storage.buckets[counter_lane][lid_][sub_counter];
            ranks_[ITEM] = *digit_counters[ITEM];
            *digit_counters[ITEM] = ranks_[ITEM] + 1;
        }
        __syncthreads();

        CounterT exclusive = block_exclusive_cumsum<
            CounterT,
            COUNTER_LANES,
            BLOCK_THREADS>(
            local_storage_.rank_storage.counters_flat,
            lid_);

        CounterT c = 0;
#pragma unroll
        for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
            exclusive = exclusive << DIGIT_BITS;
            c += exclusive;
        }

#pragma unroll
        for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
            local_storage_.rank_storage.counters[INDEX][lid_] += c;
        }
        __syncthreads();

        // inc rank
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            ranks_[ITEM] += *digit_counters[ITEM];
        }
        __syncthreads();

        if (enable_bin_offsets_) {
            int digit = lid_;
            if (lid_ < RADIX_BUCKETS) {
                if (IS_DESCENDING)
                    digit = RADIX_BUCKETS - digit - 1;
                auto sub_counter = digit >> LOG_COUNTER_LANES;
                auto counter_lane = digit & (COUNTER_LANES - 1);
                int digit_offset = local_storage_.rank_storage.buckets[counter_lane][0][sub_counter];
                local_storage_.relative_bin_offsets[lid_] = bin_offset_ - digit_offset;
            }
            __syncthreads();
        }
    }
};

template <
    typename KeyT,
    int BLOCK_THREADS_,
    int KEYS_PER_THREAD_,
    bool IS_DESCENDING_ = false,
    typename ValueT = NullType,
    typename DigitT = u_char,
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform packed prefix sum.
    int RADIX_BITS = 4,
    int WARP_SIZE = 32>
class RadixSortUpsweep {
public:
    using KeyTraitsT = typename KeyTraits<KeyT>::Type;
    enum {
        BLOCK_THREADS = BLOCK_THREADS_,
        KEYS_PER_THREAD = KEYS_PER_THREAD_,
        IS_DESCENDING = IS_DESCENDING_,

        PROCESSING_LENGTH = BLOCK_THREADS * KEYS_PER_THREAD,
        RADIX_BUCKETS = 1 << RADIX_BITS,
        KEYS_ONLY = std::is_same<ValueT, NullType>::value,
        PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
        LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE,
        COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,

        WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE,
        LANES_PER_WARP = std::max<int>(1, (COUNTER_LANES + WARPS - 1) / WARPS),
    };

    static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
    static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");

private:
    union LocalStorage {
        CounterT counters[COUNTER_LANES][BLOCK_THREADS];
        DigitT buckets[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
        int block_counters[WARP_SIZE][RADIX_BUCKETS];
    };

    const KeyT *keys_in_;
    int lid_;
    int bid_;
    int begin_bit_;
    int end_bit_;
    int num_blocks_;
    int *count_out_;
    int warp_id_;
    int warp_tid_;

    LocalStorage &local_storage_;
    int local_counts_[LANES_PER_WARP][PACKING_RATIO];

public:
    static HOST_DEVICE int LocalMemorySize() {
        return sizeof(LocalStorage);
    }

    DEVICE inline RadixSortUpsweep(const KeyT *keys_in, int lid, int bid,
                                   int begin_bit, int end_bit, int num_blocks, int *count_out, char *local_ptr) :
        keys_in_(keys_in),
        lid_(lid), bid_(bid), begin_bit_(begin_bit), end_bit_(end_bit),
        num_blocks_(num_blocks), count_out_(count_out),
        local_storage_(reinterpret_cast<LocalStorage &>(*local_ptr)) {
        warp_id_ = lid_ / WARP_SIZE;
        warp_tid_ = lid_ % WARP_SIZE;
    }

    DEVICE inline DigitT extract_digit(KeyTraitsT key) {
        auto pass_bits = end_bit_ - begin_bit_;
        pass_bits = RADIX_BITS < pass_bits ? RADIX_BITS : pass_bits;
        return ((key >> begin_bit_) & ((1 << pass_bits) - 1));
    }

    DEVICE inline void process_full_tile(int block_offset) {
        KeyTraitsT keys[KEYS_PER_THREAD];
        auto block_ptr = keys_in_ + block_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            keys[ITEM] =
                KeyTraits<KeyT>::convert(block_ptr[lid_ + ITEM * BLOCK_THREADS]);
        }
        __syncthreads();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto digit = extract_digit(keys[ITEM]);
            auto sub_counter = digit & (PACKING_RATIO - 1);
            auto row_offset = digit >> LOG_PACKING_RATIO;
            local_storage_.buckets[row_offset][lid_][sub_counter]++;
        }
    }

    DEVICE inline void process_partial_tile(int block_offset, int block_end) {
        for (int offset = block_offset + lid_; offset < block_end; offset += BLOCK_THREADS) {
            KeyTraitsT key = KeyTraits<KeyT>::convert(keys_in_[offset]);
            auto digit = extract_digit(key);
            auto sub_counter = digit & (PACKING_RATIO - 1);
            auto row_offset = digit >> LOG_PACKING_RATIO;
            local_storage_.buckets[row_offset][lid_][sub_counter]++;
        }
    }

    DEVICE inline void reset_digit_counters() {
#pragma unroll
        for (int LANE = 0; LANE < COUNTER_LANES; ++LANE)
            local_storage_.counters[LANE][lid_] = 0;
    }

    DEVICE inline void reset_unpacked_counters() {
#pragma unroll
        for (int LANE = 0; LANE < LANES_PER_WARP; ++LANE) {
#pragma unroll
            for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; ++UNPACKED_COUNTER) {
                local_counts_[LANE][UNPACKED_COUNTER] = 0;
            }
        }
    }

    DEVICE inline void unpack_digit_counts() {
#pragma unroll
        for (int LANE = 0; LANE < LANES_PER_WARP; ++LANE) {
            int counter_lane = (LANE * WARPS) + warp_id_;
            if (counter_lane < COUNTER_LANES) {
#pragma unroll
                for (int PACKED_COUNTER = 0; PACKED_COUNTER < BLOCK_THREADS; PACKED_COUNTER += WARP_SIZE) {
#pragma unroll
                    for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; ++UNPACKED_COUNTER) {
                        int counter = local_storage_.buckets[counter_lane][warp_tid_ + PACKED_COUNTER][UNPACKED_COUNTER];
                        local_counts_[LANE][UNPACKED_COUNTER] += counter;
                    }
                }
            }
        }
    }

    DEVICE inline void extract_counts() {
#pragma unroll
        for (int LANE = 0; LANE < LANES_PER_WARP; ++LANE) {
            int counter_lane = (LANE * WARPS) + warp_id_;
            if (counter_lane < COUNTER_LANES) {
                int digit_row = counter_lane << LOG_PACKING_RATIO;
#pragma unroll
                for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; ++UNPACKED_COUNTER) {
                    int bin_idx = digit_row + UNPACKED_COUNTER;
                    local_storage_.block_counters[warp_tid_][bin_idx] =
                        local_counts_[LANE][UNPACKED_COUNTER];
                }
            }
        }

        __syncthreads();

        if ((RADIX_BUCKETS % BLOCK_THREADS != 0) && (lid_ < RADIX_BUCKETS)) {
            int bin_idx = lid_;
            int bin_count = 0;
#pragma unroll
            for (int i = 0; i < WARP_SIZE; ++i)
                bin_count += local_storage_.block_counters[i][bin_idx];
            if (IS_DESCENDING)
                bin_idx = RADIX_BUCKETS - bin_idx - 1;
            count_out_[(num_blocks_ * bin_idx) + bid_] = bin_count;
        }
    }

    DEVICE inline void run(int block_offset, int block_end) {
        reset_digit_counters();
        reset_unpacked_counters();

        // Unroll batches of full tiles
        int UNROLL_COUNT = 255 / 4; // the largest value for counter
        int UNROLLED_ELEMENTS = UNROLL_COUNT * PROCESSING_LENGTH;
        while (block_offset + UNROLLED_ELEMENTS <= block_end) {
            for (int i = 0; i < UNROLL_COUNT; ++i) {
                process_full_tile(block_offset);
                block_offset += PROCESSING_LENGTH;
            }
            __syncthreads();
            unpack_digit_counts();
            __syncthreads();
            reset_digit_counters();
        }

        while (block_offset + PROCESSING_LENGTH <= block_end) {
            process_full_tile(block_offset);
            block_offset += PROCESSING_LENGTH;
        }

        process_partial_tile(block_offset, block_end);
        __syncthreads();
        unpack_digit_counts();
        __syncthreads();
        extract_counts();
    }
};

template <int BLOCK_THREADS, int THREAD_WORK_SIZE = 4, int WARP_SIZE = 32>
class RadixSortScanBins {
public:
    enum {
        PROCESSING_LENGTH = BLOCK_THREADS * THREAD_WORK_SIZE,
        NUM_WARPS = BLOCK_THREADS / WARP_SIZE,
    };

private:
    int *count_;
    int *slm_;
    int lid_;

public:
    static HOST_DEVICE int LocalMemorySize() {
        return NUM_WARPS * sizeof(int);
    }

    DEVICE inline RadixSortScanBins(int *count, int *slm, int lid) :
        count_(count), slm_(slm), lid_(lid) {
    }

    template <bool is_partial>
    DEVICE inline void consume_tile(int block_offset, int &running_prefix, int tile_bound = 0) {
        // Load
        int partial_output[THREAD_WORK_SIZE];
        auto d_local = count_ + block_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < THREAD_WORK_SIZE; ++ITEM) {
            int offset = lid_ * THREAD_WORK_SIZE + ITEM;
            if constexpr (is_partial) {
                if (offset < tile_bound) {
                    partial_output[ITEM] = d_local[offset];
                } else {
                    partial_output[ITEM] = *d_local;
                }
            } else {
                partial_output[ITEM] = d_local[offset];
            }
        }
        __syncthreads();
        // Thread reduce
        int thread_partial = partial_output[0];
#pragma unroll
        for (int ITEM = 1; ITEM < THREAD_WORK_SIZE; ++ITEM) {
            thread_partial = thread_partial + partial_output[ITEM];
        }
        // Warp scan
        int warp_tid = lid_ % WARP_SIZE;
        int warp_id = lid_ / WARP_SIZE;
        const int WARP_SCAN_STEPS = Log2<WARP_SIZE>::VALUE;
        int warp_inclusive_sum, warp_exclusive_sum;
        warp_cumsum<int, WARP_SCAN_STEPS>(
            warp_tid,
            thread_partial,
            warp_inclusive_sum,
            warp_exclusive_sum);
        if (warp_tid == (WARP_SIZE - 1))
            slm_[warp_id] = warp_inclusive_sum;
        __syncthreads();
        // Block scan
        int block_all_sum = 0, warp_prefix_sum;
#pragma unroll
        for (int i = 0; i < NUM_WARPS; ++i) {
            if (warp_id == i)
                warp_prefix_sum = block_all_sum;
            block_all_sum += slm_[i];
        }
        warp_exclusive_sum += warp_prefix_sum;
        warp_exclusive_sum += running_prefix;
        if (lid_ == 0)
            running_prefix += block_all_sum;
        // Write back
        int inclusive = partial_output[0];
        inclusive = warp_exclusive_sum + inclusive;
        partial_output[0] = warp_exclusive_sum;
        int exclusive = inclusive;
#pragma unroll
        for (int ITEM = 1; ITEM < THREAD_WORK_SIZE; ++ITEM) {
            inclusive = exclusive + partial_output[ITEM];
            partial_output[ITEM] = exclusive;
            exclusive = inclusive;
        }
#pragma unroll
        for (int ITEM = 0; ITEM < THREAD_WORK_SIZE; ITEM++) {
            int offset = lid_ * THREAD_WORK_SIZE + ITEM;
            if constexpr (is_partial) {
                if (offset < tile_bound) {
                    d_local[offset] = partial_output[ITEM];
                }
            } else {
                d_local[offset] = partial_output[ITEM];
            }
        }
    }

    DEVICE inline void run(int num_counts) {
        int block_offset = 0;
        int running_prefix = 0;
        while (block_offset + PROCESSING_LENGTH <= num_counts) {
            consume_tile<false>(block_offset, running_prefix);
            block_offset += PROCESSING_LENGTH;
        }
        if (block_offset < num_counts) {
            consume_tile<true>(block_offset, running_prefix, num_counts - block_offset);
        }
    }
};

} // namespace sorting_impl
