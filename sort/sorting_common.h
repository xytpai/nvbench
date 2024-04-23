#pragma once

#include <stdint.h>

#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

namespace sorting_impl {

struct NullType {
    using value_type = NullType;
    template <typename T>
    HOST_DEVICE_INLINE NullType &operator=(const T &) {
        return *this;
    }
    HOST_DEVICE_INLINE bool operator==(const NullType &) {
        return true;
    }
    HOST_DEVICE_INLINE bool operator!=(const NullType &) {
        return false;
    }
};

template <typename Type>
struct KeyTraits {};

template <>
struct KeyTraits<NullType> {
    using Type = uint32_t;
    DEVICE static inline Type convert(float v) {
        return 0;
    }
    DEVICE static inline NullType deconvert(Type v) {
        return NullType();
    }
};

template <>
struct KeyTraits<float> {
    using Type = uint32_t;
    DEVICE static inline Type convert(float v) {
        Type x = *((Type *)&v);
        Type mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
        return (x ^ mask);
    }
    DEVICE static inline float deconvert(Type v) {
        Type mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
        auto v_de = v ^ mask;
        return *((float *)&v_de);
    }
};

template <>
struct KeyTraits<uint8_t> {
    using Type = uint8_t;
    DEVICE static inline Type convert(uint8_t v) {
        return v;
    }
    DEVICE static inline uint8_t deconvert(Type v) {
        return v;
    }
};

template <>
struct KeyTraits<int8_t> {
    using Type = int8_t;
    DEVICE static inline Type convert(int8_t v) {
        return 128u + v;
    }
    DEVICE static inline int8_t deconvert(Type v) {
        return v - 128;
    }
};

template <>
struct KeyTraits<int16_t> {
    using Type = int16_t;
    DEVICE static inline Type convert(int16_t v) {
        return 32768u + v;
    }
    DEVICE static inline int16_t deconvert(Type v) {
        return v - 32768;
    }
};

template <>
struct KeyTraits<int32_t> {
    using Type = uint32_t;
    DEVICE static inline Type convert(int32_t v) {
        return 2147483648u + v;
    }
    DEVICE static inline int32_t deconvert(Type v) {
        return v - 2147483648u;
    }
};

template <>
struct KeyTraits<int64_t> {
    using Type = uint64_t;
    DEVICE static inline Type convert(int64_t v) {
        return 9223372036854775808ull + v;
    }
    DEVICE static inline int64_t deconvert(Type v) {
        return v - 9223372036854775808ull;
    }
};

template <>
struct KeyTraits<double> {
    using Type = uint64_t;
    DEVICE static inline Type convert(double v) {
        Type x = *((Type *)&v);
        Type mask = -((x >> 63)) | 0x8000000000000000;
        return (x ^ mask);
    }
    DEVICE static inline double deconvert(Type v) {
        Type mask = ((v >> 63) - 1) | 0x8000000000000000;
        auto v_de = v ^ mask;
        return *((double *)&v_de);
    }
};

template <typename T>
DEVICE_INLINE T NUMERIC_MIN(T A, T B) {
    return (((A) > (B)) ? (B) : (A));
}

template <typename T>
DEVICE_INLINE T DivUp(T A, T B) {
    return (((A) + (B)-1) / (B));
}

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
    enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

/*
The computational pipeline is as following:
warp0               |  warp1
in0  in1  in2  in3  |  in4  in5  in6  in7
1    1    1    1    |  1    1    1    1
-> warp_scan
1    2    3    4    |  1    2    3    4   <-inclusive_sum
0    1    2    3    |  0    1    2    3   <-exclusive_sum
STEPS should be Log2<WARP_SIZE>::VALUE
*/
template <typename T, int STEPS>
DEVICE inline void warp_cumsum(const int wid, const T input, T &inclusive_sum, T &exclusive_sum) {
    inclusive_sum = input;
#pragma unroll
    for (int i = 0; i < STEPS; ++i) {
        uint32_t offset = 1u << i;
        T temp = __shfl_up_sync(0xffffffff, inclusive_sum, offset);
        if (wid >= offset) inclusive_sum += temp;
    }
    exclusive_sum = inclusive_sum - input;
}

/*
Perform cumsum blockly.
The input sequence shuold have fixed size : prb_size = COUNTER_LANES * THREADS
Note that a thread handle COUNTER_LANES items of contiguous memory.
*/
template <
    typename T,
    int COUNTER_LANES,
    int THREADS,
    bool EXCLUSIVE = true,
    int WARP_SIZE = 32>
DEVICE inline T block_cumsum(T *storage, const int lid) {
    static_assert(THREADS % WARP_SIZE == 0, "THREADS should be n * WARP_SIZE. (n = 1, 2, 3, ...)");

    const int NUM_WARPS = THREADS / WARP_SIZE;
    const int WARP_CUMSUM_STEPS = Log2<WARP_SIZE>::VALUE;

    int warp_local_id = lid % WARP_SIZE;
    int warp_id = lid / WARP_SIZE;
    int lane_temp_values[COUNTER_LANES];

    // Read input lane sum
    auto storage_lanes = &storage[lid * COUNTER_LANES];
    T lane_all_sum = 0;

    if (EXCLUSIVE) {
#pragma unroll
        for (int lane = 0; lane < COUNTER_LANES; ++lane) {
            lane_temp_values[lane] = lane_all_sum;
            lane_all_sum += storage_lanes[lane];
        }
    } else {
#pragma unroll
        for (int lane = 0; lane < COUNTER_LANES; ++lane) {
            lane_all_sum += storage_lanes[lane];
            lane_temp_values[lane] = lane_all_sum;
        }
    }

    // Get warp level exclusive sum
    T warp_inclusive_sum, warp_exclusive_sum;
    warp_cumsum<T, WARP_CUMSUM_STEPS>(
        warp_local_id,
        lane_all_sum,
        warp_inclusive_sum,
        warp_exclusive_sum);
    __syncthreads();

    // Write to storage
    if (warp_local_id == (WARP_SIZE - 1))
        storage[warp_id] = warp_inclusive_sum;
    __syncthreads();

    // Get block prefix
    T block_all_sum = 0, block_exclusive_sum;
#pragma unroll
    for (int i = 0; i < NUM_WARPS; ++i) {
        if (warp_id == i)
            block_exclusive_sum = block_all_sum;
        block_all_sum += storage[i];
    }
    __syncthreads();

    // Write to storage
    warp_exclusive_sum += block_exclusive_sum;
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
        storage_lanes[lane] = warp_exclusive_sum + lane_temp_values[lane];
    }
    __syncthreads();

    return block_all_sum;
}

template <typename T, int COUNTER_LANES, int THREADS>
DEVICE inline T block_exclusive_cumsum(T *slm_storage, const int lid) {
    return block_cumsum<T, COUNTER_LANES, THREADS, true>(slm_storage, lid);
}

template <typename T, int COUNTER_LANES, int THREADS>
DEVICE inline T block_inclusive_cumsum(T *slm_storage, const int lid) {
    return block_cumsum<T, COUNTER_LANES, THREADS, false>(slm_storage, lid);
}

} // namespace sorting_impl
