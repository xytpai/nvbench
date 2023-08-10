#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define PRINT_PROP(PARAM) std::cout << #PARAM << ": " << prop.PARAM << std::endl;

int main() {
    int dev;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    PRINT_PROP(name)

    PRINT_PROP(totalConstMem)
    PRINT_PROP(totalGlobalMem)
    PRINT_PROP(unifiedAddressing)
    PRINT_PROP(unifiedFunctionPointers)

    PRINT_PROP(sharedMemPerBlock)
    PRINT_PROP(sharedMemPerBlockOptin)
    PRINT_PROP(sharedMemPerMultiprocessor)

    PRINT_PROP(warpSize)
    PRINT_PROP(multiProcessorCount)
    PRINT_PROP(maxBlocksPerMultiProcessor)

    PRINT_PROP(clockRate)
    PRINT_PROP(ECCEnabled)
    PRINT_PROP(memoryClockRate)
    PRINT_PROP(memoryBusWidth)
    PRINT_PROP(memPitch)

    PRINT_PROP(maxThreadsPerBlock)
    PRINT_PROP(maxThreadsPerMultiProcessor)

    PRINT_PROP(l2CacheSize)
    PRINT_PROP(localL1CacheSupported)
    PRINT_PROP(globalL1CacheSupported)
}
