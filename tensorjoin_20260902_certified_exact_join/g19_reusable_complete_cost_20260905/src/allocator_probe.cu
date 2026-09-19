#include <cuda_runtime.h>
#include <cstdio>
// Localization only: intentionally exercise the rejected deallocator, then
// release the still-live allocation correctly. No workload/kernel timing.
int main() {
    void* pointer=nullptr;
    cudaError_t allocation=cudaMallocHost(&pointer,72);
    if(allocation!=cudaSuccess)return 2;
    cudaPointerAttributes attributes{};
    cudaError_t attribute=cudaPointerGetAttributes(&attributes,pointer);
    cudaError_t rejected=cudaFree(pointer);
    cudaGetLastError();
    cudaError_t repaired=cudaFreeHost(pointer);
    std::printf("{\"bytes\":72,\"allocation\":%d,\"attributes\":%d,\"memory_type\":%d,\"wrong_free\":%d,\"correct_free\":%d,\"diagnostic_only\":true}\n",
                int(allocation),int(attribute),int(attributes.type),int(rejected),int(repaired));
    return allocation==cudaSuccess && attribute==cudaSuccess && attributes.type==cudaMemoryTypeHost
        && rejected==cudaErrorInvalidValue && repaired==cudaSuccess?0:2;
}
