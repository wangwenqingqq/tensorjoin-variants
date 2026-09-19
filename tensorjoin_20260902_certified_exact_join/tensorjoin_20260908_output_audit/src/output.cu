#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/version.cuh>
#include <climits>
#include <cstdint>

using Key = unsigned long long;
__global__ void mirror_ids(const Key* upper, Key* expanded, int count,
                          unsigned int n, unsigned int* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Key a = upper[i];
    const Key r = a / n, c = a % n;
    if (a >= Key(n) * n || r > c) atomicOr(status + 1, 1u);
    expanded[2ull * i] = a;
    expanded[2ull * i + 1] = r == c ? ULLONG_MAX : c * n + r;
    if (r == c) atomicAdd(status, 1u);
}
extern "C" int output_mirror(const Key* upper, Key* expanded, int count,
                             unsigned int n, unsigned int* status,
                             cudaStream_t stream) {
    if (count < 0 || count > INT_MAX / 2 || n == 0) return cudaErrorInvalidValue;
    if (count) mirror_ids<<<(count + 255) / 256, 256, 0, stream>>>(upper, expanded, count, n, status);
    return cudaGetLastError();
}
extern "C" int output_sort(void* temporary, size_t* bytes, const Key* input,
                           Key* output, int count, cudaStream_t stream) {
    if (count < 0) return cudaErrorInvalidValue;
    return cub::DeviceRadixSort::SortKeys(temporary, *bytes, input, output,
                                         count, 0, 64, stream);
}
extern "C" int output_cub_version() { return CUB_VERSION; }
