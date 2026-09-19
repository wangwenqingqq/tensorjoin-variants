#include <ATen/core/CachingHostAllocator.h>
// Called only after synchronized engine shutdown, never in a join/timing loop.
extern "C" void g19EmptyPinnedCache() {
    at::getHostAllocator(at::kCUDA)->empty_cache();
}
