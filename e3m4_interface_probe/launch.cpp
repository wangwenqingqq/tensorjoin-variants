// Standalone fresh-context driver: CUBIN INPUT OUTPUT [REPEATS].
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void check(CUresult r) {
    if (r != CUDA_SUCCESS) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(r, &name); cuGetErrorString(r, &msg);
        std::fprintf(stderr, "CUDA error %d: %s: %s\n", int(r),
                     name ? name : "unknown", msg ? msg : "unknown");
        std::exit(2);
    }
}
int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) return 64;
    int repeats = argc == 5 ? std::stoi(argv[4]) : 1;
    if (repeats < 1 || repeats > 1000) return 64;
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) return 65;
    std::vector<char> input((std::istreambuf_iterator<char>(file)), {});
    if (input.empty() || input.size() % 16 || input.size() / 16 > 65535) return 65;
    const unsigned n = input.size() / 16;
    std::vector<float> output(n * 128);
    check(cuInit(0)); CUdevice dev; check(cuDeviceGet(&dev, 0));
    int major, minor, driver;
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    check(cuDriverGetVersion(&driver));
    if (major != 12 || minor != 0) return 66;
    CUcontext ctx; check(cuCtxCreate(&ctx, nullptr, 0, dev));
    CUmodule mod; CUfunction fn;
    check(cuModuleLoad(&mod, argv[1])); check(cuModuleGetFunction(&fn, mod, "probe"));
    int regs, local, shared;
    check(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
    check(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
    check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
    CUdeviceptr din, dout;
    check(cuMemAlloc(&din, input.size())); check(cuMemAlloc(&dout, output.size()*4));
    check(cuMemcpyHtoD(din, input.data(), input.size()));
    std::ofstream result(argv[3], std::ios::binary | std::ios::trunc);
    if (!result) return 65;
    for (int i = 0; i < repeats; ++i) {
        check(cuMemsetD32(dout, 0x7fc00000, output.size()));
        void* args[] = {&din, &dout};
        check(cuLaunchKernel(fn, n, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
        check(cuCtxSynchronize()); check(cuMemcpyDtoH(output.data(), dout, output.size()*4));
        result.write(reinterpret_cast<char*>(output.data()), output.size()*4);
        if (!result) return 65;
    }
    result.close();
    check(cuMemFree(dout)); check(cuMemFree(din)); check(cuModuleUnload(mod));
    check(cuCtxDestroy(ctx));
    std::printf("{\"sm\":120,\"driver_api\":%d,\"records\":%u,\"repeats\":%d,"
                "\"registers\":%d,\"local_bytes\":%d,\"shared_bytes\":%d}\n",
                driver, n, repeats, regs, local, shared);
}
