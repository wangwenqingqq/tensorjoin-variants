// G6 compatibility adapter for upstream GTS commit 3bac1b725e92e98e3b69d5cb79ad9c56ccb5e639.
// The search/index arithmetic is unchanged; this file supplies raw input and canonical pair export.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "tree.cuh"
#include "search_v2.cuh"

namespace {

void require_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

std::vector<float> load_raw_f32(const std::string& path, std::size_t values) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open input: " + path);
  const auto bytes = input.tellg();
  const auto expected = static_cast<std::streamoff>(values * sizeof(float));
  if (bytes != expected) {
    throw std::runtime_error("input byte count differs from N*D*4");
  }
  input.seekg(0);
  std::vector<float> host(values);
  input.read(reinterpret_cast<char*>(host.data()), expected);
  if (!input) throw std::runtime_error("short input read");
  return host;
}

void write_pairs(const std::string& path,
                 const std::vector<unsigned long long>& pairs) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot open output: " + path);
  output.write(reinterpret_cast<const char*>(pairs.data()),
               static_cast<std::streamsize>(pairs.size() * sizeof(pairs[0])));
  if (!output) throw std::runtime_error("short output write");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: GTS_G6 vectors_f32.raw N D epsilon output_u64.bin\n";
    return 2;
  }

  try {
    const std::string input_path = argv[1];
    const int n = std::stoi(argv[2]);
    const int d = std::stoi(argv[3]);
    const float radius = std::stof(argv[4]);
    const std::string output_path = argv[5];
    if (n <= 0 || d <= 0) throw std::runtime_error("N and D must be positive");
    const std::size_t values = static_cast<std::size_t>(n) * d;
    if (values > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::runtime_error("input size overflow");
    }

    // Disk input is outside the frozen public denominator.
    std::vector<float> host_data = load_raw_f32(input_path, values);

    int* data_info = nullptr;
    float* data_d = nullptr;
    int* qid_list = nullptr;
    int* max_node_num = nullptr;
    int* id_list = nullptr;
    TN* node_list = nullptr;
    int* empty_list = nullptr;
    int tree_h = 0;

    // Upstream exposes these as managed configuration variables. MAX_H=5 is
    // required so 60,000 rows fit leaves of at most MAX_SIZE=20.
    TREE_ORDER = 10;
    MAX_SIZE = 20;
    MAX_H = 5;

    require_cuda(cudaFree(nullptr), "CUDA context initialization");
    const auto start = std::chrono::steady_clock::now();

    require_cuda(cudaMallocManaged(reinterpret_cast<void**>(&data_info),
                                   3 * sizeof(int)), "cudaMallocManaged data_info");
    data_info[0] = d;
    data_info[1] = n;
    data_info[2] = 2;
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&data_d),
                            values * sizeof(float)), "cudaMalloc data_d");
    require_cuda(cudaMemcpy(data_d, host_data.data(), values * sizeof(float),
                            cudaMemcpyHostToDevice), "input H2D");
    require_cuda(cudaMallocManaged(reinterpret_cast<void**>(&qid_list),
                                   static_cast<std::size_t>(n) * sizeof(int)),
                 "cudaMallocManaged qid_list");
    for (int i = 0; i < n; ++i) qid_list[i] = i;

    indexConstru(data_d, nullptr, nullptr, data_info, id_list, node_list,
                 max_node_num, tree_h, empty_list);

    g6_pairs.clear();
    searchIndexRnnV2(data_d, node_list, id_list, max_node_num, qid_list, n,
                     radius, tree_h, data_info, empty_list, nullptr, nullptr);
    std::sort(g6_pairs.begin(), g6_pairs.end());
    require_cuda(cudaDeviceSynchronize(), "final synchronize");

    const auto stop = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(stop - start).count();

    // File output and hashing are outside the denominator.
    write_pairs(output_path, g6_pairs);
    std::cout << "G6_METRIC elapsed_seconds=" << elapsed << "\n";
    std::cout << "G6_METRIC pair_count=" << g6_pairs.size() << "\n";
    std::cout << "G6_METRIC tree_height=" << tree_h << "\n";
    std::cout << "G6_METRIC tree_order=" << TREE_ORDER << "\n";
    std::cout << "G6_METRIC max_leaf_size=" << MAX_SIZE << "\n";
    std::cout << "G6_METRIC max_height=" << MAX_H << "\n";
    std::cout << "G6_METRIC workspace_cap_bytes=" << G6_WORKSPACE_CAP_BYTES << "\n";

    cudaFree(data_info);
    cudaFree(data_d);
    cudaFree(qid_list);
    cudaFree(id_list);
    cudaFree(node_list);
    cudaFree(max_node_num);
    cudaFree(empty_list);
    cudaFree(res);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "G6_ERROR " << error.what() << "\n";
    return 1;
  }
}
