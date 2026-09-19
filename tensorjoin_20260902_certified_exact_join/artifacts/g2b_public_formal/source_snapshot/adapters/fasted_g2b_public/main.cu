#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "DataLoader/PointList.hpp"
#include "DataLoader/half.hpp"
#include "findPairs.cuh"

using InPrec = Mma::InPrec;

namespace {

int roundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

double parseDouble(const std::string& value) {
    return std::stod(value);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <float32-raw-file> <dimensions> <epsilon>" << std::endl;
        return 1;
    }
    const char* inputPath = argv[1];
    const int dimensions = std::stoi(argv[2]);
    const double epsilon = parseDouble(argv[3]);
    if (dimensions <= 0) {
        std::cerr << "Dimensions must be positive" << std::endl;
        return 1;
    }

    // Context setup is outside the public denominator.
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaFree(nullptr));
    gpuErrchk(cudaDeviceSynchronize());

    std::ifstream input(inputPath, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Failed to open input: " << inputPath << std::endl;
        return 2;
    }
    const std::streamsize inputBytes = input.tellg();
    if (inputBytes <= 0 || inputBytes % (sizeof(float) * dimensions) != 0) {
        std::cerr << "Invalid float32 input byte count: " << inputBytes << std::endl;
        return 2;
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> source(static_cast<size_t>(inputBytes) / sizeof(float));
    if (!input.read(reinterpret_cast<char*>(source.data()), inputBytes)) {
        std::cerr << "Failed to read input" << std::endl;
        return 2;
    }
    input.close();
    const int numPoints = source.size() / dimensions;

    const Mma::mmaShape blockTile = SimSearch::GetBlockTileDims();
    const int paddedDimensions = roundUp(dimensions, blockTile.k);
    const int paddedPoints = roundUp(numPoints, SimSearch::rasterizeSize * blockTile.m);
    const Mma::mmaShape paddedShape{paddedPoints, paddedPoints, paddedDimensions};
    const Mma::mmaShape inputShape{numPoints, numPoints, dimensions};

    // Start with only the pageable float32 source resident in host memory.
    const double publicStart = omp_get_wtime();
    std::vector<half_float::half> values(
        static_cast<size_t>(paddedPoints) * paddedDimensions, half_float::half(0.0f));
    #pragma omp parallel for
    for (unsigned long long index = 0; index < source.size(); ++index) {
        const unsigned long long row = index / dimensions;
        const unsigned long long column = index - row * dimensions;
        values[row * paddedDimensions + column] = half_float::half(source[index]);
    }
    Points::PointList<half_float::half> pointList(
        std::move(values), paddedPoints, numPoints, paddedDimensions, dimensions);
    std::vector<Pairs::Pair> materializedPairs;
    auto hostParams = SimSearch::FindPairsParamsHost{
        epsilon,
        paddedShape,
        inputShape,
        std::move(pointList),
        false,
        false,
        "",
        &materializedPairs,
    };
    const SimSearch::Results searchResults = SimSearch::FindPairs(hostParams);

    std::vector<unsigned long long> canonical;
    canonical.reserve(materializedPairs.size());
    unsigned long long invalidPairs = 0;
    for (const Pairs::Pair& pair : materializedPairs) {
        if (pair.QueryPoint < 0 || pair.QueryPoint >= numPoints ||
            pair.CandidatePoint < 0 || pair.CandidatePoint >= numPoints) {
            ++invalidPairs;
            continue;
        }
        canonical.push_back(
            static_cast<unsigned long long>(pair.QueryPoint) * numPoints +
            static_cast<unsigned long long>(pair.CandidatePoint));
    }
    std::sort(canonical.begin(), canonical.end());
    const double publicSeconds = omp_get_wtime() - publicStart;

    std::cout << std::fixed << std::setprecision(9)
              << "G2B_PUBLIC_SECONDS=" << publicSeconds << std::endl;
    std::cout << "G2B_PUBLIC_OUTPUT_PAIRS=" << canonical.size() << std::endl;
    std::cout << "FASTED_G2B_PAIRS_FOUND=" << searchResults.pairsFound << std::endl;
    std::cout << "FASTED_G2B_PAIRS_STORED=" << searchResults.pairsStored << std::endl;
    std::cout << "FASTED_G2B_MATERIALIZED_PAIRS=" << materializedPairs.size() << std::endl;
    std::cout << "FASTED_G2B_INVALID_PAIRS=" << invalidPairs << std::endl;
    std::cout << "FASTED_G2B_PADDED_SHAPE=" << paddedPoints << "," << paddedPoints
              << "," << paddedDimensions << std::endl;

    const char* outputPath = std::getenv("FASTED_G2B_OUTPUT");
    if (outputPath == nullptr) {
        std::cerr << "FASTED_G2B_OUTPUT is required" << std::endl;
        return 3;
    }
    // Disk serialization is outside the public denominator.
    std::ofstream output(outputPath, std::ios::binary | std::ios::out);
    output.write(reinterpret_cast<const char*>(canonical.data()),
                 static_cast<std::streamsize>(canonical.size() * sizeof(unsigned long long)));
    output.close();
    if (!output) {
        std::cerr << "Failed to write canonical output" << std::endl;
        return 4;
    }
    SimSearch::releaseGlobalMemory();
    return invalidPairs == 0 && searchResults.pairsStored == materializedPairs.size() ? 0 : 5;
}
