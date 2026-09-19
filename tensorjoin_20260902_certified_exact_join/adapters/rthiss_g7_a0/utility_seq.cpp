#include <map>

#include "utility.h"

std::vector<Sphere> buildSpheresPrimBuffer(std::vector<uint32_t> pointsToSearch, float* dataset, unsigned DIM,
                                           const float EPSILON)
{
    const unsigned int POINTS_TO_SEARCH = pointsToSearch.size();
    std::vector<Sphere> spheres(POINTS_TO_SEARCH);
    

    if (DIM == 1)
    {
        for (int i = 0; i < POINTS_TO_SEARCH; i++)
        {
            const unsigned int index = DIM * pointsToSearch[i];
            spheres[i] = Sphere{vec3f(dataset[index], 0, 0), EPSILON};
        }
    }
    else if (DIM == 2)
    {
        for (int i = 0; i < POINTS_TO_SEARCH; i++)
        {
            const unsigned int index = DIM * pointsToSearch[i];
            spheres[i] = Sphere{vec3f(dataset[index], dataset[index + 1], 0), EPSILON};
        }
    }
    else
    {
        int i = 0;
        
        #pragma omp parallel for private(i) schedule(static) shared(dataset, spheres)
        for (i = 0; i < POINTS_TO_SEARCH; i++)
        {
            const unsigned int index = DIM * pointsToSearch[i];
            spheres[i] = Sphere{vec3f(dataset[index], dataset[index + 1], dataset[index + 2]), EPSILON};
        }
    }

    return spheres;
}

std::vector<Sphere> buildDataPartitionSpheresPrimBuffer(std::vector<uint32_t> pointsToSearch, float* dataset,
                                                        unsigned DIM, const float EPSILON, const int dataPartitionIndex)
{
    if (dataPartitionIndex == 0)
    {
        return buildSpheresPrimBuffer(pointsToSearch, dataset, DIM, EPSILON);
    }

    std::vector<Sphere> spheres;
    const unsigned int POINTS_TO_SEARCH = pointsToSearch.size();
    spheres.reserve(POINTS_TO_SEARCH);

    for (int i = 0; i < POINTS_TO_SEARCH; ++i)
    {
        const unsigned int index = DIM * pointsToSearch[i];
        spheres.push_back(Sphere{
            vec3f(dataset[index + dataPartitionIndex * DIMS_PER_DATA_PARTITION],
                  dataset[index + dataPartitionIndex * DIMS_PER_DATA_PARTITION + 1],
                  dataset[index + dataPartitionIndex * DIMS_PER_DATA_PARTITION + 2]),
            EPSILON
        });

    }

    return spheres;
}

std::vector<Sphere> buildCumDataPartitionSpheresPrimeBuffer(std::vector<uint32_t> pointsToSearch, float* dataset,
                                                            unsigned DIM, const float EPSILON,
                                                            const unsigned int NUM_DATA_PARTITIONS_)
{
    std::vector<Sphere> spheres;
    const unsigned int POINTS_TO_SEARCH = pointsToSearch.size();
    spheres.reserve(POINTS_TO_SEARCH * NUM_DATA_PARTITIONS_);
    for (int i = 0; i < NUM_DATA_PARTITIONS_; i++)
    {
        std::vector<Sphere> curDimSphere = buildDataPartitionSpheresPrimBuffer(pointsToSearch, dataset,
                                                                               DIM, EPSILON, i);
        spheres.insert(spheres.end(), curDimSphere.begin(), curDimSphere.end());
    }

    return spheres;
}

int getSampleSize(uint32_t N)
{
    int sampleSize;

    // Set default sampling size as 100
    // Ensure at least 1000 points are sampled in the dataset
    // If N < 1000, then sample all points in the dataset
    if (N <= 1000)
    {
        sampleSize = 1;
    }
    else if (N < 100000)
    {
        sampleSize = N / 1000;
    }
    else
    {
        sampleSize = 100;
    }

    return sampleSize;
}

float* reorderByDimensions(float* dataset, int N, int DIM, ReorderMode mode)
{
    double mean, devmean;
    std::vector<std::pair<float, int>> devDimPair;
    float* tempDataset = (float*)malloc(sizeof(float) * N * DIM);
    int sampleSize, sampledPoints;

    sampleSize = getSampleSize(N);
    sampledPoints = N / sampleSize;

    for (int i = 0; i < DIM; i++)
    {
        mean = 0.0;
        for (int j = 0; j < N; j += sampleSize)
        {
            mean += dataset[DIM * j + i];
        }

        mean /= sampledPoints;
        devmean = 0;

        for (int j = 0; j < N; j += sampleSize)
        {
            devmean += pow(dataset[DIM * j + i] - mean, 2);
        }

        devmean /= sampledPoints;
        devDimPair.emplace_back(sqrt(devmean), i);
    }

    mean = 0;
    devmean = 0;

    for (int i = 0; i < DIM; i++)
    {
        mean += devDimPair[i].first;
    }
    mean /= DIM;

    for (int i = 0; i < DIM; i++)
    {
        devmean += pow(devDimPair[i].first - mean, 2);
    }
    devmean /= DIM;

    double coeffOfVariance = (sqrt(devmean) / mean) * 100;
    printf("Mean = %.3lf SD = %.3lf CV = %lf\n", mean, sqrt(devmean), coeffOfVariance);


    if (mode == REORDER_RANDOM)
    {
        std::mt19937 rng(REORDER_RANDOM_SEED);
        std::shuffle(devDimPair.begin(), devDimPair.end(), rng);
        printf("Dimensions are randomly selected (seed=%u)\n", REORDER_RANDOM_SEED);
    }
    else if (mode == REORDER_LOWEST_VARIANCE)
    {
        sort(devDimPair.begin(), devDimPair.end(), std::less<>());
        puts("Dimensions are reordered by lowest variance");
    }
    else 
    {
        sort(devDimPair.begin(), devDimPair.end(), std::greater<>());
        puts("Dimensions are reordered by highest variance");
    }

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            tempDataset[i * DIM + j] = dataset[i * DIM + devDimPair[j].second];
        }
    }
    free(dataset);
    return tempDataset;
    
}


float* readData(char* fileName, unsigned int* N, unsigned int* dim)
{
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        printf("Unable to open file\n");
        exit(1);
    }

    std::string line;
    unsigned int lines = 0;
    while (std::getline(file, line))
    {
        if (!line.empty()) lines++;
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    while (std::getline(file, line) && line.empty()) {} 
    unsigned int fieldCount = 0;
    {
        const char* p = line.c_str();
        char* end = nullptr;
        while (true)
        {
            strtof(p, &end);
            if (end == p) break;
            ++fieldCount;
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        }
    }
    *N = lines;
    *dim = fieldCount;

    file.clear();
    file.seekg(0, std::ios::beg);

    float* dataset = static_cast<float*>(malloc(sizeof(float) * (size_t)(*N) * (*dim)));
    if (!dataset)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    size_t count = 0;
    size_t totalElements = (size_t)(*N) * (*dim);

    while (std::getline(file, line) && count < totalElements)
    {
        if (line.empty()) continue;
        const char* p = line.c_str();
        char* end = nullptr;
        for (unsigned int col = 0; col < *dim; ++col)
        {
            float value = strtof(p, &end);
            if (end == p) break; // no more numbers on this row
            dataset[count++] = value;
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\t') ++p; // skip separators
        }
    }

    return dataset;
}


std::unordered_set<uint32_t> findClusteredPoints(std::vector<keyValData>& sortedKeyValDataset)
{
    std::unordered_set<uint32_t> clusteredPIDs;

    uint32_t currentDim = 0;

    if (sortedKeyValDataset.empty()) return clusteredPIDs;

    std::vector<uint32_t> currentCluster;
    currentCluster.push_back(sortedKeyValDataset[0].pid);

    for (size_t i = 1; i < sortedKeyValDataset.size(); ++i)
    {
        double diff = sortedKeyValDataset[i].point[currentDim] - sortedKeyValDataset[i - 1].point[currentDim];

        if (diff <= THRESHOLD)
        {
            currentCluster.push_back(sortedKeyValDataset[i].pid);
        }
        else
        {
            if (currentCluster.size() >= MIN_CLUSTER_POINTS)
            {
                for (uint32_t pid : currentCluster)
                {
                    clusteredPIDs.insert(pid);
                }
            }
            currentCluster.clear();
            currentCluster.push_back(sortedKeyValDataset[i].pid);
        }
    }

    if (currentCluster.size() >= MIN_CLUSTER_POINTS)
    {
        for (uint32_t pid : currentCluster)
        {
            clusteredPIDs.insert(pid);
        }
    }

    return clusteredPIDs;
}

struct fixedQueryVariableDataPointPartitions generateQueryDataPartitionsOnDemand(
    std::vector<keyValData>& sortedKeyValDataset,
    std::unordered_set<uint32_t> clusteredPoints,
    uint32_t* queryStartIndex,
    float* dataset,
    unsigned int NPOINTS,
    uint32_t DIM,
    const float EPSILON)
{
    std::vector<uint32_t> curQueryPartition;
    std::vector<uint32_t> curDataPartition;
    std::unordered_set<uint32_t> curDataPartitionSet;

    uint32_t currentDim = 0;

    int startIndex = *queryStartIndex;

    while (*queryStartIndex < NPOINTS && curQueryPartition.size() < MAX_QUERY_POINTS)
    {
        uint32_t pid = sortedKeyValDataset[*queryStartIndex].pid;
        if (clusteredPoints.find(pid) == clusteredPoints.end())
        {
            curQueryPartition.push_back(pid);
        }
        // IMPORTANT - Even if the current query point is not included, it should be included in the data space
        curDataPartitionSet.insert(pid);
        (*queryStartIndex) += 1;
    }

    uint32_t endIndex = *queryStartIndex;

    if (curQueryPartition.empty())
    {
        return fixedQueryVariableDataPointPartitions();
    }

    float queryStartRange = dataset[curQueryPartition.front() * DIM] - EPSILON;
    float queryEndRange = dataset[curQueryPartition.back() * DIM] + EPSILON;

    // SEARCH LEFT
    for (int i = startIndex; i >= 0; i--)
    {
        auto& point = sortedKeyValDataset[i];
        if (point.point[currentDim] >= queryStartRange)
        {
            curDataPartitionSet.insert(point.pid);
        }
        else
        {
            break;
        }
    }

    // SEARCH RIGHT
    for (uint32_t i = endIndex; i < NPOINTS; i++)
    {
        auto& point = sortedKeyValDataset[i];
        if (point.point[currentDim] <= queryEndRange)
        {
            curDataPartitionSet.insert(point.pid);
        }
        else
        {
            break;
        }
    }

    // Update query index
    curDataPartition.insert(curDataPartition.begin(), curDataPartitionSet.begin(), curDataPartitionSet.end());

    std::vector<std::vector<uint32_t>> variableDataSpace;
    // Now make queries fixed but data space variable
    if (curDataPartition.size() < MAX_DATA_PARTITION_POINTS)
    {
        variableDataSpace.push_back(curDataPartition);
    }
    else
    {
        const uint32_t curDataSpaceSize = curDataPartition.size();

        const uint32_t numSubPartitions = std::ceil(static_cast<float>(curDataSpaceSize) / MAX_DATA_PARTITION_POINTS);

        const uint32_t pointsPerPartition = std::ceil(static_cast<float>(curDataSpaceSize) / numSubPartitions);

        for (uint32_t i = 0; i < numSubPartitions; i++)
        {
            uint32_t start = i * pointsPerPartition;
            if (i != numSubPartitions - 1)
            {
                uint32_t end = start + pointsPerPartition;
                variableDataSpace.emplace_back(curDataPartition.begin() + start, curDataPartition.begin() + end);
            }
            else
            {
                variableDataSpace.emplace_back(curDataPartition.begin() + start, curDataPartition.end());
            }
        }
    }

    return {curQueryPartition, variableDataSpace};
}

// Generate work for sampled points
struct curQueryDataPointPartitions fetchSampledQueryAndDataSpace(uint32_t N)
{
   
    std::vector<uint32_t> queries(N);
    std::vector<uint32_t> dataPartitions(N);

    uint32_t i = 0;
    #pragma omp parallel for private(i) shared(queries, dataPartitions) schedule(static)
    for(i = 0 ; i < N ; i++)
    {
        queries[i] = i;
        dataPartitions[i] = i;
    }

    struct curQueryDataPointPartitions sampledQueryDataPartition(queries, dataPartitions);

    return sampledQueryDataPartition;
}

std::vector<uint32_t> fetchNextQueries(std::vector<keyValData>& sortedKeyValDataset, uint32_t startingIndex,
                                       uint32_t batchSize)
{
    size_t endIndex = std::min((size_t)(startingIndex + batchSize), sortedKeyValDataset.size());

    std::vector<uint32_t> queries;

    for (auto itr = sortedKeyValDataset.begin() + startingIndex;
         itr != sortedKeyValDataset.begin() + endIndex;
         ++itr)
    {
        queries.push_back(itr->pid);
    }


    return queries;
}


std::vector<uint32_t> fetchNextSortedQueries(std::vector<uint32_t> sortedQueries, uint32_t startingIndex,
                                             uint32_t batchSize)
{
    size_t endIndex = std::min((size_t)(startingIndex + batchSize), sortedQueries.size());

    std::vector<uint32_t> queries;

    for (auto itr = sortedQueries.begin() + startingIndex;
         itr != sortedQueries.begin() + endIndex; ++itr)
    {
        queries.push_back(*itr);
    }

    return queries;
}

struct queryDataPointPartitions insertAllWithIdx(std::vector<keyValData>& keyValDataset, unsigned int NPOINTS,
                                                 unsigned int IDIM,
                                                 unsigned int numMaxTreeLevels,
                                                 std::vector<std::vector<unsigned int>>& partitions,
                                                 const float EPSILON)
{
    exit(1);
}

void setupModuleAndContext(const char* deviceCode_ptx, OWLContext& context, OWLModule& module)
{
    // Set up OWL context
    double timeInitialSetupStart = omp_get_wtime();
    context = owlContextCreate(nullptr, 1);

    // Set up OWL module
    module = owlModuleCreate(context, deviceCode_ptx);
    double timeInitialSetupEnd = omp_get_wtime();

    double totalTimeInitialSetup = (timeInitialSetupEnd - timeInitialSetupStart) * 1000;
    std::cout << "Time for initial setup of context and module is " << totalTimeInitialSetup << " (ms)" << std::endl;
}


void setupIAS(OWLGeom& SpheresGeom, OWLGroup& handle, OWLGroup& spheresGroups, OWLContext context, OWLRayGen rayGen)
{
    // Free memory after use
    uint8_t* visibilityMasks = new uint8_t[MAX_VISIBILITY_MASKS];

    OWLGeom userGeoms[] = {SpheresGeom};
    (spheresGroups) = owlUserGeomGroupCreate(context, 1, userGeoms);
    owlGroupBuildAccel((spheresGroups));
    uint32_t instanceID = 0;
    visibilityMasks[0] = 1 << 0;
    

    // Free handle after use
    handle = owlInstanceGroupCreate(context, 1, &spheresGroups, &instanceID);


    owlInstanceGroupSetVisibilityMasks(handle, visibilityMasks);
    owlGroupBuildAccel(handle);

    owlRayGenSetGroup(rayGen, "handle", handle);

    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}


void setupVarsForShaderPrograms(OWLContext context, OWLModule module, OWLGeom& userSpheresGeom,
                                OWLRayGen& rayGen,
                                OWLParams& lp)
{
    // FOR SPHERES GEOM
    OWLVarDecl SpheresGeomVars[] = {
        {"prims", OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom, prims)},
        {
            /* sentinel to mark end of list sphere*/
        }
    };

    OWLGeomType SpheresGeomType = owlGeomTypeCreate(
        context, OWL_GEOMETRY_USER, sizeof(SpheresGeom), SpheresGeomVars, -1);

    // AH, BOUNDS, AND INTERSECTION PROGRAMS
    owlGeomTypeSetIntersectProg(SpheresGeomType, 0, module, "SpheresIntersect");
    owlGeomTypeSetBoundsProg(SpheresGeomType, module, "SpheresBounds");
    owlGeomTypeSetAnyHit(SpheresGeomType, 0, module, "SpheresAnyHit");

    userSpheresGeom = owlGeomCreate(context, SpheresGeomType);
        
    owlBuildPrograms(context);

    // VARS FOR LAUNCH PARAMS
    OWLVarDecl LaunchParamsVars[] = {
        {"maxCandidatePointsPerQueryPoint", OWL_UINT, OWL_OFFSETOF(LaunchParams, maxCandidatePointsPerQueryPoint)},
        {"candidatePoints", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, candidatePoints)},
        {"spheres", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, spheres)},
        {"totalCandidateCount", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, totalCandidateCount)},
        {"queryIndices", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, queryIndices)},
        {"intersectionCountTest", OWL_UINT, OWL_OFFSETOF(LaunchParams, intersectionCountTest)},
        {"intersectionCount", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, intersectionCount)},
        {"primitiveIntersectionCount", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, primitiveIntersectionCount)},
        {"primitivePosTowrite", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, primitivePosTowrite)},
        {
            /* sentinel to mark end of list sphere*/
        }
    };


    lp = owlParamsCreate(context, sizeof(LaunchParams), LaunchParamsVars, -1);

    // RAY GENERATION VARS
    OWLVarDecl rayGenVars[] = {
        {"handle", OWL_GROUP, OWL_OFFSETOF(RayGenData, handle)},
        {
            /* sentinel to mark end of list */
        }
    };

    rayGen = owlRayGenCreate(context, module, "rayGen",
                             sizeof(RayGenData), rayGenVars, -1);
}

void setInterSectionTestAndDataPartitions(OWLParams& lp, unsigned int INTERSECTION_TEST,
                                          unsigned int NUM_DATA_PARTITIONS_)
{
    // owlParamsSet1ui(lp, "INTERSECTION_TEST", INTERSECTION_TEST);
    // owlParamsSet1ui(lp, "NUM_DATA_PARTITION", NUM_DATA_PARTITIONS_);
}


// Construct a bounding sphere for a group of points
Sphere constructBoundingSphere(std::vector<SphereData>& pointsGroup)
{
    if (pointsGroup.size() == 1)
    {
        return {pointsGroup[0].sphere.center, pointsGroup[0].sphere.radius};
    }

    if (pointsGroup.size() == 2)
    {
        const vec3f c0 = pointsGroup[0].sphere.center;
        const vec3f c1 = pointsGroup[1].sphere.center;
        const float r0 = pointsGroup[0].sphere.radius;
        const float r1 = pointsGroup[1].sphere.radius;

        const float diffX = c1.x - c0.x;
        const float diffY = c1.y - c0.y;
        const float diffZ = c1.z - c0.z;
        const float d = sqrtf(diffX * diffX + diffY * diffY + diffZ * diffZ);

        // If one sphere contains the other, return the larger one.
        if (r0 >= d + r1) return {c0, r0};
        if (r1 >= d + r0) return {c1, r1};

        // Partial overlap / disjoint: minimal enclosing sphere
        const float newRadius = 0.5f * (d + r0 + r1);

        vec3f center;
        if (d > 0.0f)
        {
            const float k = (newRadius - r0) / d; // shift from c0 toward c1
            center = {
                c0.x + k * diffX,
                c0.y + k * diffY,
                c0.z + k * diffZ
            };
        }
        else
        {
            // Same center; just pick either and set radius to max(r0, r1)
            center = c0;
        }
        return {center, newRadius};
    }

    float xMin = +FLT_MAX;
    float yMin = +FLT_MAX;
    float zMin = +FLT_MAX;

    float xMax = -FLT_MAX;
    float yMax = -FLT_MAX;
    float zMax = -FLT_MAX;

    for (auto& [sphere, _] : pointsGroup)
    {
        auto center = sphere.center;

        if (center.x < xMin) xMin = center.x;
        if (center.y < yMin) yMin = center.y;
        if (center.z < zMin) zMin = center.z;

        if (center.x > xMax) xMax = center.x;
        if (center.y > yMax) yMax = center.y;
        if (center.z > zMax) zMax = center.z;
    }

    vec3f pMin = {xMin, yMin, zMin};
    vec3f pMax = {xMax, yMax, zMax};

    vec3f initialCenter = {
        0.5f * (pMin.x + pMax.x),
        0.5f * (pMin.y + pMax.y),
        0.5f * (pMin.z + pMax.z)
    };

    float dx = pMax.x - pMin.x;
    float dy = pMax.y - pMin.y;
    float dz = pMax.z - pMin.z;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    float initialRadius = 0.5f * dist;

    // Ritter grow pass - bounding sphere should contain the point sphere
    for (auto& [sphere, _] : pointsGroup)
    {
        auto center = sphere.center;
        auto radius = sphere.radius; // member sphere radius

        // vector from current enclosing center -> member center
        float dx = center.x - initialCenter.x;
        float dy = center.y - initialCenter.y;
        float dz = center.z - initialCenter.z;

        float d2 = dx * dx + dy * dy + dz * dz;
        float d = sqrtf(d2);
        float R = initialRadius;
        float r = radius;

        // --- Case 1: member sphere already inside current sphere
        if (d + r <= R)
        {
            // nothing to do
        }
        // --- Case 2: member sphere strictly contains the current sphere
        else if (r >= R + d)
        {
            initialCenter = center;
            initialRadius = r;
        }
        // --- Case 3: partial overlap / disjoint
        else
        {
            // New radius is half the span covering both spheres
            float newR = 0.5f * (R + d + r);

            // Shift center toward center
            float k = (newR - R) / d;
            initialCenter.x += k * dx;
            initialCenter.y += k * dy;
            initialCenter.z += k * dz;

            // set radius exactly to the post-update distance + r
            float vx = center.x - initialCenter.x;
            float vy = center.y - initialCenter.y;
            float vz = center.z - initialCenter.z;
            float d_after = sqrtf(vx * vx + vy * vy + vz * vz);
            initialRadius = d_after + r;
        }
    }

    return {initialCenter, initialRadius};
}

Sphere constructBoundingSphereWithTwoSpheres(Sphere a, Sphere b)
{
    const vec3f c0 = a.center;
    const vec3f c1 = b.center;
    const float r0 = a.radius;
    const float r1 = b.radius;

    const float diffX = c1.x - c0.x;
    const float diffY = c1.y - c0.y;
    const float diffZ = c1.z - c0.z;
    const float d = sqrtf(diffX * diffX + diffY * diffY + diffZ * diffZ);

    // If one sphere contains the other, return the larger one.
    if (r0 >= d + r1) return {c0, r0};
    if (r1 >= d + r0) return {c1, r1};

    // Partial overlap / disjoint: minimal enclosing sphere
    const float newRadius = 0.5f * (d + r0 + r1);

    vec3f center;
    if (d > 0.0f)
    {
        const float k = (newRadius - r0) / d; // shift from c0 toward c1
        center = {
            c0.x + k * diffX,
            c0.y + k * diffY,
            c0.z + k * diffZ
        };
    }
    else
    {
        // Same center; just pick either and set radius to max(r0, r1)
        center = c0;
    }
    return {center, newRadius};
}

void writeSphereDataToFileForPlots(const std::vector<boundingSphereGroup>& sphereGroups)
{
    std::string filePath = "plots/data_" +
        std::to_string(SMALL_GROUP_THRESHOLD) + "_" + std::to_string(LARGE_GROUP_THRESHOLD) + "_" ".txt";

    FILE* fptr = fopen(filePath.c_str(), "w");
    if (!fptr)
    {
        perror("Failed to open file");
        return;
    }

    printf("\n Writing %zu lines to file", sphereGroups.size());
    for (auto& [boundingSphere, group] : sphereGroups)
    {
        fprintf(fptr, "%f %f %f %f %zu\n",
                boundingSphere.center.x,
                boundingSphere.center.y,
                boundingSphere.center.z,
                boundingSphere.radius,
                group.size());
    }

    fclose(fptr);
}

std::vector<boundingSphereGroup> constructDistinctNonOverlappingSpheres(std::vector<boundingSphereGroup>& sphereGroups)
{
    std::vector<bool> keep(sphereGroups.size(), true);

    for (size_t i = 0; i < sphereGroups.size(); i++)
    {
        for (size_t j = 0; j < sphereGroups.size(); j++)
        {
            if (i == j || !keep[i] || !keep[j]) continue;

            vec3f center1 = sphereGroups[i].sphere.center;
            vec3f center2 = sphereGroups[j].sphere.center;

            float radius1 = sphereGroups[i].sphere.radius;
            float radius2 = sphereGroups[j].sphere.radius;

            float diffX = center1.x - center2.x;
            float diffY = center1.y - center2.y;
            float diffZ = center1.z - center2.z;

            float distance = diffX * diffX + diffY * diffY + diffZ * diffZ;

            float diff = radius1 - radius2;

            if (diff > 0 && distance <= diff * diff)
            {
                sphereGroups[i].sphereGroup.insert(
                    sphereGroups[i].sphereGroup.end(),
                    sphereGroups[j].sphereGroup.begin(),
                    sphereGroups[j].sphereGroup.end()
                );

                keep[j] = false;
            }
        }
    }

    std::vector<boundingSphereGroup> filtered;
    for (size_t i = 0; i < sphereGroups.size(); i++)
    {
        if (keep[i]) filtered.push_back(sphereGroups[i]);
    }

    return filtered;
}

std::vector<boundingSphereGroup> constructDynamicGroupedSpheres(std::vector<SphereData> allPointSpheresData)
{
    std::vector<SphereData> curSphereGroup;
    Sphere curBoundingSphere;
    float curRadius;
    std::vector<boundingSphereGroup> sphereGroups;

    std::vector<Sphere> boundingSpheres;

    curSphereGroup.push_back(allPointSpheresData[0]);
    curBoundingSphere = allPointSpheresData[0].sphere;
    curRadius = curBoundingSphere.radius;

    uint32_t spheresWithSinglePoint = 0;

    for (uint32_t i = 1; i < allPointSpheresData.size(); i++)
    {
        Sphere newBoundingSphere = constructBoundingSphereWithTwoSpheres(
            curBoundingSphere, allPointSpheresData[i].sphere);


        float newVolume = newBoundingSphere.radius * newBoundingSphere.radius * newBoundingSphere.radius;
        float oldVolume = curRadius * curRadius * curRadius;

        float delta = (newVolume - oldVolume) / oldVolume * 100;

        float THRESHOLD = curSphereGroup.size() <= POINTS_IN_SMALL_GROUP
                              ? SMALL_GROUP_THRESHOLD
                              : LARGE_GROUP_THRESHOLD;


        if (delta < THRESHOLD)
        {
            curSphereGroup.push_back(allPointSpheresData[i]);
            curBoundingSphere = newBoundingSphere;
            curRadius = curBoundingSphere.radius;
        }
        else
        {
            Sphere rittersBoundingSphere = constructBoundingSphere(curSphereGroup);
            sphereGroups.push_back({rittersBoundingSphere, curSphereGroup});

            if (curSphereGroup.size() == 1)
            {
                spheresWithSinglePoint += 1;
            }

            curSphereGroup.clear();
            curBoundingSphere = allPointSpheresData[i].sphere;
            curRadius = curBoundingSphere.radius;
            curSphereGroup.push_back(allPointSpheresData[i]);
        }
    }

    if (!curSphereGroup.empty())
    {
        Sphere rittersBoundingSphere = constructBoundingSphere(curSphereGroup);
        sphereGroups.push_back({rittersBoundingSphere, curSphereGroup});
    }

    printf("\nBEFORE -  Spheres with single point are %u out of %zu spheres", spheresWithSinglePoint,
           sphereGroups.size());

    auto filteredSphereGroups = constructDistinctNonOverlappingSpheres(sphereGroups);

    printf("\n AFTER FILTERING - Filteres sphere count is %zu", filteredSphereGroups.size());
    return filteredSphereGroups;
}

Sphere getBoundingSphere(boundingSphereGroupBestMate a, boundingSphereGroupBestMate b)
{
    vec3f centerA = a.sphereGroup.sphere.center;
    vec3f centerB = b.sphereGroup.sphere.center;

    float radiusA = a.sphereGroup.sphere.radius;
    float radiusB = b.sphereGroup.sphere.radius;

    float diffX = centerA.x - centerB.x;
    float diffY = centerA.y - centerB.y;
    float diffZ = centerA.z - centerB.z;

    float distance = sqrtf(diffX * diffX + diffY * diffY + diffZ * diffZ);

    vec3f newCenter = vec3f((centerA.x + centerB.x) / 2, (centerA.y + centerB.y) / 2, (centerA.z + centerB.z) / 2);
    float newRadius = (distance + radiusA + radiusB) / 2;

    return {newCenter, newRadius};
}

void mergeGroupsCPU(std::vector<boundingSphereGroupBestMate>& groups,
                    std::vector<BoundingSphereGroupGPU>& gpuResults)
{
    size_t N = groups.size();

    std::vector<boundingSphereGroupBestMate> newGroups;

    std::vector<BoundingSphereGroupCPU> cpuResults(gpuResults.size());

    size_t i;
#pragma omp parallel private(i)
    for (i = 0; i < gpuResults.size(); i++)
    {
        cpuResults[i].bestMateId = gpuResults[i].bestMateId;
        cpuResults[i].center.x = gpuResults[i].center.x;
        cpuResults[i].center.y = gpuResults[i].center.y;
        cpuResults[i].center.z = gpuResults[i].center.z;
        cpuResults[i].radius = gpuResults[i].radius;
        cpuResults[i].bestMateCost = gpuResults[i].bestMateCost;
    }

    // std::sort(cpuResults.begin(), cpuResults.end(), [](BoundingSphereGroupCPU a, BoundingSphereGroupCPU b) {return a.bestMateCost<b.bestMateCost;});

    thrust::sort_by_key(cpuResults.begin(), cpuResults.end(), groups.begin(),
                        [](BoundingSphereGroupCPU& a, BoundingSphereGroupCPU& b)
                        {
                            return a.bestMateCost < b.bestMateCost;
                        });

    const size_t minGroupCount = BALL_MIN_FRACTION * cpuResults.size();

    uint32_t diff = cpuResults.size() - minGroupCount;
    uint32_t itr = 0;
    while (cpuResults.size() > minGroupCount)
    {
        
        itr++;

        auto& top = cpuResults[0];
        auto& bestMateTop = cpuResults[top.bestMateId];

        BoundingSphereGroupCPU mergedTop;

        mergedTop.center.x = (top.center.x + bestMateTop.center.x) * 0.5f;
        mergedTop.center.y = (top.center.y + bestMateTop.center.y) * 0.5f;
        mergedTop.center.z = (top.center.z + bestMateTop.center.z) * 0.5f;
        mergedTop.radius = top.bestMateCost;

        cpuResults.erase(cpuResults.begin() + top.bestMateId);
        cpuResults.erase(cpuResults.begin());

        // Remove from groups as well
        std::vector<SphereData> mergedMembers;

        mergedMembers.insert(mergedMembers.end(), groups.begin()->sphereGroup.sphereGroup.begin(),
                             groups.begin()->sphereGroup.sphereGroup.end());

        mergedMembers.insert(mergedMembers.end(), groups[top.bestMateId].sphereGroup.sphereGroup.begin(),
                             groups[top.bestMateId].sphereGroup.sphereGroup.end());

        groups.erase(groups.begin() + top.bestMateId);
        groups.erase(groups.begin());

        int bestMate = -1;
        float bestMateCost = FLT_MAX;


        findBestMateGPUOneQuery(cpuResults.data(), mergedTop.center, mergedTop.radius, cpuResults.size(),
                                &bestMateCost, &bestMate);

        assert(bestMate!=-1);

        mergedTop.bestMateCost = bestMateCost;
        mergedTop.bestMateId = bestMate;

        Sphere currentSphere;
        currentSphere.center.x = mergedTop.center.x;
        currentSphere.center.y = mergedTop.center.y;
        currentSphere.center.z = mergedTop.center.z;

        boundingSphereGroup cur = {currentSphere, mergedMembers};

        groups.emplace_back(cur, 0);

        cpuResults.push_back(mergedTop);

        thrust::sort_by_key(cpuResults.begin(), cpuResults.end(), groups.begin(),
                            [](BoundingSphereGroupCPU& a, BoundingSphereGroupCPU& b)
                            {
                                return a.bestMateCost < b.bestMateCost;
                            });
    }
}


std::vector<boundingSphereGroup> constructBallTreeGroupedSpheres(
    std::vector<SphereData> allPointSpheresData)
{
    std::vector<boundingSphereGroup> finalGroups;
    std::vector<boundingSphereGroupBestMate> groupsBestMates;

    for (auto& p : allPointSpheresData)
    {
        boundingSphereGroup g{p.sphere, {p}};
        groupsBestMates.emplace_back(g, 0);
    }

    size_t N = groupsBestMates.size();
    uint32_t currentLevel = 0;

    // Not a loop priority queue approach
    // while (groupsBestMates.size() > 1 && currentLevel < MAX_BALL_LEVELS)
    {
        std::vector<BoundingSphereGroupGPU> gpuGroups(groupsBestMates.size());
        for (size_t i = 0; i < groupsBestMates.size(); i++)
        {
            gpuGroups[i].center = {
                groupsBestMates[i].sphereGroup.sphere.center.x,
                groupsBestMates[i].sphereGroup.sphere.center.y,
                groupsBestMates[i].sphereGroup.sphere.center.z
            };
            gpuGroups[i].radius = groupsBestMates[i].sphereGroup.sphere.radius;
            gpuGroups[i].bestMateId = -1;
        }

        findBestMateGPU(gpuGroups.data(), gpuGroups.size());


        // --- Merge on CPU
        mergeGroupsCPU(groupsBestMates, gpuGroups);

        currentLevel++;
    }

    // Collect final groups
    for (auto& g : groupsBestMates)
        finalGroups.push_back(g.sphereGroup);

    return finalGroups;
}

std::pair<uint32_t, float> findCost(const std::vector<SphereData>& allPointSpheresData)
{
    uint32_t N = allPointSpheresData.size();
    if (N <= 1)
    {
        return {0u, 0.0f};
    }

    std::vector<float> costLeft(N);
    std::vector<float> costRight(N);

    Sphere boundingLeft = allPointSpheresData[0].sphere;
    costLeft[0] = boundingLeft.radius;
    for (uint32_t i = 1; i < N; i++)
    {
        boundingLeft = constructBoundingSphereWithTwoSpheres(boundingLeft, allPointSpheresData[i].sphere);
        costLeft[i] = boundingLeft.radius;
    }

    Sphere boundingRight = allPointSpheresData[N - 1].sphere;
    costRight[N - 1] = boundingRight.radius;
    for (int i = N - 2; i >= 0; i--)
    {
        boundingRight = constructBoundingSphereWithTwoSpheres(boundingRight, allPointSpheresData[i].sphere);
        costRight[i] = boundingRight.radius;
    }

    float bestCost = FLT_MAX;
    uint32_t splitIndex = 0;

    for (uint32_t i = 0; i < N - 1; i++)
    {
        float rL = costLeft[i];
        float rR = costRight[i + 1];
        float cost = 0.5f * (rL * rL * rL + rR * rR * rR);

        if (cost < bestCost)
        {
            bestCost = cost;
            splitIndex = i;
        }
    }

    return {splitIndex, bestCost};
}

std::pair<std::vector<SphereData>, std::vector<SphereData>> splitGroup(std::vector<SphereData> allPointSpheresData)
{
    std::vector<SphereData> allPointSpheresDataX(allPointSpheresData.begin(), allPointSpheresData.end());
    std::vector<SphereData> allPointSpheresDataY(allPointSpheresData.begin(), allPointSpheresData.end());
    std::vector<SphereData> allPointSpheresDataZ(allPointSpheresData.begin(), allPointSpheresData.end());

    uint32_t splitIndexX, splitIndexY, splitIndexZ;
    float costX, costY, costZ;

#pragma omp sections
    {
#pragma omp section
        {
            // Sort on x
            std::sort(allPointSpheresDataX.begin(), allPointSpheresDataX.end(),
                      [](const SphereData& a, const SphereData& b)
                      {
                          return a.sphere.center.x < b.sphere.center.x;
                      });

            auto pair = findCost(allPointSpheresDataX);
            splitIndexX = pair.first;
            costX = pair.second;
        }

#pragma omp section
        {
            // Cost fn on Y
            std::sort(allPointSpheresDataY.begin(), allPointSpheresDataY.end(),
                      [](const SphereData& a, const SphereData& b)
                      {
                          return a.sphere.center.y < b.sphere.center.y;
                      });

            auto pair = findCost(allPointSpheresDataY);
            splitIndexY = pair.first;
            costY = pair.second;
        }

#pragma omp section
        {
            // Cost fn on Z
            std::sort(allPointSpheresDataZ.begin(), allPointSpheresDataZ.end(),
                      [](const SphereData& a, const SphereData& b)
                      {
                          return a.sphere.center.z < b.sphere.center.z;
                      });

            auto pair = findCost(allPointSpheresDataZ);
            splitIndexZ = pair.first;
            costZ = pair.second;
        }
    }


    if (costX <= costY && costX <= costZ)
    {
        return std::make_pair(
            std::vector<SphereData>(allPointSpheresDataX.begin(), allPointSpheresDataX.begin() + splitIndexX + 1),
            std::vector<SphereData>(allPointSpheresDataX.begin() + splitIndexX + 1, allPointSpheresDataX.end()));
    }

    if (costY <= costX && costY <= costZ)
    {
        return std::make_pair(
            std::vector<SphereData>(allPointSpheresDataY.begin(), allPointSpheresDataY.begin() + splitIndexY + 1),
            std::vector<SphereData>(allPointSpheresDataY.begin() + splitIndexY + 1, allPointSpheresDataY.end()));
    }


    return std::make_pair(
        std::vector<SphereData>(allPointSpheresDataZ.begin(), allPointSpheresDataZ.begin() + splitIndexZ + 1),
        std::vector<SphereData>(allPointSpheresDataZ.begin() + splitIndexZ + 1, allPointSpheresDataZ.end()));
    
}

std::vector<std::vector<SphereData>> constructTopDownBallTree(std::vector<SphereData> allPointSpheresData,
                                                              int currentLevel)
{
     if (currentLevel >= MAX_BALL_LEVELS || allPointSpheresData.size() <= MIN_BALL_LEAF_SIZE)
    // if (allPointSpheresData.size() <= MIN_BALL_LEAF_SIZE)
    {
        return {allPointSpheresData};
    }

    auto [leftSplitData, rightSplitData] = splitGroup(allPointSpheresData);

    assert(!leftSplitData.empty());
    assert(!rightSplitData.empty());

    // Recursive call for sub splits
    std::vector<std::vector<SphereData>> res;

    std::vector<std::vector<SphereData>> leftGroup;
    std::vector<std::vector<SphereData>> rightGroup;

// #pragma omp sections
    {
// #pragma omp section
        {
            if (!leftSplitData.empty())
            {
                leftGroup = constructTopDownBallTree(leftSplitData, currentLevel + 1);

            }
        }
// #pragma omp section
        {
            if (!rightSplitData.empty())
            {
                rightGroup = constructTopDownBallTree(rightSplitData, currentLevel + 1);

            }
        }
    }

    res.insert(res.end(), leftGroup.begin(), leftGroup.end());
    res.insert(res.end(), rightGroup.begin(), rightGroup.end());

    return res;
}

std::vector<boundingSphereGroup> constructTopDownBallTreeGroups(
    std::vector<SphereData> allPointSpheresData)
{
    std::vector<boundingSphereGroup> res;
    auto ballTree = constructTopDownBallTree(allPointSpheresData, 0);

    res.reserve(ballTree.size());

    for(auto &group : ballTree)
    {
        res.push_back({constructBoundingSphere(group), group});
    }

    return res;

}

std::vector<boundingSphereGroup> constructGridGroups(
    std::vector<SphereData> allPointSpheresData)
{
    const float GRID_WIDTH = allPointSpheresData[0].sphere.radius;

    std::unordered_map<std::string, std::vector<SphereData>> grid;

    for(auto & i : allPointSpheresData)
    {
        float x  = i.sphere.center.x;
        float y  = i.sphere.center.y;
        float z  = i.sphere.center.z;

        uint32_t gridIdx = floor(x/GRID_WIDTH);
        uint32_t gridIdy = floor(y/GRID_WIDTH);
        uint32_t gridIdz = floor(z/GRID_WIDTH);

        std::string gridIndex = std::to_string(gridIdx) + std::to_string(gridIdy) + std::to_string(gridIdz);

        if(grid.find(gridIndex) == grid.end())
        {
            grid[gridIndex] = std::vector<SphereData>(1, i);
        }else
        {
            grid[gridIndex].push_back(i);
        }
    }

    std::vector<boundingSphereGroup> result;

    for(auto&[_, curSphereData]: grid)
    {
        Sphere curSphere = constructBoundingSphere(curSphereData);
        boundingSphereGroup a = {curSphere, curSphereData};
        result.push_back(a);
    }

    result = constructDistinctNonOverlappingSpheres(result);


    // Compute morton for locality
    std::vector<uint32_t> mortonCode;
    for(auto & i : result)
    {
        uint32_t x = static_cast<uint32_t>(i.sphere.center.x * 1023);
        uint32_t y = static_cast<uint32_t>(i.sphere.center.y * 1023);
        uint32_t z = static_cast<uint32_t>(i.sphere.center.z * 1023);

        mortonCode.push_back(EncodeMorton3(x, y, z));
    }

    thrust::sort_by_key(mortonCode.begin(), mortonCode.end(), result.begin());

    return result;
}


void sortBasedOnX(std::vector<SphereData> &curData)
{
    thrust::sort(curData.begin(), curData.end(), [](const SphereData&a, const SphereData&b)
    {
        return a.sphere.center.x < b.sphere.center.x;
    });
}

void sortBasedOnY(std::vector<SphereData> &curData)
{
    thrust::sort(curData.begin(), curData.end(), [](const SphereData&a, const SphereData&b)
    {
        return a.sphere.center.y < b.sphere.center.y;
    });
}

void sortBasedOnZ(std::vector<SphereData> &curData)
{
    thrust::sort(curData.begin(), curData.end(), [](const SphereData&a, const SphereData&b)
    {
        return a.sphere.center.z < b.sphere.center.z;
    });
}

// Multi core cpu implementation
std::vector<std::vector<SphereData>> constructKDGroupsRec(
    std::vector<SphereData> allPointSpheresData, int currentLevel)
{
    uint32_t N = allPointSpheresData.size();

    if(currentLevel >= MAX_KD_LEVELS || N <= 1)
    {
        return { allPointSpheresData };
    }

    if(currentLevel%3 == 0)
    {
        sortBasedOnX(allPointSpheresData);
    }else if (currentLevel %3 == 1)
    {
        sortBasedOnY(allPointSpheresData);
    }else
    {
        sortBasedOnZ(allPointSpheresData);
    }

    uint32_t mid = N/2;

    std::vector<SphereData> left(allPointSpheresData.begin(), allPointSpheresData.begin() + mid);
    std::vector<SphereData> right(allPointSpheresData.begin() + mid, allPointSpheresData.end());

    std::vector<std::vector<SphereData>> leftPartition = constructKDGroupsRec(
    left, currentLevel + 1);

    std::vector<std::vector<SphereData>> rightPartition = constructKDGroupsRec(
    right, currentLevel + 1);

    std::vector<std::vector<SphereData>> res;

    res.insert(res.end(), leftPartition.begin(), leftPartition.end());
    res.insert(res.end(), rightPartition.begin(),rightPartition.end());
    return res;
}

// GPU sort 
std::vector<std::vector<SphereData>> constructKDGroupsGPUHelper(
    std::vector<SphereData> allPointSpheresData)
{
    std::vector<SphereData> sortedData = std::move(allPointSpheresData);

    KDPartitionRange* partitionArray = nullptr;
    uint32_t partitionCount = constructKDGroupsGPU(sortedData.data(),
                                                   static_cast<uint32_t>(sortedData.size()),
                                                   &partitionArray);

    std::vector<std::vector<SphereData>> res;

    if (partitionCount == 0u || partitionArray == nullptr)
    {
        return res;
    }

    res.reserve(partitionCount);

    for (uint32_t i = 0; i < partitionCount; ++i)
    {
        const KDPartitionRange& range = partitionArray[i];
        std::vector<SphereData> group(sortedData.begin() + range.begin,
                                      sortedData.begin() + range.end);
        res.push_back(std::move(group));
    }

    delete[] partitionArray;

    return res;
}


// CPU based bounding groups constructor
// std::vector<boundingSphereGroup> constructKDGroups(
//     std::vector<SphereData> allPointSpheresData)
// {
//     auto kdTree = constructKDGroupsRec(allPointSpheresData, 0);
//     std::vector<boundingSphereGroup> res;

//     res.reserve(kdTree.size());

//     for(auto &group : kdTree)
//     {
//         res.push_back({constructBoundingSphere(group), group});
//     }

//     // This will try to merge overlapping spheres
//     // res = constructDistinctNonOverlappingSpheres(res);

//     // Compute morton for locality of spheres - this might help with BVH construction
//     std::vector<uint32_t> mortonCode;
//     for(auto & i : res)
//     {
//         // mortonCode.push_back(EncodeMorton3(x, y, z));
//         uint32_t x = static_cast<uint32_t>(i.sphere.center.x * 1023);
//         uint32_t y = static_cast<uint32_t>(i.sphere.center.y * 1023);
//         uint32_t z = static_cast<uint32_t>(i.sphere.center.z * 1023);

//         mortonCode.push_back(EncodeMorton3(x, y, z));
//     }

//     thrust::sort_by_key(mortonCode.begin(), mortonCode.end(), res.begin());

//     return res;

// }

// GPU based bounding groups constructor
std::vector<boundingSphereGroup> constructKDGroups(
    std::vector<SphereData> allPointSpheresData)
{
    auto kdTreeGroups = constructKDGroupsGPUHelper(allPointSpheresData);

    std::vector<boundingSphereGroup> res;

    res.reserve(kdTreeGroups.size());

    for(auto &group : kdTreeGroups)
    {
        res.push_back({constructBoundingSphere(group), group});
    }

    // This will try to merge overlapping spheres
    // res = constructDistinctNonOverlappingSpheres(res);

    // Compute morton for locality of spheres - this might help with BVH construction
    std::vector<uint32_t> mortonCode;
    for(auto & i : res)
    {
        // mortonCode.push_back(EncodeMorton3(x, y, z));
        uint32_t x = static_cast<uint32_t>(i.sphere.center.x * 1023);
        uint32_t y = static_cast<uint32_t>(i.sphere.center.y * 1023);
        uint32_t z = static_cast<uint32_t>(i.sphere.center.z * 1023);

        mortonCode.push_back(EncodeMorton3(x, y, z));
    }

    thrust::sort_by_key(mortonCode.begin(), mortonCode.end(), res.begin());

    return res;

}

// First Setup
void setupDataSpaceAndRayGenSpheresBuffer(OWLContext context, OWLGeom& SpheresGeom, OWLRayGen rayGen,
                                          std::vector<uint32_t> pointsToSearch, std::vector<uint32_t> dataPoints,
                                          OWLBuffer& pointsToSearchSpheresBuffer,
                                          OWLBuffer& dataPointsSpheresBuffer,
                                          OWLGroup& handle,
                                          OWLGroup &spheresGroups,
                                          float* dataset,
                                          OWLBuffer &OWL_datasetReordered,
                                          OWLBuffer &owl_groupCountBuffer,
                                          std::vector<uint32_t> &groupCount,
                                          OWLBuffer &owl_groupedPointsBuffer,
                                          std::vector<uint32_t> mortonCode,
                                          OWLBuffer &queryIndices,
                                          uint32_t* GROUPED_QUERY_COUNT,
                                          uint32_t*maxPointsWithinPrimitive,
                                          unsigned int DIM, float EPSILON, OWLParams& lp)
{
    std::vector<Sphere> pointsToSearchSpheres;
    std::vector<std::vector<Sphere>> dataPointsSpheres;

    static double dataBufferTime = 0;

    double dataBufferTimeStart = omp_get_wtime();

    std::vector<Sphere> allPointSpheres = buildSpheresPrimBuffer(dataPoints, dataset, DIM, EPSILON);
    const uint32_t N = dataPoints.size();

    std::vector<SphereData> allPointSpheresData;

    for (uint32_t i = 0; i < allPointSpheres.size(); i++)
    {
        auto& sphere = allPointSpheres[i];
        allPointSpheresData.emplace_back(sphere, i);
    }

    // These are spheres that are grouped together
    std::vector<Sphere> groupedSpheres;
    std::vector<uint32_t> groupedPoints;

    // Temp code to reorder dataset
    float*datasetReordered = (float*)malloc(sizeof(float)*N*DIM);
    
    double kdStart = omp_get_wtime();
    auto boundingSphereGroups =  constructKDGroups(allPointSpheresData);
    double kdEnd = omp_get_wtime();

    kdTreeConstructionTime = kdEnd - kdStart;
    printf("\n Time for KD tree construction is %.3f seconds", kdEnd-kdStart);

    groupCount.assign(boundingSphereGroups.size() + 1, 0);
    
    uint32_t pointIdx = 0;
    for (auto& sphereGroup : boundingSphereGroups)
    {
        groupedSpheres.push_back(sphereGroup.sphere);

        // Finding maximum points per primtive
        if (sphereGroup.sphereGroup.size() > *maxPointsWithinPrimitive)
        {
            *maxPointsWithinPrimitive = sphereGroup.sphereGroup.size();
        }

        for (auto& sphereData : sphereGroup.sphereGroup)
        {
            // Create new reordered dataset with good memory accesses
            uint32_t curPoint = sphereData.pID;

            for(uint32_t j = 0; j < DIM ; j++)
            {
                datasetReordered[pointIdx*DIM +j] = dataset[curPoint*DIM +j];
            }

            groupedPoints.push_back(pointIdx);

            pointIdx++;
        }
    }

    for (uint32_t i = 0; i < boundingSphereGroups.size(); i++)
    {
        groupCount[i + 1] = groupCount[i] + boundingSphereGroups[i].sphereGroup.size();
    }


    pointsToSearchSpheres = buildSpheresPrimBuffer(pointsToSearch, datasetReordered, DIM, EPSILON);

    owlParamsSetBuffer(lp, "queryIndices", queryIndices);

    OWL_datasetReordered = owlDeviceBufferCreate(context, OWL_FLOAT, N*DIM, datasetReordered);

    const uint32_t GROUPS_COUNT = groupedSpheres.size();

    fprintf(stderr, "\nNodes in BVH tree is %u", GROUPS_COUNT);
    fprintf(stderr, "\n Max points per node in BVH tree is approximately %u", *maxPointsWithinPrimitive);

    *GROUPED_QUERY_COUNT = GROUPS_COUNT;

    owl_groupCountBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), (GROUPS_COUNT + 1),
                                                           groupCount.data());


    dataPointsSpheresBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(Sphere), GROUPS_COUNT,
                                                       groupedSpheres.data());

    pointsToSearchSpheresBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(Sphere), pointsToSearchSpheres.size(),
                                                        pointsToSearchSpheres.data());

    assert(groupedPoints.size() == N);

    owl_groupedPointsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), groupedPoints.size(),
                                                     groupedPoints.data());

    owlParamsSetBuffer(lp, "spheres", pointsToSearchSpheresBuffer);

    owlGeomSetPrimCount(SpheresGeom, GROUPS_COUNT);

    owlGeomSetBuffer(SpheresGeom, "prims", dataPointsSpheresBuffer);

    double dataBufferTimeEnd = omp_get_wtime();

    dataBufferTime += dataBufferTimeEnd - dataBufferTimeStart;

    printf("\n Data buffer setup time is %.3f seconds", dataBufferTime);
}


void updateLaunchParams(unsigned int NUM_POINTS_TO_SEARCH, unsigned int MAX_CANDIDATE_POINTS,
                        OWLBuffer& d_totalCandidateCount,
                        OWLBuffer& candidatePointsCount, OWLBuffer& candidatePoints, uint32_t** candidateCount,
                        uint64_t** candidates, OWLContext context, OWLParams& lp)
{
    owlParamsSet1ui(lp, "maxCandidatePointsPerQueryPoint", MAX_CANDIDATE_POINTS);

    cudaMemsetAsync((uint32_t*)owlBufferGetPointer(d_totalCandidateCount, 0), 0, sizeof(uint32_t),
                    owlParamsGetCudaStream(lp, 0));

    owlParamsSetBuffer(lp, "totalCandidateCount", d_totalCandidateCount);

    if (MAX_CANDIDATE_POINTS != UINT_MAX)
    {
        owlParamsSetBuffer(lp, "candidatePoints", candidatePoints);
    }
}

// Use this when queries change but data points remains the same
void updateRayGenSphereBuffer(OWLContext context, std::vector<uint32_t> pointsToSearch,
                              OWLBuffer& pointsToSearchSpheresBuffer,
                              OWLBuffer& truthBuffer,
                              float* dataset, unsigned int DIM, float EPSILON,
                              OWLParams& lp, const unsigned int NUM_DATA_PARTITIONS_, const unsigned int N)
{
    std::vector<Sphere> pointsToSearchSpheres;

    pointsToSearchSpheres = buildCumDataPartitionSpheresPrimeBuffer(pointsToSearch, dataset,
                                                                    DIM, EPSILON,
                                                                    NUM_DATA_PARTITIONS_);

    static double pointToSearchTime = 0;

    double pointToSearchTimeStart = omp_get_wtime();

    const size_t pointsToSearchSpheresCount = pointsToSearchSpheres.size();

    cudaMemcpyAsync((void*)owlBufferGetPointer(pointsToSearchSpheresBuffer, 0), pointsToSearchSpheres.data(),
                    sizeof(pointsToSearchSpheres[0]) * pointsToSearchSpheresCount, cudaMemcpyHostToDevice,
                    owlParamsGetCudaStream(lp, 0));

    owlParamsSetBuffer(lp, "spheres", pointsToSearchSpheresBuffer);


    double pointToSearchTimeEnd = omp_get_wtime();

    pointToSearchTime += pointToSearchTimeEnd - pointToSearchTimeStart;

    printf("\n Points setup time is %.2f", pointToSearchTime * 1000);

    // Set Truth Buffer
    const uint32_t pointToSearchCount = pointsToSearch.size();


    static double truthSetupTime = 0;

    double truthSetupStart = omp_get_wtime();

#if NUM_DATA_PARTITIONS != 1
    {
        auto* d_truthBuffer = (uint32_t*)owlBufferGetPointer(truthBuffer, 0);

        uint32_t minBufferPerQuery = (N + NUM_SUB_TRUTH_BUFFER_PER_WORD - 1) / NUM_SUB_TRUTH_BUFFER_PER_WORD;
        uint32_t totalTruthBufferSize = minBufferPerQuery * pointsToSearch.size();

        cudaMemsetAsync(d_truthBuffer, 0, totalTruthBufferSize * sizeof(uint32_t), owlParamsGetCudaStream(lp, 0));

        owlParamsSetBuffer(lp, "truthBuffer", truthBuffer);
    }
#endif
    double truthSetupEnd = omp_get_wtime();

    truthSetupTime += truthSetupEnd - truthSetupStart;

    printf("\n Truth buffer setup time is %.2f\n", truthSetupTime * 1000);
}

// Use this to set ray gen buffer with existing query buffer
void setRayGenBuffer(OWLRayGen rayGen, OWLBuffer pointsToSearchSpheresBuffer)
{
    owlRayGenSetBuffer(rayGen, "spheres", pointsToSearchSpheresBuffer);
}

// Use this when exact number of candidate points is determined
void updateLaunchParamsWithExactBuffer(unsigned int NUM_POINTS_TO_SEARCH, const uint32_t* candidatePointsPerQuery,
                                       uint64_t totalCandidatePoints, OWLBuffer& candidatePointsCount,
                                       OWLBuffer& candidatePoints, OWLBuffer& candidatePointsOffset,
                                       uint32_t** candidateCount, uint32_t** candidates,
                                       uint32_t** candidatePointsOffsetPerPartition, OWLContext context, OWLParams& lp)
{
    if (*candidateCount != nullptr)
    {
        owlBufferDestroy(candidatePointsCount);
        free(*candidateCount);
    }

    if (*candidates != nullptr)
    {
        owlBufferDestroy(candidatePoints);
        free(*candidates);
    }

    if (*candidatePointsOffsetPerPartition != nullptr)
    {
        owlBufferDestroy(candidatePointsOffset);
        free(*candidatePointsOffsetPerPartition);
    }

    // Buffer for candidate count
    *candidateCount = static_cast<uint32_t*>(calloc(NUM_POINTS_TO_SEARCH, sizeof(uint32_t)));


    candidatePointsCount = registerMemory(context, OWL_USER_TYPE((*candidateCount)[0]), NUM_POINTS_TO_SEARCH,
                                          *candidateCount,
                                          sizeof((*candidateCount)[0]) * NUM_POINTS_TO_SEARCH);

    owlParamsSetBuffer(lp, "candidatePointsCount", candidatePointsCount);


    // Use this as an identifier for first intersection test
    owlParamsSet1ui(lp, "maxCandidatePointsPerQueryPoint", UINT_MAX);

    // Buffer for storing all candidate points
    *candidates = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * totalCandidatePoints));


    candidatePoints = registerMemory(context, OWL_USER_TYPE((*candidates)[0]), totalCandidatePoints,
                                     *candidates,
                                     sizeof((*candidates)[0]) * totalCandidatePoints);
    owlParamsSetBuffer(lp, "candidatePoints", candidatePoints);

    // Buffer for offsets
    *candidatePointsOffsetPerPartition = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * (NUM_POINTS_TO_SEARCH + 1)));

    // Set initial offset as 0
    (*candidatePointsOffsetPerPartition)[0] = 0;

    // Build offsets for each query
    for (int i = 1; i <= NUM_POINTS_TO_SEARCH; i++)
    {
        (*candidatePointsOffsetPerPartition)[i] = (*candidatePointsOffsetPerPartition)[i - 1] + candidatePointsPerQuery[
            i - 1];
    }

    candidatePointsOffset = registerMemory(context, OWL_USER_TYPE((*candidatePointsOffsetPerPartition)[0]),
                                           (NUM_POINTS_TO_SEARCH + 1),
                                           *candidatePointsOffsetPerPartition,
                                           sizeof((*candidatePointsOffsetPerPartition)[0]) * (NUM_POINTS_TO_SEARCH +
                                               1));

    owlParamsSetBuffer(lp, "candidatePointsBufferOffset", candidatePointsOffset);
}

double getUsableMemory(int deviceID)
{
    cudaDeviceProp prop{};
    double totalDRAMSize;
    cudaGetDeviceProperties(&prop, deviceID);
    totalDRAMSize = static_cast<double>(prop.totalGlobalMem) / (1024 * 1024 * 1024);
    // conservatively reduce dram size by 256 MB.
    totalDRAMSize -= 0.25;
    std::cerr << "\tMax usable Memory: " << totalDRAMSize << " GB" << std::endl;
    return totalDRAMSize;
}

void updateQueriesInLaunchParams(OWLParams& lp, OWLBuffer& pointsToSearchSpheresBuffer, const uint32_t POINTS_TO_SEARCH)
{
    owlParamsSetBuffer(lp, "spheres", pointsToSearchSpheresBuffer);
    owlParamsSet1ui(lp, "POINTS_TO_SEARCH", POINTS_TO_SEARCH);
}

void resetTruthBuffer(OWLContext context, OWLParams& lp, OWLBuffer& truthBuffer, const unsigned int POINTS_TO_SEARCH,
                      const unsigned int N)
{
    uint32_t* h_truthBuffer = (uint32_t*)calloc(POINTS_TO_SEARCH * N, sizeof(uint32_t));
    truthBuffer = registerMemory(context, OWL_USER_TYPE(h_truthBuffer[0]), POINTS_TO_SEARCH * N,
                                 h_truthBuffer,
                                 sizeof(uint32_t) * POINTS_TO_SEARCH * N);
    owlParamsSetBuffer(lp, "truthBuffer", truthBuffer);
}

// void updateDataSpace(OWLContext context, OWLParams& lp, OWLRayGen rayGen, OWLGeom* SpheresGeom,
//                      std::vector<OWLBuffer>& dataPointsSpheresBuffer, const unsigned int NUM_DATA_POINTS,
//                      const unsigned int NUM_DATA_PARTITIONS)
// {
//     for (int i = 0; i < NUM_DATA_PARTITIONS; i++)
//     {
//         owlGeomSetPrimCount(SpheresGeom[i], NUM_DATA_POINTS);
//         owlGeomSetBuffer(SpheresGeom[i], "prims", dataPointsSpheresBuffer[i]);
//     }
//
//     owlParamsSet1ui(lp, "TOTAL_POINTS", NUM_DATA_POINTS);
//     setupIAS(SpheresGeom, context, rayGen, NUM_DATA_PARTITIONS);
// }

// Here count is the number of elemtnents and not hethe size
OWLBuffer registerMemory(OWLContext context,
                         OWLDataType type,
                         size_t count,
                         const void* data,
                         size_t allocatedMemory)
{
    //NOTE:May be use owlBufferSizeInBytes
    // Impl details don't change significantly. Readability issue
    OWLBuffer buffer = owlDeviceBufferCreate(context, type, count, data);
    memoryTracker[buffer] = {BufferType::OWL_ALLOCATED_BUFFER, allocatedMemory};
    curAllocatedMemory += allocatedMemory;
    return buffer;
}

void registerMemory(void** d_ptr, size_t allocatedMemory)
{
    cudaErrorCheck(cudaMalloc(d_ptr, allocatedMemory), "cudaMalloc(d_ptr, allocatedMemory)");
    memoryTracker[*d_ptr] = {BufferType::DEVICE_BUFFER, allocatedMemory};
    curAllocatedMemory += allocatedMemory;
}

void unRegisterMemory(void* d_ptr, BufferType bufferType)
{
    if (bufferType == BufferType::OWL_ALLOCATED_BUFFER)
    {
        auto owlBufferToRelease = reinterpret_cast<OWLBuffer>(d_ptr);
        // owlBufferRelease(owlBufferToRelease);
        owlBufferDestroy(owlBufferToRelease);
    }
    else
    {
        cudaErrorCheck(cudaFree(d_ptr), "cudaFree(d_ptr)");
    }

    if (memoryTracker.find(d_ptr) != memoryTracker.end())
    {
        curAllocatedMemory -= memoryTracker[d_ptr].size;
        memoryTracker.erase(d_ptr);
    }
}

void releaseAll()
{
    for (const auto& bufferPointer : memoryTracker)
    {
        if (bufferPointer.second.type == BufferType::OWL_ALLOCATED_BUFFER)
        {
            auto owlBufferToRelease = reinterpret_cast<OWLBuffer>(bufferPointer.first);
            // owlBufferRelease(owlBufferToRelease);
            owlBufferDestroy(owlBufferToRelease);
        }
        else
        {
            cudaErrorCheck(cudaFree(bufferPointer.first), "cudaFree(bufferPointer.first)");
        }
    }
}

size_t estimateMemoryForEachPartitionForRT(const uint64_t* partitionCandidatePoints,
                                           const uint32_t* pointsToSearchPerPartition,
                                           std::vector<size_t>& memoryForEachPartitionForRT,
                                           uint32_t NUM_PARTITIONS, unsigned int N)
{
    size_t maxReqMemory = 0;
    for (uint32_t i = 0; i < NUM_PARTITIONS; i++)
    {
        // Candidate buffer + candidate count buffer + candidate offset buffer
        size_t curReqMem = partitionCandidatePoints[i] * sizeof(uint32_t) + (2 *
            pointsToSearchPerPartition[i] + 1) * sizeof(uint32_t);

        // This is for truth buffer
        curReqMem += sizeof(uint32_t) * N * pointsToSearchPerPartition[i];

        memoryForEachPartitionForRT.push_back(curReqMem);

        if (memoryForEachPartitionForRT[i] > maxReqMemory)
        {
            maxReqMemory = memoryForEachPartitionForRT[i];
        }
    }

    return maxReqMemory;
}

size_t estimateMemoryForEachPartitionForCuda(const uint32_t* pointsToSearchPerPartition,
                                             std::vector<size_t>& memoryForEachPartitionForCuda,
                                             uint32_t NUM_PARTITIONS)
{
    size_t maxReqMemory = 0;
    for (uint32_t i = 0; i < NUM_PARTITIONS; i++)
    {
        size_t curReqMem = sizeof(uint32_t) * pointsToSearchPerPartition[i];

        memoryForEachPartitionForCuda.push_back(curReqMem);

        if (memoryForEachPartitionForCuda[i] > maxReqMemory)
        {
            maxReqMemory = memoryForEachPartitionForCuda[i];
        }
    }

    return maxReqMemory;
}

bool hasEnoughMemory(size_t requiredMem)
{
    return (totalAvailableMemory - curAllocatedMemory) > requiredMem;
}

void waitForMemory(size_t requiredMem)
{
    while (!hasEnoughMemory(requiredMem))
    {
        //Ref:https://www.geeksforgeeks.org/sleep-function-in-cpp/
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

size_t getCurrentAvailableMemory()
{
    size_t freeMem, totalMem;
    cudaErrorCheck(cudaMemGetInfo(&freeMem, &totalMem), "cudaMemGetInfo(&freeMem, &totalMem)");
    return freeMem;
}


void partitionCandidatesToFitMemory(std::vector<std::vector<unsigned int>>& partitions,
                                    std::vector<uint32_t*>& candidatePointsForEachQueryPerPartition,
                                    std::vector<std::vector<uint32_t>>& candidatePointsForEachQueryPerPartitionVec,
                                    std::vector<uint64_t>& partitionCandidatePoints,
                                    std::vector<unsigned int>& PARTITION_SIZE,
                                    std::vector<OWLBuffer> pointsToSearchSpheresBuffer,
                                    std::vector<OWLBuffer> truthBuffer,
                                    uint32_t* NUM_PARTITIONS,
                                    size_t maxMemConservative,
                                    uint32_t NUM_DATA_PARTITIONS_)
{
    std::vector<std::vector<unsigned int>> newPartitions;
    std::vector<uint64_t> newPartitionCandidatePoints;
    std::vector<unsigned int> NEW_PARTITION_SIZE;

    size_t curAvailableMemory = getCurrentAvailableMemory();

    uint32_t numSplits = 1;

    // Cut into half until memory is sufficient
    //TODO: Change this.
    while (curAvailableMemory < maxMemConservative / numSplits)
    {
        numSplits = numSplits * 2;
    }

    for (uint32_t i = 0; i < partitions.size(); i++)
    {
        thrust::stable_sort_by_key(candidatePointsForEachQueryPerPartition[i],
                                   candidatePointsForEachQueryPerPartition[i] + partitions[i].size(),
                                   partitions[i].begin(), thrust::less<uint32_t>());

        std::vector<std::vector<uint32_t>> subPartitions(numSplits, std::vector<uint32_t>());
        std::vector<uint32_t> subPartitionsSum(numSplits, 0);

        std::vector<std::vector<uint32_t>> subPartitionsCandidatePointsForEachQuery(numSplits, std::vector<uint32_t>());

        for (uint32_t j = 0; j < partitions[i].size(); j++)
        {
            // Ref https://stackoverflow.com/questions/55304846/how-to-get-the-index-of-the-smallest-element-in-a-vector
            uint32_t minSubPartitionIndex = std::distance(subPartitionsSum.begin(),
                                                          std::min_element(
                                                              subPartitionsSum.begin(), subPartitionsSum.end()));
            subPartitions[minSubPartitionIndex].push_back(partitions[i][j]);
            subPartitionsCandidatePointsForEachQuery[minSubPartitionIndex].push_back(
                candidatePointsForEachQueryPerPartition[i][j]);
            subPartitionsSum[minSubPartitionIndex] += candidatePointsForEachQueryPerPartition[i][j];
        }


        // Update new placeholders
        for (uint32_t j = 0; j < numSplits; j++)
        {
            newPartitions.push_back(subPartitions[j]);

            candidatePointsForEachQueryPerPartitionVec.push_back(subPartitionsCandidatePointsForEachQuery[j]);
            newPartitionCandidatePoints.push_back(subPartitionsSum[j]);
            NEW_PARTITION_SIZE.push_back(subPartitions[j].size());
        }
    }

    // TODO: FIX THIS
    // BASICALLY UPDATE CANDIDATE POINTS PER QUERY
    for (uint32_t i = 0; i < candidatePointsForEachQueryPerPartition.size(); i++)
    {
        free(candidatePointsForEachQueryPerPartition[i]);
    }

    candidatePointsForEachQueryPerPartition.clear();

    for (uint32_t i = 0; i < newPartitions.size(); i++)
    {
        uint32_t* temp = new uint32_t[candidatePointsForEachQueryPerPartitionVec[i].size()];
        std::copy(candidatePointsForEachQueryPerPartitionVec[i].begin(),
                  candidatePointsForEachQueryPerPartitionVec[i].end(), temp);
        candidatePointsForEachQueryPerPartition.push_back(temp);
    }

    partitions = std::move(newPartitions);
    partitionCandidatePoints = std::move(newPartitionCandidatePoints);
    PARTITION_SIZE = std::move(NEW_PARTITION_SIZE);
    *NUM_PARTITIONS = partitions.size();
    pointsToSearchSpheresBuffer.resize(*NUM_PARTITIONS);

    truthBuffer.resize(*NUM_PARTITIONS);
}

// Source - https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes// source
// "Insert" a 0 bit after each of the 16 low bits of x
uint32_t Part1By1(uint32_t x)
{
    x &= 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
uint32_t Part1By2(uint32_t x)
{
    x &= 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

uint32_t EncodeMorton2(uint32_t x, uint32_t y)
{
    return (Part1By1(y) << 1) + Part1By1(x);
}

uint32_t EncodeMorton3(uint32_t x, uint32_t y, uint32_t z)
{
    return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
}

std::vector<uint32_t> computeMortonCode(float* dataset, uint32_t N, uint32_t DIM)
{
    std::vector<uint32_t> primitiveMortonCode(N);

    if (DIM >= 3)
    {
        uint32_t i;
#pragma omp parallel for private(i) shared(dataset, DIM)
        for (i = 0; i < N; i++)
        {
            float* point = dataset + i * DIM;
            // Using 10 bits
            uint32_t x = static_cast<uint32_t>(*(point + 0) * 1023);
            uint32_t y = static_cast<uint32_t>(*(point + 1) * 1023);
            uint32_t z = static_cast<uint32_t>(*(point + 2) * 1023);

            primitiveMortonCode[i] = EncodeMorton3(x, y, z);
        }
    }
    else if(DIM == 2)
    {
        uint32_t i;
#pragma omp parallel for private(i) shared(dataset, DIM)
        for (i = 0; i < N; i++)
        {
            float* point = dataset + i * DIM;
            // Using 10 bits
            uint32_t x = static_cast<uint32_t>(*(point + 0) * 1023);
            uint32_t y = static_cast<uint32_t>(*(point + 1) * 1023);
            primitiveMortonCode[i] = EncodeMorton2(x, y);
        }
    }
    else
    {
        uint32_t i;
#pragma omp parallel for private(i) shared(dataset, DIM)
        for (i = 0; i < N; i++)
        {
            float* point = dataset + i * DIM;
            uint32_t x = static_cast<uint32_t>(*(point + 0) * 1023);
            primitiveMortonCode[i] = x;
        }
    }

    return primitiveMortonCode;
}

std::vector<uint32_t> sortQueryPrimitives(uint64_t* queryPrimitives, std::vector<uint32_t> mortonCode,
                                          uint32_t queryCount)
{
    // std::map<uint32_t, std::vector<uint32_t>> primitiveToQueryMap;
    //
    // for(uint32_t i = 0 ; i < queryCount ; i++)
    // {
    //     uint64_t embed = queryPrimitives[i];
    //     uint32_t queryIndex = static_cast<uint32_t>(embed >> 32);
    //     uint32_t primitiveIndex = static_cast<uint32_t>(embed & 0xFFFFFFFF);
    //     if(primitiveToQueryMap.find(primitiveIndex) == primitiveToQueryMap.end())
    //     {
    //         primitiveToQueryMap[primitiveIndex] = std::vector<uint32_t>(1, queryIndex);
    //     }else
    //     {
    //         primitiveToQueryMap[primitiveIndex].push_back(queryIndex);
    //     }
    // }
    //
    // std::vector<uint32_t> sortedQueries;
    //
    // for(auto&[primitiveIndex, queryIndexVec] : primitiveToQueryMap)
    // {
    //     sortedQueries.insert(sortedQueries.end(), queryIndexVec.begin(), queryIndexVec.end());
    // }
    // return sortedQueries;

    thrust::host_vector<uint32_t> queries(queryCount);
    thrust::host_vector<uint32_t> primitives(queryCount);

    for (uint32_t i = 0; i < queryCount; i++)
    {
        queries[i] = static_cast<uint32_t>(queryPrimitives[i] >> 32);
        primitives[i] = static_cast<uint32_t>(queryPrimitives[i] & 0xFFFFFFFF);

        if (primitives[i] > queryCount)
        {
            printf("Prim Id should not more than total available data points.");
            exit(1);
        }
        primitives[i] = mortonCode[primitives[i]];
    }

    thrust::sort_by_key(primitives.begin(), primitives.end(), queries.begin());

    return std::vector<uint32_t>(queries.begin(), queries.end());
}
