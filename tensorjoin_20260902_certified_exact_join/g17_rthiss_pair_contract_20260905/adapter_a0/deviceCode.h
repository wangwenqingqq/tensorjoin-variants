#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>

#define STORE_CANDIDATE_POINTS true
#define COUNT_METRICS false
#define MAX_REGISTERS 4

using namespace owl;

struct Sphere
{
    vec3f center;
    float radius;

    __host__ __device__ Sphere(vec3f center, float radius)
    {
        this->center = center;
        this->radius = radius;
    }

    __host__ __device__ Sphere()
    {
        this->center = {0.0f, 0.0f, 0.0f};
        this->radius = -1;
    }

    __host__ __device__ Sphere(const Sphere&) = default;
    __host__ __device__ Sphere& operator=(const Sphere&) = default;
};

struct SpheresGeom
{
    Sphere* prims;
};

struct rayQuery
{
    struct Sphere*sphere;
    int queryPartition;
};


struct PRD{

};

struct LaunchParams
{
    unsigned int maxCandidatePointsPerQueryPoint;
    uint32_t * candidatePoints;
    Sphere*spheres;
    uint32_t*totalCandidateCount;
    unsigned int * queryIndices;
    unsigned int*intersectionCount;
    unsigned int intersectionCountTest;
    uint32_t*primitiveIntersectionCount;
    uint32_t*primitivePosTowrite;
};


struct RayGenData
{
    OptixTraversableHandle handle;
};

struct MissProgData
{
};

