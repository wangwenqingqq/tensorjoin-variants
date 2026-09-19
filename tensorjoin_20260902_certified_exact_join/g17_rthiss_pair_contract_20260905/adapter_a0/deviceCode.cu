#include "deviceCode.h"
#include <optix_device.h>
#include <thread>
#include "utility.h"
#include "math.h"

__constant__ LaunchParams optixLaunchParams;

OPTIX_BOUNDS_PROGRAM(SpheresBounds)(const void* geomData,
                                    box3f& primBounds,
                                    const int primID)
{
    const SpheresGeom& self = *(const SpheresGeom*)geomData;

    const Sphere& sphere = self.prims[primID];

    primBounds = box3f()
                 .extend(sphere.center - vec3f(sphere.radius))
                 .extend(sphere.center + vec3f(sphere.radius));
}


OPTIX_INTERSECT_PROGRAM(SpheresIntersect)()
{
    if (!optixLaunchParams.intersectionCountTest)
    {
        uint32_t query = optixLaunchParams.queryIndices[optixGetLaunchIndex().x];
        uint32_t primId = optixGetPrimitiveIndex();

        uint32_t totalCount = atomicAdd(optixLaunchParams.totalCandidateCount ,1);
        uint32_t curCandidatesCount = atomicAdd(&optixLaunchParams.primitiveIntersectionCount[optixGetPrimitiveIndex()] ,1);

        if(totalCount < optixLaunchParams.maxCandidatePointsPerQueryPoint)
        {   
            optixLaunchParams.candidatePoints[optixLaunchParams.primitivePosTowrite[optixGetPrimitiveIndex()] + curCandidatesCount] = query;
        }else
        {
            printf("\nOUT OF BOUNDS -IS shader");
        }
    }
    else
    {
        atomicAdd(&optixLaunchParams.intersectionCount[optixLaunchParams.queryIndices[optixGetLaunchIndex().x]], 1);
        atomicAdd(&optixLaunchParams.primitiveIntersectionCount[optixGetPrimitiveIndex()], 1u);
    }
}

OPTIX_ANY_HIT_PROGRAM(SpheresAnyHit)()
{
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();

    uint32_t queryIndex = optixLaunchParams.queryIndices[optixGetLaunchIndex().x];

    vec3f center = optixLaunchParams.spheres[queryIndex].center;

    PRD prd{};

    owl::Ray ray(center, vec3f(1, 0, 0), 0.0f, 1.e-16f, 0b00000001u);

    owl::traceRay(self.handle, ray, prd, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
}
