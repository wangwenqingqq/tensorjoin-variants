#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace g17 {
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline std::vector<uint64_t> decode(
    uint32_t n, const std::vector<uint32_t>& permutation,
    const std::vector<uint32_t>& groups, const std::vector<uint32_t>& grouped,
    const std::vector<uint32_t>& prefix, const std::vector<uint32_t>& counts,
    const std::vector<uint32_t>& starts, const std::vector<uint32_t>& candidates,
    const std::vector<uint32_t>& compressed) {
    require(n > 0 && n <= 4096, "G17 N outside bounded admission");
    const size_t p = counts.size();
    require(p > 0 && groups.size()==p+1 && prefix.size()==p+1 && starts.size()==p+1,
            "G17 inconsistent primitive metadata");
    require(permutation.size()==n && grouped.size()==n, "G17 missing point map");
    for (const auto* v : {&permutation,&grouped}) {
        auto sorted=*v; std::sort(sorted.begin(),sorted.end());
        for (uint32_t i=0;i<n;++i) require(sorted[i]==i,"G17 nonbijective point map");
    }
    require(groups[0]==0 && groups.back()==n && prefix[0]==0,"G17 bad prefix endpoints");
    for (size_t i=0;i<p;++i) {
        require(groups[i]<=groups[i+1],"G17 unordered group prefix");
        uint64_t bits=uint64_t(groups[i+1]-groups[i])*counts[i];
        require(uint64_t(prefix[i])+((bits+31)/32)==prefix[i+1],"G17 inconsistent mask prefix");
        require(uint64_t(starts[i])+counts[i]<=candidates.size(),"G17 candidate slice overflow");
    }
    require(uint64_t(prefix.back())*32 < (uint64_t(1)<<32),"G17 global bit position overflow");
    for (auto q:candidates) require(q<n,"G17 candidate ID out of range");
    auto bits=compressed;std::sort(bits.begin(),bits.end());
    require(std::adjacent_find(bits.begin(),bits.end())==bits.end(),"G17 duplicate compressed bit");
    std::vector<uint64_t> ids;ids.reserve(bits.size());
    for(auto b:bits) {
        uint32_t word=b>>5;
        require(word<prefix.back(),"G17 bit outside mask");
        size_t prim=std::upper_bound(prefix.begin(),prefix.end(),word)-prefix.begin()-1;
        require(prim<p && counts[prim]>0,"G17 bit in empty primitive");
        uint64_t local=uint64_t(b)-uint64_t(prefix[prim])*32;
        uint64_t point=local/counts[prim];uint32_t qslot=local%counts[prim];
        require(point<groups[prim+1]-groups[prim],"G17 set padding bit");
        uint32_t query=candidates[starts[prim]+qslot];
        uint32_t candidate=grouped[groups[prim]+point];
        ids.push_back(uint64_t(permutation[query])*n+permutation[candidate]);
    }
    std::sort(ids.begin(),ids.end());
    require(std::adjacent_find(ids.begin(),ids.end())==ids.end(),"G17 duplicate decoded pair");
    return ids;
}
}
