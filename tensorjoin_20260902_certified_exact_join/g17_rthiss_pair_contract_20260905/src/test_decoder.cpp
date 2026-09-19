#include "../adapter_a0/pair_decoder.h"
#include <iostream>
int main() {
    using V=std::vector<uint32_t>;
    V perm{2,0,3,1}, groups{0,2,2,4}, grouped{0,1,2,3}, prefix{0,1,1,2}, counts{2,0,1}, starts{0,2,2,3}, candidates{1,3,0};
    auto run=[&](V bits){return g17::decode(4,perm,groups,grouped,prefix,counts,starts,candidates,bits);};
    g17::require(run({32,3,0})==std::vector<uint64_t>({2,4,11}),"unit mapping");
    g17::require(run({}).empty(),"unit empty");
    auto a=run({32});auto b=run({3,0});a.insert(a.end(),b.begin(),b.end());std::sort(a.begin(),a.end());
    g17::require(a==run({0,3,32}),"unit multibatch");
    unsigned invalid=0;
    for(auto bits:{V{0,0},V{4},V{64}})try {run(bits);} catch(const std::runtime_error&) {++invalid;}
    perm={2,0,3,3};try{run({0});}catch(const std::runtime_error&){++invalid;}perm={2,0,3,1};
    candidates[0]=4;try{run({0});}catch(const std::runtime_error&){++invalid;}candidates[0]=1;
    prefix[2]=0;try{run({0});}catch(const std::runtime_error&){++invalid;}
    g17::require(invalid==6,"invalid input admitted");
    std::cout<<"{\"positive_cases\":3,\"rejected_invalid_cases\":6,\"pass\":true}"<<std::endl;
}
