#include "gpu_compute.hpp"
#include <future>
#include <iostream>
#include <numeric>

namespace {
constexpr std::uint32_t code[] =
#include "command_chain.inc"
;
void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
}
int main(){try {
    using namespace radar;
    auto context=std::make_shared<gpu::Context>();
    gpu::Kernel first(code,8,{},context),second(code,8,{},context),snapshot(code,8,{},context);
    constexpr unsigned count=65536;
    std::vector<std::uint32_t> input(count);std::iota(input.begin(),input.end(),0U);
    first.upload(0,std::span<const std::uint32_t>{input});
    for(auto* kernel:{&first,&second,&snapshot})kernel->storage(1,count*sizeof(std::uint32_t));
    second.bind(0,first.view(1));snapshot.bind(0,second.view(1));
    ComputeTimings timings;
    for(unsigned pass=0;pass<32;++pass) {
        gpu::Commands commands(context);
        for(auto* kernel:{&first,&second,&snapshot}) {
            const auto cmd=kernel->begin(&timings,2,&commands);
            kernel->dispatch(cmd,kernel->primary(),std::array{count,pass+1},count);kernel->finish(cmd);
        }
        // CPU readers wait on a value ticket, not a borrowed kernel or caller
        // stack accumulator. The recording thread may independently collect.
        auto complete=commands.submit();
        auto reader=std::async(std::launch::async,[complete,view=snapshot.view(1),pass]{
            complete.wait();const auto* values=static_cast<const std::uint32_t*>(view.mapped);
            for(unsigned i=0;i<count;++i)check(values[i]==i+3*(pass+1),"Dependent command buffers lost a write or used stale descriptors");
        });
        first.wait();second.wait();reader.get();snapshot.wait();
    }
    check(timings.batches==96,"Deferred timestamp collection was lost or repeated");
    // Reusing a recording kernel forces submission before command-buffer reset
    // and upload mutation, including when the caller has not submitted yet.
    gpu::Commands commands(context);
    auto cmd=first.begin(nullptr,2,&commands);
    first.dispatch(cmd,first.primary(),std::array{count,7U},count);first.finish(cmd);
    cmd=second.begin(nullptr,2,&commands);
    second.dispatch(cmd,second.primary(),std::array{count,11U},count);second.finish(cmd);
    first.upload(0,std::span<const std::uint32_t>{input});
    const auto* values=static_cast<const std::uint32_t*>(second.view(1).mapped);
    for(unsigned i=0;i<count;++i)check(values[i]==i+18,"Resource reuse overtook its dependent GPU consumer");
    commands.wait();
    std::cout<<"GPU dependency grouping, worker readiness, bounded reuse and query lifetime: PASS\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
