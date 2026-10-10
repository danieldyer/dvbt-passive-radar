#include "gpu_reconstruction.hpp"
#include "gpu_compute.hpp"
#include <bit>
#include <limits>

namespace radar {
namespace {
constexpr std::uint32_t encode_code[] =
#include "reencode.inc"
;
constexpr std::uint32_t map_code[] =
#include "remap.inc"
;
}
struct GpuReconstruction::Impl {
    gpu::Kernel kernel;
    vk::UniquePipeline remap;
    int constellation=-1;
    explicit Impl(std::shared_ptr<gpu::Context> context):kernel(encode_code,16,{},std::move(context)),
        remap(kernel.pipeline(map_code)){}
    void configure(const Tps& mode,const Modem& tables) {
        const auto m=qam_bits(mode);
        if(constellation==mode.constellation)return;
        const auto forward=tables.payload_bit_indices(mode);
        std::vector<std::uint32_t> inverse(4*6817*m,UINT32_MAX);
        for(unsigned mod=0;mod<4;++mod)for(unsigned at=0;at<6048*m;++at) {
            const auto out=forward[mod*6048*m+at];
            inverse[(mod*6817+out/8)*m+out%8]=at;
        }
        const auto pattern=tables.pilot_pattern();
        std::vector<C> constants(64+6817+1);
        const auto points=qam_constellation(m);std::copy(points.begin(),points.end(),constants.begin());
        for(unsigned k=0;k<6817;++k)constants[64+k]={pattern.values[k],0};
        for(auto k:pattern.tps)constants[64+k].imag(1);
        constants.back()={static_cast<float>(m),0};
        kernel.upload(1,std::span<const std::uint32_t>{inverse});kernel.upload(6,std::span<const C>{constants});
        constellation=mode.constellation;
    }
};
GpuReconstruction::GpuReconstruction(std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
GpuReconstruction::~GpuReconstruction()=default;
ReferenceBatch GpuReconstruction::reconstruct(const FecBatch& packets,std::span<const Tps> frames,const Modem& tables,ComputeTimings* timings,gpu::Commands* commands) {
    if(frames.empty())return {};
    if(frames.size()>15)throw std::invalid_argument("GPU reconstruction exceeds 15 OFDM frames");
    const auto& mode=frames.front();auto& p=*impl_;p.configure(mode,tables);
    std::vector<float> signs;signs.reserve(frames.size()*68);
    for(unsigned frame=0;frame<frames.size();++frame) {
        Tps checked;
        if(!tables.tps_bits(frames[frame].bits,checked) || checked.constellation!=mode.constellation ||
           checked.rate!=mode.rate || checked.mode!=mode.mode || checked.hierarchy!=mode.hierarchy ||
           checked.guard!=mode.guard || checked.frame!=(mode.frame+int(frame))%4)
            throw std::invalid_argument("Reconstruction requires consecutive BCH-valid TPS frames of one mode");
        float sign=1;
        for(unsigned j=0;j<68;++j){if(j && checked.bits[j])sign=-sign;signs.push_back(sign);}
    }
    const unsigned count=static_cast<unsigned>(signs.size()),m=2+2*mode.constellation;
    const auto puncture=fec::puncture_pattern(mode.rate);
    const auto sent=std::count(puncture.begin(),puncture.end(),1U);
    const auto bits=count*6048*m/sent*(puncture.size()/2);
    if(packets.trellis_bits!=bits || packets.count!=bits/8/204-11 || packets.packets.size()!=packets.count)
        throw std::invalid_argument("RS packet coordinates disagree with reconstruction frames");
    // CPU packet staging belongs to reconstruction; the decoder owns no Vulkan resources.
    auto* words=static_cast<std::uint32_t*>(p.kernel.storage(0,packets.count*208*sizeof(std::uint32_t)));
    for(std::size_t i=0;i<packets.count;++i) {
        const auto& packet=packets.packets[i];
        std::copy(packet.bytes.begin(),packet.bytes.end(),words+i*208);
        words[i*208+204]=packet.valid;words[i*208+205]=packet.corrected;
        words[i*208+206]=words[i*208+207]=0;
    }
    p.kernel.upload(5,std::span<const float>{signs});
    p.kernel.storage(2,bits*sizeof(std::uint32_t));p.kernel.storage(3,count*6817*sizeof(C));
    p.kernel.storage(4,count*6817*sizeof(std::uint32_t));
    const std::array push{static_cast<unsigned>(bits),count,static_cast<unsigned>(packets.count),static_cast<unsigned>(mode.rate)};
    const auto cmd=p.kernel.begin(timings,2,commands);
    p.kernel.dispatch(cmd,p.kernel.primary(),push,bits);p.kernel.barrier(cmd);
    p.kernel.dispatch(cmd,*p.remap,push,count*6817);p.kernel.finish(cmd);
    return {p.kernel.view(3),p.kernel.view(4),count};
}
Reconstruction GpuReconstruction::download(const ReferenceBatch& batch) const {
    Reconstruction result;
    if(!batch.count)return result;
    if(batch.carriers.context!=impl_->kernel.context() || batch.valid.context!=impl_->kernel.context() ||
       !batch.carriers.mapped || !batch.valid.mapped || batch.carriers.bytes<batch.count*6817*sizeof(C) ||
       batch.valid.bytes<batch.count*6817*sizeof(std::uint32_t))throw std::invalid_argument("Invalid reconstructed carrier batch");
    const auto* carriers=static_cast<const C*>(batch.carriers.mapped);
    const auto* valid=static_cast<const std::uint32_t*>(batch.valid.mapped);
    result.carriers.assign(carriers,carriers+batch.count*6817);result.valid.assign(valid,valid+batch.count*6817);
    return result;
}
}
