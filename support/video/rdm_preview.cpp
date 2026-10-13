// Optional streaming display reducer. Scientific RDM1 maps stay unchanged in
// the production pipeline. This tool emits distinctly framed RDP1 previews.
#include <boost/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

namespace {
struct Complex {float real,imaginary;};
void write(const void* data,std::size_t bytes){if(std::fwrite(data,1,bytes,stdout)!=bytes)throw std::runtime_error("Preview write failed");}
template<class T> void write(const T& value){write(&value,sizeof(value));}
bool read(void* data,std::size_t bytes,bool eof=false) {
    const auto count=std::fread(data,1,bytes,stdin);
    if(eof && !count && std::feof(stdin))return false;
    if(count!=bytes)throw std::runtime_error("Truncated map");
    return true;
}
template<class T> T read(){T value{};read(&value,sizeof(value));return value;}
template<unsigned F>
void pool(const std::vector<float>& input,std::vector<float>& output,std::size_t nr,std::size_t nf) {
    const auto width=(nr+1)/2,height=(nf+F-1)/F;
    for(std::size_t r=0;r<width;++r) {
        const auto* a=input.data()+2*r*nf;
        const auto* b=input.data()+std::min(2*r+1,nr-1)*nf;
        for(std::size_t f=0;f<nf/F;++f) {
            float maximum=0;
            for(unsigned j=0;j<F;++j)maximum=std::max(maximum,std::max(a[f*F+j],b[f*F+j]));
            output[(height-1-f)*width+r]=maximum;
        }
        if(nf%F) {
            float maximum=0;
            for(auto f=nf-nf%F;f<nf;++f)maximum=std::max(maximum,std::max(a[f],b[f]));
            output[r]=maximum;
        }
    }
}
void pool_dynamic(const std::vector<float>& input,std::vector<float>& output,std::size_t nr,std::size_t nf,unsigned rd,unsigned fd) {
    const auto width=(nr+rd-1)/rd,height=(nf+fd-1)/fd;
    std::fill(output.begin(),output.end(),0);
    for(std::size_t r=0;r<nr;++r)for(std::size_t f=0;f<nf;++f) {
        auto& value=output[(height-1-f/fd)*width+r/rd];
        value=std::max(value,input[r*nf+f]);
    }
}
}
int main() {
    try {
        static_assert(std::endian::native==std::endian::little && sizeof(Complex)==8);
        std::vector<Complex> input;std::vector<float> output,power;
        std::map<unsigned,std::pair<std::uint64_t,double>> previous;
        for(;;) {
            std::uint32_t magic{};if(!read(&magic,sizeof(magic),true))break;
            const auto length=read<std::uint32_t>();const auto bytes=read<std::uint64_t>();
            if(magic!=0x314d4452 || !length || length>1024*1024 || bytes>2ULL*1024*1024*1024)throw std::runtime_error("Invalid RDM1 map framing");
            std::string header(length,'\0');read(header.data(),length);
            auto metadata=boost::json::parse(header);auto& m=metadata.as_object();
            const auto nr=boost::json::value_to<std::size_t>(m.at("ranges")),nf=boost::json::value_to<std::size_t>(m.at("frequencies"));
            const auto channel=boost::json::value_to<unsigned>(m.at("channel"));
            const auto sequence=boost::json::value_to<std::uint64_t>(m.at("sequence"));
            const auto end=boost::json::value_to<double>(m.at("time_end_s"));
            auto& p=previous[channel];
            if(!nr || !nf || bytes!=nr*nf*sizeof(Complex) || m.at("format")!="cf32le" || m.at("order")!="range,doppler" || sequence!=p.first+1 || end<p.second)
                throw std::runtime_error("Invalid map layout or sequence");
            p={sequence,end};input.resize(nr*nf);
            read(input.data(),bytes);
            power.resize(input.size());unsigned invalid=0;
            for(std::size_t i=0;i<input.size();++i) {
                const auto x=input[i];
                invalid|=unsigned(!std::isfinite(x.real))|unsigned(!std::isfinite(x.imaginary));
                power[i]=x.real*x.real+x.imaginary*x.imaginary;
            }
            if(invalid)throw std::runtime_error("Nonfinite scientific map");
            const auto rd=unsigned((nr+1623)/1624),fd=unsigned((nf+739)/740);
            const auto width=(nr+rd-1)/rd,height=(nf+fd-1)/fd;output.resize(width*height);
            if(rd==2 && fd==7)pool<7>(power,output,nr,nf);
            else if(rd==2 && fd==4)pool<4>(power,output,nr,nf);
            else pool_dynamic(power,output,nr,nf,rd,fd);
            m.erase("noise_range_factor");m.erase("noise_doppler_power");
            m["display_shape"]=boost::json::array{height,width};m["preview_format"]="power_f32le";m["preview_order"]="doppler_down,range";
            m["map_peak_db"]=10*std::log10(std::max(*std::max_element(output.begin(),output.end()),1e-45f));
            header=boost::json::serialize(metadata);
            write(std::uint32_t{0x31504452});write(std::uint32_t(header.size()));write(std::uint64_t(output.size()*sizeof(float)));
            write(header.data(),header.size());write(output.data(),output.size()*sizeof(float));
        }
        if(std::fflush(stdout)!=0)throw std::runtime_error("Preview output flush failed");
    }catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
