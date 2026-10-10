#include "sar_handoff.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace radar {
namespace {
constexpr std::uint64_t exact_limit=std::uint64_t{1}<<53;
template<class T> void put(std::ofstream& file,const T& x) {
    static_assert(std::is_arithmetic_v<T>);
    file.write(reinterpret_cast<const char*>(&x),sizeof(x));
}
void record(std::ofstream& file,const char* name,std::uint64_t bytes) {
    file.write(name,4);put(file,bytes);
}
void plane(std::ofstream& file,const gpu::BufferView& view,std::size_t bytes) {
    if(!view.mapped || view.bytes<bytes)throw std::invalid_argument("Incomplete SAR device plane");
    file.write(static_cast<const char*>(view.mapped),static_cast<std::streamsize>(bytes));
}
bool same(const ClockSegment& a,const ClockSegment& b) {
    return a.start==b.start && a.offset==b.offset && a.rate==b.rate;
}
bool same(const SarCapture& a,const SarCapture& b) {
    if(a.input_rate!=b.input_rate || a.decimation!=b.decimation || a.origin!=b.origin ||
       a.recording_rf_hz!=b.recording_rf_hz || a.clock_phase_degrees!=b.clock_phase_degrees ||
       a.channels.size()!=b.channels.size())return false;
    for(std::size_t c=0;c<a.channels.size();++c)
        if(a.channels[c].rf_hz!=b.channels[c].rf_hz || a.channels[c].nco_hz!=b.channels[c].nco_hz ||
           a.channels[c].guard!=b.channels[c].guard)return false;
    return true;
}
}
void SarCapture::validate() const {
    if(!input_rate || input_rate%8'000'000 || input_rate>128'000'000 || decimation!=input_rate/8'000'000 ||
       origin>=exact_limit || !std::isfinite(recording_rf_hz) || recording_rf_hz<=0 ||
       !std::isfinite(clock_phase_degrees) || clock_phase_degrees<=0 || clock_phase_degrees>30 ||
       channels.empty() || channels.size()>64)
        throw std::invalid_argument("Invalid SAR capture coordinates");
    for(const auto& c:channels)
        if(!std::isfinite(c.rf_hz) || c.rf_hz<=0 || !std::isfinite(c.nco_hz) ||
           std::abs(c.nco_hz)+3.5e6>input_rate/2.0 ||
           (c.guard!=4 && c.guard!=8 && c.guard!=16 && c.guard!=32))
            throw std::invalid_argument("Invalid SAR channel coordinates");
}
SarBatch::SarBatch(const SarCapture& c,const SpectralBatch& s,std::vector<ClockSegment> trajectory):
    capture(c),spectra(s),clock(std::move(trajectory)) {
    capture.validate();
    const auto& f=spectra.frame;
    if(f.channel>=capture.channels.size() || f.first>=f.last || f.last>f.symbols.size() ||
       f.carriers.count!=f.symbols.size() || spectra.transmitted.count!=f.symbols.size() ||
       spectra.residual.count!=f.symbols.size() || !std::isfinite(spectra.estimate.reference_origin) ||
       clock.empty() || clock.front().start!=0)
        throw std::invalid_argument("Invalid SAR spectral batch");
    for(std::size_t i=0;i<clock.size();++i) {
        const auto& t=clock[i];
        if(t.start>=exact_limit || !std::isfinite(t.offset) || !std::isfinite(t.rate) || t.rate<=0 ||
           (i && (t.start<=clock[i-1].start || std::abs(t.offset-clock[i-1].offset_at(double(t.start)))>1e-7)))
            throw std::invalid_argument("Discontinuous SAR clock trajectory");
    }
    for(std::size_t row=0;row<f.symbols.size();++row) {
        const auto& symbol=f.symbols[row];
        if(symbol.channel!=f.channel || symbol.segment!=f.segment || !std::isfinite(symbol.window.phase) ||
           !std::isfinite(symbol.window.frequency_hz) || symbol.modulo4>3)
            throw std::invalid_argument("Inconsistent SAR symbol coordinates");
        corrected_adc(row);
    }
}
std::uint64_t SarBatch::corrected_adc(std::size_t row) const {
    const auto sample=spectra.frame.symbols.at(row).window.sample;
    const auto available=(exact_limit-1-capture.origin)/capture.decimation;
    if(available<8192 || sample>available-8192)
        throw std::out_of_range("SAR sample coordinate exceeds exact double range");
    return capture.origin+sample*capture.decimation;
}
double SarBatch::seconds(std::size_t row) const {
    return double(corrected_adc(row))/capture.input_rate;
}
double SarBatch::original_adc(std::size_t row,double within_window) const {
    if(!std::isfinite(within_window) || within_window<0 || within_window>8192)
        throw std::out_of_range("SAR within-window coordinate out of range");
    const auto n=double(corrected_adc(row))+within_window*capture.decimation;
    auto it=std::upper_bound(clock.begin(),clock.end(),n,[](double x,const ClockSegment& s){return x<double(s.start);});
    --it;return n+it->offset_at(n);
}
double SarBatch::carrier_rf_hz(unsigned k) const {
    if(k>=6817)throw std::out_of_range("SAR carrier out of range");
    return capture.channels[spectra.frame.channel].rf_hz+(int(k)-3408)*(8e6/8192);
}
SarWriter::SarWriter(const std::filesystem::path& path,const SarCapture& capture):capture_(capture) {
    capture_.validate();
    static_assert(std::endian::native==std::endian::little && sizeof(C)==8 && sizeof(float)==4 && sizeof(double)==8);
    // C++23 noreplace makes the no-overwrite guarantee atomic.
    file_.exceptions(std::ios::badbit|std::ios::failbit);
    file_.open(path,std::ios::binary|std::ios::out|std::ios::noreplace);
    file_.write("DVBTSAR1",8);
    put(file_,capture.input_rate);put(file_,capture.decimation);put(file_,std::uint32_t(capture.channels.size()));
    put(file_,std::uint32_t{0}); // flags: no UTC, absolute RF calibration or surveyed geometry asserted
    put(file_,capture.origin);put(file_,capture.recording_rf_hz);put(file_,capture.clock_phase_degrees);
    for(const auto& c:capture.channels) {put(file_,c.rf_hz);put(file_,c.nco_hz);put(file_,c.guard);put(file_,std::uint32_t{0});}
}
void SarWriter::write(const SarBatch& batch) {
    if(!file_.is_open() || !same(capture_,batch.capture))throw std::invalid_argument("SAR writer capture changed or closed");
    if(batch.clock.size()<clock_.size() || !std::equal(clock_.begin(),clock_.end(),batch.clock.begin(),
        [](const auto& a,const auto& b){return same(a,b);}))throw std::invalid_argument("Published SAR clock history changed");
    if(batch.clock.size()>clock_.size()) {
        const auto count=std::uint32_t(batch.clock.size()-clock_.size());record(file_,"CLCK",4+24ULL*count);put(file_,count);
        for(std::size_t i=clock_.size();i<batch.clock.size();++i) {
            const auto& s=batch.clock[i];put(file_,s.start);put(file_,s.offset);put(file_,s.rate);
        }
        clock_=batch.clock;
    }
    const auto& s=batch.spectra;const auto& f=s.frame;
    const auto rows=std::uint32_t(f.symbols.size()),frames=std::uint32_t(f.frames.size());
    // Fixed fields, rows, TPS, response, measured/reference/mask planes.
    record(file_,"SPEC",40+32ULL*rows+92ULL*frames+8ULL*6817+20ULL*rows*6817);
    put(file_,std::uint32_t(f.channel));put(file_,std::uint32_t(f.segment));put(file_,rows);
    put(file_,std::uint32_t(f.first));put(file_,std::uint32_t(f.last));put(file_,frames);
    put(file_,s.estimate.reference_origin);put(file_,s.estimate.relative_error);
    for(const auto& row:f.symbols) {
        put(file_,row.window.sample);put(file_,row.window.phase);put(file_,row.window.frequency_hz);
        put(file_,std::uint32_t(row.modulo4));put(file_,std::uint32_t(row.discontinuity));
    }
    for(const auto& t:f.frames) {
        for(auto x:{t.frame,t.constellation,t.rate,t.guard,t.mode,t.hierarchy})put(file_,std::uint32_t(x));
        for(auto bit:t.bits)put(file_,std::uint8_t(bit));
    }
    plane(file_,s.estimate.response,6817*sizeof(C));
    plane(file_,f.carriers.measured,rows*6817ULL*sizeof(C));
    plane(file_,s.transmitted.carriers,rows*6817ULL*sizeof(C));
    plane(file_,s.transmitted.valid,rows*6817ULL*sizeof(std::uint32_t));
    ++batches_;
}
void SarWriter::finish() {
    if(!file_.is_open())throw std::logic_error("SAR writer already closed");
    record(file_,"DONE",8);put(file_,batches_);file_.close();
}
}
