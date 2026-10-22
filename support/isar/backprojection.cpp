// Exact moving-body bistatic backprojection of archived matched profiles.
// Offline support tool: no phase estimation, windowing or DC projection here.
#include "gpu_compute.hpp"
#include "binary_pipe.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

namespace {
using Complex=std::complex<float>;
using Quad=std::array<float,4>;
using Vector=std::array<double,3>;
constexpr unsigned phases=4096,taps=64,block_rows=256;
constexpr double tau=2*std::numbers::pi,c=299792458.,fs=8e6;
constexpr std::uint32_t code[]=
#include "backprojection.inc"
;
constexpr std::uint32_t channel_update_code[]=
#include "backprojection_update.inc"
;
constexpr std::uint32_t reduction[]=
#include "reduce.inc"
;
constexpr std::uint32_t dense_interpolation[]=
#include "interpolate_profiles.inc"
;
struct Parameters {
    unsigned rows,ranges,pixels,phases,block_rows,blocks;
    float k,symbol_duration;
    unsigned oversample;
};
double dot(Vector a,Vector b){double s=0;for(unsigned i=0;i<3;++i)s+=a[i]*b[i];return s;}
double norm(Vector a){return std::sqrt(dot(a,a));}
Vector add(Vector a,Vector b){for(unsigned i=0;i<3;++i)a[i]+=b[i];return a;}
Vector xyz(Quad a){return {a[0],a[1],a[2]};}
Quad quad(Vector a,double w=0){return {float(a[0]),float(a[1]),float(a[2]),float(w)};}
Vector offset(std::span<const Quad> g,Quad pixel,unsigned first){
    Vector d{};for(unsigned axis=0;axis<3;++axis)for(unsigned i=0;i<3;++i)d[i]+=pixel[axis]*double(g[first+axis][i]);return d;
}
double delta_path(Vector tx,Vector rx,Vector d){
    return (2*dot(tx,d)+dot(d,d))/(norm(add(tx,d))+norm(tx))+
           (2*dot(rx,d)+dot(d,d))/(norm(add(rx,d))+norm(rx));
}
double bessel0(double x){
    double sum=1,term=1;
    for(unsigned i=1;i<40;++i){term*=x*x/(4*i*i);sum+=term;if(term<sum*1e-16)break;}
    return sum;
}
double sinc(double x){return x==0 ? 1 : std::sin(std::numbers::pi*x)/(std::numbers::pi*x);}
std::vector<float> interpolation(){
    std::vector<float> table((phases+1)*taps);
    for(unsigned phase=0;phase<=phases;++phase){
        double sum=0;std::array<double,taps> weight;
        for(unsigned tap=0;tap<taps;++tap){
            const double x=double(phase)/phases-double(tap)+31;
            weight[tap]=sinc(x)*bessel0(8*std::sqrt(std::max(0.,1-x*x/1024)))/bessel0(8);
            sum+=weight[tap];
        }
        for(unsigned tap=0;tap<taps;++tap)table[phase*taps+tap]=float(weight[tap]/sum);
    }
    return table;
}

class Backprojection {
    radar::gpu::Kernel kernel_{code,sizeof(Parameters)};
    vk::UniquePipeline update_=kernel_.pipeline(channel_update_code);
    vk::UniquePipeline reduce_=kernel_.pipeline(reduction);
    vk::UniquePipeline interpolate_=kernel_.pipeline(dense_interpolation);
    std::size_t partial_count_=0;
public:
    Backprojection(){const auto table=interpolation();kernel_.upload<float>(3,table);}
    std::span<const Quad> calculate(std::span<const Quad> geometry,std::span<const Complex> profiles,
                                    std::span<const Quad> pixels,unsigned ranges,float k,unsigned block_size=block_rows,
                                    unsigned oversample=0,bool channel_update=false,unsigned channels=5){
        if(geometry.empty() || geometry.size()%10 || pixels.empty() || ranges<64 || ranges>1024 ||
           profiles.size()!=2*(geometry.size()/10)*ranges || !std::isfinite(k) || k<=0 || !block_size ||
           (oversample && (oversample>phases || phases%oversample)) || !channels || channels>5)
            throw std::invalid_argument("Inconsistent backprojection dimensions");
        const unsigned rows=unsigned(geometry.size()/10),blocks=channel_update ? channels : (rows+block_size-1)/block_size;
        const Parameters p{rows,ranges,unsigned(pixels.size()),phases,block_size,blocks,k,.001024F,oversample};
        kernel_.upload<Quad>(0,geometry);kernel_.upload<Complex>(1,profiles);kernel_.upload<Quad>(2,pixels);
        partial_count_=std::size_t(blocks)*pixels.size();
        kernel_.storage(4,partial_count_*sizeof(Quad));kernel_.storage(5,pixels.size_bytes());
        kernel_.storage(6,oversample ? std::size_t(rows)*(ranges-63)*oversample*sizeof(Quad) : sizeof(Quad));
        const auto cmd=kernel_.begin();
        if(oversample){
            kernel_.groups(cmd,*interpolate_,p,((ranges-63)*oversample+63)/64,rows);
            kernel_.barrier(cmd);
        }
        kernel_.groups(cmd,channel_update ? *update_ : kernel_.primary(),p,(p.pixels+63)/64,blocks);
        kernel_.barrier(cmd);kernel_.groups(cmd,*reduce_,p,(p.pixels+63)/64);kernel_.finish(cmd);
        return {static_cast<const Quad*>(kernel_.view(5).mapped),pixels.size()};
    }
    std::span<const Quad> histories() const {return {static_cast<const Quad*>(kernel_.view(4).mapped),partial_count_};}
    std::vector<Quad> update_pixels(std::span<const Quad> geometry,std::span<const Complex> profiles,
                                   std::span<const Quad> pixels,unsigned ranges,std::size_t tile_limit=4000000,unsigned channels=5){
        if(!tile_limit || tile_limit>4000000)throw std::invalid_argument("Invalid bounded pixel batch");
        if(!channels || channels>5)throw std::invalid_argument("Invalid active channel count");
        std::vector<Quad> result(channels*pixels.size());
        for(std::size_t first=0;first<pixels.size();first+=tile_limit){
            const auto tile=pixels.subspan(first,std::min(tile_limit,pixels.size()-first));
            calculate(geometry,profiles,tile,ranges,1.F,unsigned(geometry.size()/10),32,true,channels);
            const auto partial=histories();
            for(unsigned channel=0;channel<channels;++channel)
                std::copy_n(partial.begin()+channel*tile.size(),tile.size(),result.begin()+channel*pixels.size()+first);
        }
        return result;
    }
};

// Deliberately simple double-precision cross-check of the GPU operator.
std::array<double,4> reference(std::span<const Quad> geometry,std::span<const Complex> profiles,
                              Quad pixel,unsigned ranges,double k){
    static const auto table=interpolation();const auto rows=geometry.size()/10;
    std::complex<double> sum{};double information=0,squared=0;
    for(std::size_t row=0;row<rows;++row){
        const auto g=geometry.subspan(10*row,10);const double w=g[9][3];if(w<=0)continue;
        const double row_k=g[5][3]>0 ? g[5][3] : k,gain_power=g[8][3]>0 ? g[8][3] : 1.;
        const auto d=offset(g,pixel,2),tx=xyz(g[0]),rx=xyz(g[1]);
        const double delta=delta_path(tx,rx,d),coordinate=g[9][0]+delta*fs/c;
        const int base=int(std::floor(coordinate));if(base<31 || base+32>=int(ranges))continue;
        const double fraction=(coordinate-base)*phases;const auto phase=std::min(unsigned(fraction),phases-1);
        const double mix=fraction-phase;std::complex<double> zero{},one{};
        for(unsigned tap=0;tap<taps;++tap){
            const auto sample=row*ranges+base+tap-31;
            const double a=table[phase*taps+tap],b=table[(phase+1)*taps+tap],weight=a+(b-a)*mix;
            zero+=weight*std::complex<double>(profiles[sample]);one+=weight*std::complex<double>(profiles[rows*ranges+sample]);
        }
        const auto v=add(xyz(g[5]),offset(g,pixel,6)),a=add(tx,d),b=add(rx,d);
        const double omega=-row_k*(dot(a,v)/norm(a)+dot(b,v)/norm(b))*.001024;
        sum+=w*(zero-std::complex<double>(0,omega)*one)*std::complex<double>(g[9][1],g[9][2])*std::polar(1.,row_k*delta);
        information+=w*gain_power;squared+=w*w*gain_power*gain_power;
    }
    return {sum.real(),sum.imag(),information,squared};
}

int self_test(){
    constexpr unsigned rows=729,ranges=192,count=97;
    constexpr double rf=226.5e6;
    const float k=float(tau*rf/c);
    std::mt19937 random(428713);std::uniform_real_distribution<float> uniform(-1,1);
    std::vector<Quad> pixels(count),geometry(10*rows);
    for(auto& p:pixels)p={150*uniform(random),70*uniform(random),5*uniform(random),0};
    pixels[0]={-37.25F,18.7F,2.9F,0};
    std::vector<Complex> profiles(2*rows*ranges);
    Backprojection gpu;double maximum_reference=0,maximum_point=0,maximum_phase=0,maximum_dense=0,maximum_dense_phase=0;
    for(unsigned mode=0;mode<3;++mode){
        const unsigned moving=mode>0;
        std::complex<double> analytic{};double information=0;
        const std::complex<double> amplitude{.4,.7};
        for(unsigned row=0;row<rows;++row){
            auto g=std::span(geometry).subspan(10*row,10);
            constexpr std::array<double,5> frequency{177.5e6,184.5e6,191.625e6,219.5e6,226.5e6};
            const double row_k=mode==2 ? float(tau*frequency[row%5]/c) : k;
            const auto gain=mode==2 ? std::polar(.5+.1*(row%5),.7*(row%5)) : std::complex<double>(1);
            const double t=(double(row)/rows-.5)*30,angle=moving*.007*t,rate=moving*.007;
            const Vector tx{38000+moving*2*t,6000+moving*t,700},rx{8100+moving*2*t,4000+moving*t,40};
            g[0]=quad(tx,norm(tx));g[1]=quad(rx,norm(rx));
            g[2]=quad({std::cos(angle),std::sin(angle),0});g[3]=quad({-std::sin(angle),std::cos(angle),0});g[4]=quad({0,0,1});
            g[5]=quad({2.*moving,1.*moving,0});g[6]=quad({-rate*std::sin(angle),rate*std::cos(angle),0});
            g[7]=quad({-rate*std::cos(angle),-rate*std::sin(angle),0});g[8]={};
            if(mode==2){g[5][3]=float(row_k);g[8][3]=float(std::norm(gain));}
            const double centre_path=14300+norm(tx)+norm(rx)-std::hypot(38000.,6000.)-std::hypot(8100.,4000.);
            const double phase=row_k*centre_path;
            const double w=row%11==0 ? 0 : 1+.2*std::sin(row*.013);
            const float coordinate=float(96+(centre_path-14300)*fs/c);
            const auto rotation=std::polar(1.,phase)*std::conj(gain);
            g[9]={coordinate,float(rotation.real()),float(rotation.imag()),float(w)};
            const auto d=offset(g,pixels[0],2),v=add(xyz(g[5]),offset(g,pixels[0],6));
            const double delta=delta_path(tx,rx,d),position=coordinate+delta*fs/c;
            const auto a=add(tx,d),b=add(rx,d);
            const double omega=-row_k*(dot(a,v)/norm(a)+dot(b,v)/norm(b))*.001024;
            std::complex<double> m0{},m1{};
            for(unsigned n=0;n<128;++n){const double u=(n+.5)/128-.5;const auto tone=std::polar(1.,omega*u);m0+=tone/128.;m1+=u*tone/128.;}
            for(unsigned sample=0;sample<ranges;++sample){
                const auto value=gain*amplitude*std::polar(1.,-phase-row_k*delta)*sinc(.82*(sample-position));
                profiles[row*ranges+sample]=Complex(value*m0);
                profiles[(rows+row)*ranges+sample]=Complex(value*m1);
            }
            analytic+=double(g[9][3])*std::norm(gain)*amplitude*(m0-std::complex<double>(0,omega)*m1);
            information+=double(g[9][3])*std::norm(gain);
        }
        const auto result=gpu.calculate(geometry,profiles,pixels,ranges,k);
        for(unsigned pixel=0;pixel<count;++pixel){
            const auto expected=reference(geometry,profiles,pixels[pixel],ranges,k);
            const auto value=std::complex<double>(result[pixel][0],result[pixel][1]);
            maximum_reference=std::max(maximum_reference,std::abs(value-std::complex<double>(expected[0],expected[1]))/information);
            if(std::abs(result[pixel][2]-expected[2])/information>2e-6)throw std::runtime_error("Information/erasure mismatch");
        }
        const auto measured=std::complex<double>(result[0][0],result[0][1]);
        maximum_point=std::max(maximum_point,std::abs(measured-analytic)/information);
        maximum_phase=std::max(maximum_phase,std::abs(std::arg(measured/analytic))*180/std::numbers::pi);
        const std::vector<Quad> saved(result.begin(),result.end());
        const auto dense=gpu.calculate(geometry,profiles,pixels,ranges,k,block_rows,32);
        for(unsigned pixel=0;pixel<count;++pixel){
            const auto direct=std::complex<double>(saved[pixel][0],saved[pixel][1]);
            const auto value=std::complex<double>(dense[pixel][0],dense[pixel][1]);
            maximum_dense=std::max(maximum_dense,std::abs(value-direct)/information);
            if(std::abs(direct)>.1*information)
                maximum_dense_phase=std::max(maximum_dense_phase,std::abs(std::arg(value/direct))*180/std::numbers::pi);
            if(std::abs(value-direct)/information>3e-4)
                throw std::runtime_error("Dense interpolation changes the complex image");
            if(std::abs(direct)>.1*information && std::abs(std::arg(value/direct))*180/std::numbers::pi>.01)
                throw std::runtime_error("Dense interpolation changes coherent phase");
        }
        const auto blocked=gpu.calculate(geometry,profiles,pixels,ranges,k,16);
        const auto history=gpu.histories();
        for(unsigned pixel=0;pixel<count;++pixel){
            std::array<double,4> total{};
            for(std::size_t at=pixel;at<history.size();at+=count)
                for(unsigned component=0;component<4;++component)total[component]+=history[at][component];
            for(unsigned component=0;component<4;++component)
                if(std::abs(saved[pixel][component]-blocked[pixel][component])/information>2e-6 ||
                   std::abs(total[component]-blocked[pixel][component])/information>2e-6)
                    throw std::runtime_error("History partition changes the coherent image or information");
        }
        if(mode==2){
            for(unsigned row=0;row<rows;++row){geometry[10*row+6][3]=float(row%5);geometry[10*row+7][3]=row%7 ? 1.F : -1.F;}
            gpu.calculate(geometry,profiles,pixels,ranges,k,rows,32,true);
            const auto flat=std::vector<Quad>(gpu.histories().begin(),gpu.histories().end());
            const auto tiled=gpu.update_pixels(geometry,profiles,pixels,ranges,31);
            for(std::size_t i=0;i<flat.size();++i)for(unsigned j=0;j<4;++j)
                if(std::abs(flat[i][j]-tiled[i][j])/information>1e-7)
                    throw std::runtime_error("Pixel batching changed signed multichannel complex phase or information");
            std::cerr<<"Independent pixel batching: signed five-channel complex images and information unchanged\n";
            for(unsigned channel=0;channel<5;++channel){
                std::vector<Quad> selected_geometry;std::vector<Complex> selected_profiles;
                for(unsigned row=0;row<rows;++row)if(unsigned(geometry[10*row+6][3])==channel){
                    selected_geometry.insert(selected_geometry.end(),geometry.begin()+10*row,geometry.begin()+10*(row+1));
                    selected_geometry[selected_geometry.size()-4][3]=0.F;
                }
                for(unsigned moment=0;moment<2;++moment)for(unsigned row=0;row<rows;++row)if(unsigned(geometry[10*row+6][3])==channel)
                    selected_profiles.insert(selected_profiles.end(),profiles.begin()+(moment*rows+row)*ranges,profiles.begin()+(moment*rows+row+1)*ranges);
                const auto active=gpu.update_pixels(selected_geometry,selected_profiles,pixels,ranges,31,1);
                for(unsigned pixel=0;pixel<count;++pixel)for(unsigned j=0;j<4;++j)
                    if(std::abs(active[pixel][j]-flat[channel*count+pixel][j])/information>1e-7)
                        throw std::runtime_error("Active-band scheduling changed complex phase or information");
            }
            std::cerr<<"Active-band scheduling: signed images and information unchanged\n";
        }
    }
    std::cerr<<"GPU/double maximum complex error per unit information "<<maximum_reference
             <<"\nIndependent analytic moving/static point error "<<maximum_point
             <<"\nMaximum coherent phase error (degrees) "<<maximum_phase<<'\n';
    std::cerr<<"Dense interpolation maximum complex error per unit information "<<maximum_dense
             <<"\nDense interpolation maximum strong-cell phase error (degrees) "<<maximum_dense_phase<<'\n';
    return maximum_reference<2e-4 && maximum_point<2e-4 && maximum_phase<.03 ? 0 : 1;
}
template<class T>T read(std::istream& input){T x;if(&input==&std::cin)radar::support::read_stdin(&x,sizeof(x));else input.read(reinterpret_cast<char*>(&x),sizeof(x));return x;}
template<class T>std::vector<T> read_array(std::istream& input,std::size_t n){
    std::vector<T> values(n);if(&input==&std::cin)radar::support::read_stdin(values.data(),values.size()*sizeof(T));else input.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(T)));return values;
}
// Signed entering/leaving rows update all five complex channel images.
// Keep the sample/geometry ring on the host; no aperture-sized image ring.
int update_channels(const char* path,bool active_only=false){
    std::ifstream coordinates(path,std::ios::binary);coordinates.exceptions(std::ios::badbit|std::ios::failbit);
    const auto count=read<std::uint32_t>(coordinates);
    if(!count || count>8000000)throw std::invalid_argument("Invalid update pixel count");
    const auto pixels=read_array<Quad>(coordinates,count);
    if(coordinates.rdbuf()->sgetc()!=std::char_traits<char>::eof())throw std::invalid_argument("Trailing update pixels");
    std::ios::sync_with_stdio(false);std::cin.tie(nullptr);Backprojection gpu;
    for(;;){
        std::cin.exceptions(std::ios::badbit|std::ios::failbit);
        std::uint32_t rows;if(!radar::support::read_stdin(&rows,sizeof(rows),true))break;const auto ranges=read<std::uint32_t>(std::cin);
        if(!rows || rows>4096 || ranges<64 || ranges>1024)throw std::invalid_argument("Invalid update packet");
        auto geometry=read_array<Quad>(std::cin,10*std::size_t(rows));
        const auto profiles=read_array<Complex>(std::cin,2*std::size_t(rows)*ranges);
        std::array<double,5> expected{},squared{},absolute{};
        std::array<bool,5> present{};
        for(unsigned row=0;row<rows;++row){
            const auto g=std::span(geometry).subspan(10*row,10);
            if(g[6][3]<0 || g[6][3]>4 || std::floor(g[6][3])!=g[6][3] || std::abs(g[7][3])!=1 || g[9][3]<0)
                throw std::invalid_argument("Invalid signed channel geometry");
            const auto channel=unsigned(g[6][3]);const double w=g[9][3];
            present[channel]=true;
            expected[channel]+=g[7][3]*w;absolute[channel]+=w;squared[channel]+=w*w;
        }
        std::vector<unsigned> channels;
        for(unsigned c=0;c<5;++c)if(!active_only || present[c])channels.push_back(c);
        if(active_only)for(unsigned row=0;row<rows;++row){
            auto& c=geometry[10*row+6][3];c=float(std::lower_bound(channels.begin(),channels.end(),unsigned(c))-channels.begin());
        }
        const auto bands=unsigned(channels.size());
        const auto history=gpu.update_pixels(geometry,profiles,pixels,ranges,4000000,bands);radar::support::write_stdout(&bands,4);
        std::vector<Complex> compact(count);
        for(unsigned index=0;index<bands;++index){
            const auto channel=channels[index];
            const auto result=std::span(history).subspan(std::size_t(index)*count,count);
            for(unsigned pixel=0;pixel<count;++pixel){
                if(std::abs(double(result[pixel][2])-expected[channel])>1e-5*absolute[channel]+1e-28 ||
                   std::abs(double(result[pixel][3])-squared[channel])>1e-5*squared[channel]+1e-28){
                    std::cerr<<"Channel "<<channel<<", pixel "<<pixel<<", rows "<<rows
                             <<": information "<<result[pixel][2]<<" expected "<<expected[channel]
                             <<", squared information "<<result[pixel][3]<<" expected "<<squared[channel]<<'\n';
                    throw std::runtime_error("Update viewport leaves complete original range support");
                }
                compact[pixel]={result[pixel][0],result[pixel][1]};
            }
            if(active_only)radar::support::write_stdout(&channel,sizeof(channel));
            radar::support::write_stdout(&expected[channel],sizeof(double));
            radar::support::write_stdout(compact.data(),compact.size()*sizeof(Complex));
        }
        std::cin.exceptions(std::ios::badbit);
    }
    return 0;
}
int stream(const char* path){
    std::ifstream coordinates(path,std::ios::binary);coordinates.exceptions(std::ios::badbit|std::ios::failbit);
    const auto count=read<std::uint32_t>(coordinates);
    if(!count || count>4000000)throw std::invalid_argument("Invalid streaming pixel count");
    const auto pixels=read_array<Quad>(coordinates,count);
    if(coordinates.rdbuf()->sgetc()!=std::char_traits<char>::eof())throw std::invalid_argument("Trailing pixel data");
    std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
    Backprojection gpu;
    for(;;){
        std::cin.exceptions(std::ios::badbit|std::ios::failbit);
        std::uint32_t rows;if(!radar::support::read_stdin(&rows,sizeof(rows),true))break;const auto ranges=read<std::uint32_t>(std::cin);
        if(!rows || rows>4096 || ranges<64 || ranges>1024)
            throw std::invalid_argument("Invalid bounded streaming block");
        const auto geometry=read_array<Quad>(std::cin,10*std::size_t(rows));
        const auto profiles=read_array<Complex>(std::cin,2*std::size_t(rows)*ranges);
        gpu.calculate(geometry,profiles,pixels,ranges,1.F,128,32);
        const auto result=gpu.histories();
        const std::uint32_t blocks=(rows+127)/128;
        radar::support::write_stdout(&blocks,4);
        std::vector<Complex> compact(count);
        for(unsigned block=0;block<blocks;++block){
            const auto values=result.subspan(std::size_t(block)*count,count);
            const double information=values.front()[2];
            for(unsigned pixel=0;pixel<count;++pixel){
                if(std::abs(double(values[pixel][2])-information)>4e-6*information+1e-20)
                    throw std::runtime_error("Streaming viewport leaves complete range support");
                compact[pixel]={values[pixel][0],values[pixel][1]};
            }
            radar::support::write_stdout(&information,sizeof(information));
            radar::support::write_stdout(compact.data(),compact.size()*sizeof(Complex));
        }
        std::cin.exceptions(std::ios::badbit);
    }
    return 0;
}
}

int main(int argc,char** argv){try{
    static_assert(std::endian::native==std::endian::little && sizeof(Complex)==8 && sizeof(Quad)==16);
    if(argc==2 && std::string_view(argv[1])=="--self-test")return self_test();
    if(argc==3 && std::string_view(argv[1])=="--update-channels")return update_channels(argv[2]);
    if(argc==3 && std::string_view(argv[1])=="--update-active-channels")return update_channels(argv[2],true);
    if(argc==3 && std::string_view(argv[1])=="--stream-coherent")return stream(argv[2]);
    const bool fast=argc==4 && std::string_view(argv[3])=="--fast-histories";
    const bool histories=fast || (argc==4 && std::string_view(argv[3])=="--histories");
    if(argc!=3 && !histories)throw std::invalid_argument("isar_backprojection INPUT.bin OUTPUT.bin [--histories|--fast-histories]; or --self-test");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    const auto magic=read<std::array<char,8>>(input);
    if(std::string_view(magic.data(),8)!=std::string_view("ISARBP1\0",8))throw std::invalid_argument("Invalid ISAR input");
    const auto rows=read<std::uint32_t>(input),ranges=read<std::uint32_t>(input),pixels=read<std::uint32_t>(input);
    const auto k=read<float>(input);
    if(!rows || rows>1000000 || ranges<64 || ranges>1024 || !pixels || pixels>1000000 || !std::isfinite(k) || k<=0)
        throw std::invalid_argument("Invalid ISAR dimensions");
    const auto geometry=read_array<Quad>(input,10*std::size_t(rows)),positions=read_array<Quad>(input,pixels);
    const auto profiles=read_array<Complex>(input,2*std::size_t(rows)*ranges);
    if(input.rdbuf()->sgetc()!=std::char_traits<char>::eof())throw std::invalid_argument("Unexpected trailing ISAR input");
    auto finite=[](const Quad& x){return std::ranges::all_of(x,[](float v){return std::isfinite(v);});};
    if(!std::ranges::all_of(geometry,finite) || !std::ranges::all_of(positions,finite) ||
       !std::ranges::all_of(profiles,[](Complex z){return std::isfinite(z.real()) && std::isfinite(z.imag());}))
        throw std::invalid_argument("Nonfinite ISAR input");
    for(unsigned row=0;row<rows;++row)if(geometry[10*row+9][3]<0 || geometry[10*row][3]<=0 || geometry[10*row+1][3]<=0)
        throw std::invalid_argument("Invalid information or centre geometry");
    const unsigned rows_per_block=histories ? 16 : block_rows;
    Backprojection gpu;auto result=gpu.calculate(geometry,profiles,positions,ranges,k,rows_per_block,fast ? 32 : 0);
    std::ofstream output(argv[2],std::ios::binary|std::ios::noreplace);output.exceptions(std::ios::badbit|std::ios::failbit);
    if(histories){
        result=gpu.histories();const std::uint32_t blocks=(rows+rows_per_block-1)/rows_per_block;
        output.write("ISARHI1\0",8);output.write(reinterpret_cast<const char*>(&blocks),sizeof(blocks));
        output.write(reinterpret_cast<const char*>(&rows_per_block),sizeof(rows_per_block));
    }else output.write("ISARIM1\0",8);
    output.write(reinterpret_cast<const char*>(&pixels),sizeof(pixels));
    output.write(reinterpret_cast<const char*>(result.data()),std::streamsize(result.size_bytes()));
    std::cerr<<"Backprojected "<<rows<<" rows into "<<pixels<<" pixels; no DC projection or phase fitting\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
