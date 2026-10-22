#include "profile_history.hpp"
#include <iostream>

namespace {void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}}
int main(){try{
    using namespace radar;video::ProfileHistory history(.5,.125,.25);
    std::array<C,1024> zero{},one{};
    for(unsigned n=0;n<6;++n){
        for(unsigned k=0;k<1024;++k){zero[k]={float(n)+k*.01F,-float(n)-k*.02F};one[k]={100+float(n)+k*.01F,-100-float(n)-k*.02F};}
        const auto ready=history.append(10+n*.125,1+n,true,0,zero,one);
        check(ready==(n==3 || n==5),"Wrong full-CPI admission or update cadence");
    }
    std::vector<C> moments;std::vector<double> information;history.snapshot(moments,information);
    check(history.required()==4 && history.first_time()==10.25 && information==std::vector<double>({3,4,5,6}),"Changed interval or archived noise weights");
    for(unsigned r=0;r<4;++r)for(unsigned k=0;k<1024;++k){
        check(moments[r*1024+k]==C{2+float(r)+k*.01F,-2-float(r)-k*.02F},"Reordered or rotated complex zero moments");
        check(moments[(4+r)*1024+k]==C{102+float(r)+k*.01F,-102-float(r)-k*.02F},"Reordered or rotated complex first moments");
    }
    check(!history.append(10.75,1,false,0,zero,one),"Invalid oracle produced a map");
    for(unsigned n=0;n<4;++n)check(history.append(10.875+n*.125,1,true,0,zero,one)==(n==3),"CPI bridged an invalid row");
    for(unsigned n=0;n<4;++n)check(history.append(20+n*.125,1,true,0,zero,one)==(n==3),"CPI bridged a timestamp gap");
    for(unsigned n=0;n<4;++n)check(history.append(30+n*.125,1,true,1,zero,one)==(n==3),"CPI bridged an acquisition reset");
    bool rejected=false;try{video::ProfileHistory invalid(.5,0,.25);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"Accepted an invalid sample interval");
    video::ProfileHistory masked(.5,.125,.125,true);
    for(unsigned n=0;n<8;++n)check(masked.append(40+n*.125,2,n%3!=1,0,zero,one)==(n>=3),"Decoder erasure discarded a fixed-grid CPI");
    masked.snapshot(moments,information);
    check(information==std::vector<double>({0,2,2,0}),"Erased observations retained information");
    for(unsigned r:{0U,3U})for(unsigned k=0;k<1024;++k)
        check(moments[r*1024+k]==C{} && moments[(4+r)*1024+k]==C{},"Erased observations retained unknown phase");
    for(unsigned n=0;n<4;++n)check(masked.append(50+n*.125,2,true,0,zero,one)==(n==3),"Erasure policy bridged a sample-clock gap");
    video::ProfileHistory extended(.5,.125,.25,true);
    std::vector<C> wide_zero(4096),wide_one(4096);
    wide_zero[2668]={.2F,.3F};wide_one[2668]={-.1F,.4F};
    for(unsigned n=0;n<4;++n)extended.append(n*.125,1,n!=1,0,wide_zero,wide_one);
    extended.snapshot(moments,information);
    check(moments.size()==2*4*4096 && moments[2668]==wide_zero[2668] && moments[4*4096+2668]==wide_one[2668],
        "Replay truncated or changed a 100 km echo");
    check(moments[4096+2668]==C{} && moments[5*4096+2668]==C{},"Extended replay retained an erased echo");
    std::cout<<"Fixed CPI, archived complex phase/moments/weights, cadence, invalid rows, gaps and segments: PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
