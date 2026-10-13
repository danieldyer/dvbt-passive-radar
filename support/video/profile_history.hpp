#pragma once
#include "complex_sample.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace radar::video {
// Offline replay of the archived detector's consecutive-row/erasure policy.
// Values already carry the capture's correction: never rotate/resample again.
class ProfileHistory {
    struct Row {std::array<C,2048> values;double time,information;};
    std::deque<Row> rows_;
    std::size_t required_;
    double interval_,update_,last_publish_=-std::numeric_limits<double>::infinity();
    std::uint64_t segment_=UINT64_MAX;
    bool erasures_;
    static std::size_t required(double integration,double interval,double update){
        if(!std::isfinite(integration) || integration<.5 || integration>10 || !std::isfinite(interval) || interval<=0 ||
           !std::isfinite(update) || update<=0 || integration/interval>10000)throw std::invalid_argument("Invalid profile history coordinates");
        return std::size_t(std::ceil(integration/interval));
    }
public:
    ProfileHistory(double integration,double interval,double update,bool erasures=false):
        required_(required(integration,interval,update)),interval_(interval),update_(update),erasures_(erasures){}
    bool append(double time,double information,bool valid,std::uint64_t segment,std::span<const C> zero,std::span<const C> one){
        if(!std::isfinite(time) || !std::isfinite(information) || information<=0 || zero.size()!=1024 || one.size()!=1024)
            throw std::invalid_argument("Invalid archived profile row");
        if(segment!=segment_){rows_.clear();segment_=segment;last_publish_=-std::numeric_limits<double>::infinity();}
        if((!valid && !erasures_) || (!rows_.empty() && std::abs(time-rows_.back().time-interval_)>interval_*.01))rows_.clear();
        if(!valid && !erasures_)return false;
        auto& row=rows_.emplace_back();row.time=time;row.information=valid ? information : 0;
        if(valid){std::ranges::copy(zero,row.values.begin());std::ranges::copy(one,row.values.begin()+1024);}
        else row.values.fill(C{});
        if(rows_.size()>required_)rows_.pop_front();
        if(rows_.size()!=required_ || time-last_publish_+interval_*1e-8<update_)return false;
        if(std::ranges::none_of(rows_,[](const Row& r){return r.information>0;}))return false;
        last_publish_=time;return true;
    }
    std::size_t required() const{return required_;}
    double first_time() const{return rows_.front().time;}
    void snapshot(std::vector<C>& moments,std::vector<double>& information) const{
        if(rows_.size()!=required_)throw std::logic_error("Incomplete archived coherent interval");
        moments.resize(2*required_*1024);information.resize(required_);
        for(std::size_t r=0;r<required_;++r){information[r]=rows_[r].information;
            for(unsigned moment=0;moment<2;++moment)
                std::copy_n(rows_[r].values.data()+moment*1024,1024,moments.data()+(moment*required_+r)*1024);
        }
    }
};
}
