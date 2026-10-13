#pragma once
#include "matched_detector.hpp"

namespace radar {
// Optional tools share production argument validation and receiver coordinates.
struct DetectionSink {MatchedDetector::Projector project;MatchedDetector::Consumer publish;};
using DetectionSinkFactory=std::function<DetectionSink(std::shared_ptr<gpu::Context>,std::size_t,std::ostream&)>;
int detection_cli(int argc,char** argv,const DetectionSinkFactory& ={});
}
