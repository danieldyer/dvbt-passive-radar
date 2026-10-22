#include "capture_source.hpp"
namespace radar {
std::unique_ptr<CaptureSource> make_capture_source(const SdrCaptureSettings& s) {
#ifdef RADAR_HAS_USDR
    if(s.driver=="usdr")return make_usdr_source(s);
#endif
#ifdef RADAR_HAS_UHD
    if(s.driver=="uhd")return make_uhd_source(s);
#endif
    throw std::invalid_argument("Capture driver was not built: "+s.driver);
}
}
