#include "recording_metadata.hpp"
#include <array>
#include <iostream>
#include <boost/json.hpp>

using namespace radar;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F> void rejects(F action) {
    bool failed=false;try{action();}catch(const std::exception&){failed=true;}
    check(failed,"Invalid recording declaration was accepted");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Expected metadata fixture");
        const auto path=std::filesystem::temp_directory_path()/"radar-ci16-reader-test.wav";
        struct Remove {std::filesystem::path path;~Remove(){std::filesystem::remove(path);}} remove{path};
        const std::array<std::int16_t,10> samples{-32768,32767,-1,1,0,0,12345,-23456,32767,-32768};
        {
            std::ofstream file(path,std::ios::binary);
            auto put=[&](auto x){file.write(reinterpret_cast<const char*>(&x),sizeof(x));};
            file.write("RIFF",4);put(std::uint32_t{36+sizeof(samples)});file.write("WAVEfmt ",8);put(std::uint32_t{16});
            put(std::uint16_t{1});put(std::uint16_t{2});put(std::uint32_t{61'000'000});put(std::uint32_t{244'000'000});
            put(std::uint16_t{4});put(std::uint16_t{16});file.write("data",4);put(std::uint32_t{sizeof(samples)});
            file.write(reinterpret_cast<const char*>(samples.data()),sizeof(samples));
        }
        WaveReader wave(path);check(wave.info.sample_rate==61'000'000 && wave.info.samples==5 && wave.info.sample_bytes==4,"Wrong ci16 WAV coordinates");
        std::array<C,5> output;wave.read(0,output);
        for(unsigned j=0;j<output.size();++j)
            check(output[j]==C{samples[2*j]/32768.f,samples[2*j+1]/32768.f},"Int16 conversion changed ADC bits or I/Q order");
        std::array<C,2> seek;wave.read(3,seek);check(seek[0]==output[3] && seek[1]==output[4],"Int16 seek used wrong sample stride");
        rejects([&]{wave.read(4,seek);});
        const auto metadata=RecordingMetadata::read(argv[1]);metadata.validate_wave(wave.info);
        SarCapture capture;metadata.configure(capture);
        check(capture.channels.size()==5 && capture.channels[0].nco_hz==-24'500'000 && capture.channels[4].nco_hz==24'500'000,"Metadata failed to configure VHF channel bank");
        capture.input_rate=64'000'000;capture.decimation=8;capture.source_rate=61'000'000;capture.validate();
        SarCapture selected;metadata.configure(selected,"nine");
        check(selected.channels.size()==1 && selected.channels[0].rf_hz==191'625'000 && selected.channels[0].nco_hz==-10'375'000,"Wrong physical RF or NCO for selected signal");
        rejects([&]{metadata.configure(capture);});
        rejects([&]{SarCapture c;metadata.configure(c,"unknown");});
        auto mismatch=wave.info;mismatch.sample_rate=56'000'000;rejects([&]{metadata.validate_wave(mismatch);});
        mismatch=wave.info;mismatch.sample_bytes=8;rejects([&]{metadata.validate_wave(mismatch);});
        const auto usdr_path=path.parent_path()/"radar-usdr-metadata-test.json";Remove remove_usdr{usdr_path};
        auto document=boost::json::parse(metadata.json());
        document.as_object()["capture"]=boost::json::object{{"driver","usdr"}};
        auto save=[&]{std::ofstream file(usdr_path);file<<boost::json::serialize(document);};save();
        auto usdr=RecordingMetadata::read(usdr_path);
        check(usdr.capture->driver=="usdr" && usdr.capture->device_args=="bus=pcie" && usdr.capture->antenna=="auto" && usdr.capture->bandwidth_hz==28e6,"Wrong uSDR metadata defaults");
        usdr.write(usdr_path);usdr=RecordingMetadata::read(usdr_path);
        check(usdr.capture->driver=="usdr","Saved metadata lost capture driver");
        document.at("capture").as_object().erase("driver");save();
        check(RecordingMetadata::read(usdr_path).capture->driver=="uhd","Legacy metadata changed default driver");
        document.at("capture").as_object()["driver"]="typo";save();rejects([&]{RecordingMetadata::read(usdr_path);});
        std::cout<<"Exact ci16 I/Q, seek, bounds, five-channel metadata and declaration/header consistency passed\n";
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
