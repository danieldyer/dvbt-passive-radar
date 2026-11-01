// Qualification of the typed MCAP proposal against existing binary profile
// records. This is a format study, not a selectable production archiver.
#define MCAP_IMPLEMENTATION
#define MCAP_COMPRESSION_NO_LZ4
#define MCAP_COMPRESSION_NO_ZSTD
#include "range_profiles_generated.h"
#include <boost/json.hpp>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <flatbuffers/verifier.h>
#include <fstream>
#include <iostream>
#include <mcap/writer.hpp>
#include <span>
#include <unistd.h>
#include <vector>
namespace fb = pl64::radar::fb;
using Clock = std::chrono::steady_clock;
double cpu() {
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
// Synchronous sink: all short writes and storage errors propagate in Release.
class Sink final : public mcap::IWritable {
    std::ofstream file_;
    std::uint64_t size_{};
    void handleWrite(const std::byte *p, std::uint64_t n) override {
        file_.write(reinterpret_cast<const char *>(p), std::streamsize(n));
        size_ += n;
    }

  public:
    explicit Sink(const std::filesystem::path &path) {
        file_.exceptions(std::ios::badbit | std::ios::failbit);
        file_.open(path, std::ios::binary | std::ios::noreplace);
    }
    std::uint64_t size() const override {
        return size_;
    }
    void flush() override {
        file_.flush();
    }
    void end() override {
        file_.close();
    }
};
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("archive_format_benchmark SCHEMA.bfbs OUTPUT_DIRECTORY");
        const std::filesystem::path dir = argv[2];
        if (!std::filesystem::create_directory(dir))
            throw std::runtime_error("Output exists");
        std::ifstream schema_file(argv[1], std::ios::binary);
        const std::string schema{std::istreambuf_iterator<char>(schema_file), {}};
        if (schema.size() < 8)
            throw std::runtime_error("Missing binary schema");
        constexpr unsigned rows = 952, lags = 3072, repeats = 12;
        std::vector<fb::Complex32> values(2ULL * rows * lags), response(6817);
        std::vector<float> noise(6817, 1);
        std::vector<fb::ProfileRow> coordinates;
        boost::json::array row_json;
        for (unsigned i = 0; i < rows; ++i) {
            coordinates.emplace_back((1ULL << 35) + i * 8704, 12 + i * .001088, 123456789012. + i * 8704,
                                     .13 + i * .0001, -347.25, .125, 1e-6, true);
            row_json.emplace_back(boost::json::array{(1ULL << 35) + i * 8704, 12 + i * .001088,
                                                     123456789012. + i * 8704, .13 + i * .0001, -347.25, true,
                                                     .125, 1e-6});
        }
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = fb::Complex32(float(i % 4021), -float(i % 2039));
        const auto metadata = boost::json::serialize(boost::json::object{{"channel", 2},
                                                                         {"segment", 1},
                                                                         {"count", rows},
                                                                         {"size", lags},
                                                                         {"information", 1.25},
                                                                         {"reference_origin", 128.375},
                                                                         {"rows", row_json}});
        const auto payload_bytes =
            (values.size() + response.size()) * sizeof(fb::Complex32) + noise.size() * sizeof(float);
        const auto create = [&](flatbuffers::FlatBufferBuilder &b) {
            b.Clear();
            const auto v = b.CreateVectorOfStructs(values), h = b.CreateVectorOfStructs(response);
            const auto n = b.CreateVector(noise);
            const auto r = b.CreateVectorOfStructs(coordinates);
            fb::FinishProfileBatchBuffer(b, fb::CreateProfileBatch(b, 2, 1, lags, 1.25, 128.375, r, v, h, n));
        };
        boost::json::array results;
        for (unsigned mode = 0; mode < 4; ++mode) {
            const auto path = dir / (std::to_string(mode) + ".bin");
            Sink sink(path);
            mcap::McapWriter writer;
            flatbuffers::FlatBufferBuilder builder(payload_bytes + 1024 * 1024);
            mcap::Channel channel;
            if (mode) {
                mcap::McapWriterOptions options("");
                options.compression = mcap::Compression::None;
                options.chunkSize = 64 * 1024;
                options.enableDataCRC = mode == 1;
                options.noChunkCRC = mode == 3;
                writer.open(sink, options);
                mcap::Schema definition("pl64.radar.fb.ProfileBatch", "flatbuffer", schema);
                writer.addSchema(definition);
                channel = mcap::Channel("/radar/profiles/2", "flatbuffer", definition.id);
                writer.addChannel(channel);
            }
            const auto start = Clock::now();
            const auto started_cpu = cpu();
            double serialize_cpu = 0;
            for (unsigned i = 0; i < repeats; ++i) {
                if (mode) {
                    const auto before = cpu();
                    create(builder);
                    serialize_cpu += cpu() - before;
                    flatbuffers::Verifier verify(builder.GetBufferPointer(), builder.GetSize());
                    if (i == 0 && !fb::VerifyProfileBatchBuffer(verify))
                        throw std::runtime_error("Invalid FlatBuffer");
                    mcap::Message msg;
                    msg.channelId = channel.id;
                    msg.sequence = i;
                    msg.logTime = 1000000000ULL * i;
                    msg.publishTime = msg.logTime;
                    msg.data = reinterpret_cast<const std::byte *>(builder.GetBufferPointer());
                    msg.dataSize = builder.GetSize();
                    const auto status = writer.write(msg);
                    if (!status.ok())
                        throw std::runtime_error(status.message);
                } else {
                    const std::uint32_t length = metadata.size();
                    const std::uint64_t n = payload_bytes;
                    sink.write(reinterpret_cast<const std::byte *>("PROF"), 4);
                    sink.write(reinterpret_cast<const std::byte *>(&length), 4);
                    sink.write(reinterpret_cast<const std::byte *>(&n), 8);
                    sink.write(reinterpret_cast<const std::byte *>(metadata.data()), metadata.size());
                    sink.write(reinterpret_cast<const std::byte *>(values.data()),
                               values.size() * sizeof(values[0]));
                    sink.write(reinterpret_cast<const std::byte *>(response.data()),
                               response.size() * sizeof(response[0]));
                    sink.write(reinterpret_cast<const std::byte *>(noise.data()),
                               noise.size() * sizeof(float));
                }
            }
            if (mode)
                writer.close();
            else
                sink.end();
            const auto cpu_elapsed = cpu() - started_cpu;
            const double wall = std::chrono::duration<double>(Clock::now() - start).count();
            const double input_seconds =
                repeats * rows * .001088 / 3; // Three channels at the measured batch size.
            results.emplace_back(boost::json::object{
                {"mode", mode == 0   ? "binary-current"
                         : mode == 1 ? "mcap-navigation-dual-crc"
                         : mode == 2 ? "mcap-chunk-crc"
                                     : "mcap-no-crc"},
                {"bytes", sink.size()},
                {"payload_bytes", payload_bytes * repeats},
                {"input_seconds", input_seconds},
                {"cpu_ms_per_input_second", 1000 * cpu_elapsed / input_seconds},
                {"wall_ms_per_input_second", 1000 * wall / input_seconds},
                {"flatbuffer_cpu_ms_per_input_second", 1000 * serialize_cpu / input_seconds}});
            std::filesystem::remove(path);
        }
        std::cout << boost::json::serialize(results) << '\n';
        std::filesystem::remove(dir);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
