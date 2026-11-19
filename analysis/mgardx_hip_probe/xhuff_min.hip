// Minimal MGARD-X HIP Huffman round trip (encode on HIP, decode on HIP) to test header patches without rebuilding MGARD.
// usage: xhuff_min <kind 1 sparse90|2 wide|3 sparse999> <zstd_level>
#include <mgard/mgard-x/RuntimeX/RuntimeX.h>
#include <mgard/mgard-x/Lossless/Lossless.hpp>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int kind = argc > 1 ? std::atoi(argv[1]) : 1; const int lvl = argc > 2 ? std::atoi(argv[2]) : 0;
    std::mt19937 rng(5); std::normal_distribution<double> N(0.0, 3.0); std::uniform_real_distribution<double> U(0, 1);
    const size_t n = 1u << 20; std::vector<int32_t> v(n), codes(n);
    for (auto& x : v) x = kind == 0 ? 0 : kind == 1 ? ((U(rng) < 0.9) ? 0 : (int32_t)std::lround(N(rng))) : kind == 2 ? (int32_t)std::lround(N(rng) * 20) : ((U(rng) < 0.999) ? 0 : (int32_t)std::lround(N(rng)));
    for (size_t i = 0; i < n; ++i) { const uint32_t z = (static_cast<uint32_t>(v[i]) << 1) ^ static_cast<uint32_t>(v[i] >> 31); codes[i] = (int32_t)z; }
    mgard_x::DeviceRuntime<mgard_x::HIP>::Initialize();
    mgard_x::Config cfg; cfg.lossless = lvl > 0 ? mgard_x::lossless_type::Huffman_Zstd : mgard_x::lossless_type::Huffman;
    cfg.huff_dict_size = 8192; cfg.huff_block_size = 20480; cfg.zstd_compress_level = lvl > 0 ? lvl : 3; cfg.estimate_outlier_ratio = 1.0; cfg.log_level = 0;
    std::printf("kind %d lvl %d: ", kind, lvl);
    try {
        mgard_x::Array<1, std::int32_t, mgard_x::HIP> arr({(mgard_x::SIZE)n}); arr.load(codes.data());
        mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> comp;
        mgard_x::ComposedLosslessCompressor<std::int32_t, std::uint64_t, mgard_x::HIP> clc((mgard_x::SIZE)n, cfg);
        clc.Compress(arr, comp, 0); clc.Serialize(comp, 0); mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
        const size_t bytes = comp.shape(0); std::printf("hip-enc ok (%zu B) ", bytes);
        mgard_x::Array<1, std::int32_t, mgard_x::HIP> dec({(mgard_x::SIZE)n});
        mgard_x::ComposedLosslessCompressor<std::int32_t, std::uint64_t, mgard_x::HIP> clc2((mgard_x::SIZE)n, cfg);
        clc2.Deserialize(comp, 0); clc2.Decompress(comp, dec, 0); mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
        const std::int32_t* p = dec.hostCopy(false, 0); mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
        bool match = true; for (size_t i = 0; i < n && match; ++i) if (p[i] != codes[i]) match = false;
        std::printf("hip-dec %s\n", match ? "MATCH" : "MISMATCH");
    } catch (const std::exception& e) { std::printf("EXC %s\n", e.what()); }
    return 0;
}
