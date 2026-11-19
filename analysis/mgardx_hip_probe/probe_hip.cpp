// probe_hip <kind:0 zeros|1 sparse90|2 wide|3 sparse999> <zstd_level> <mode:A enc-compare|B serial-enc->hip-dec|C hip-enc->serial-dec|D hip-enc->hip-dec>
#include <hpmgard/codec/Codec.h>
#include <hpmgard/codec/MgardXHuff.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace hpmgard::codec;
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int kind = argc > 1 ? std::atoi(argv[1]) : 1; const int lvl = argc > 2 ? std::atoi(argv[2]) : 0; const char mode = argc > 3 ? argv[3][0] : 'A';
    std::mt19937 rng(5); std::normal_distribution<double> N(0.0, 3.0); std::uniform_real_distribution<double> U(0, 1);
    const char* names[] = {"zeros", "sparse90", "wide", "sparse999"};
    const size_t n = 1u << 20; std::vector<int32_t> v(n);
    for (auto& x : v) x = kind == 0 ? 0 : kind == 1 ? ((U(rng) < 0.9) ? 0 : (int32_t)std::lround(N(rng))) : kind == 2 ? (int32_t)std::lround(N(rng) * 20) : ((U(rng) < 0.999) ? 0 : (int32_t)std::lround(N(rng)));
    std::printf("%-9s lvl %d mode %c: ", names[kind], lvl, mode);
    std::vector<uint8_t> bh, bs; std::vector<int32_t> back(n, 0);
    auto enc = [&](int dev, std::vector<uint8_t>& b) { const bool ok = xhuff::Encode(dev, v.data(), n, lvl, b); std::printf("%s-enc %s (%zu B) ", dev ? "hip" : "serial", ok ? "ok" : "FAIL", b.size()); return ok; };
    auto dec = [&](int dev, const std::vector<uint8_t>& b) { const bool ok = xhuff::Decode(dev, b.data(), b.size(), back.data(), n, lvl); std::printf("%s-dec %s %s", dev ? "hip" : "serial", ok ? "ok" : "FAIL", (ok && back == v) ? "MATCH" : "MISMATCH"); };
    switch (mode) {
    case 'A': if (enc(1, bh) && enc(0, bs)) std::printf("streams %s", bh == bs ? "IDENTICAL" : "DIFFER"); break;
    case 'B': if (enc(0, bs)) dec(1, bs); break;
    case 'C': if (enc(1, bh)) dec(0, bh); break;
    case 'D': if (enc(1, bh)) dec(1, bh); break;
    }
    std::printf("\n");
}
