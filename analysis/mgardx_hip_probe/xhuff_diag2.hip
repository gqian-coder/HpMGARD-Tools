// Host-side consistency check of the MGARD-X HIP Huffman codebook/decodebook after a device Compress.
#include <mgard/mgard-x/RuntimeX/RuntimeX.h>
#include <mgard/mgard-x/Lossless/Lossless.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int kind = argc > 1 ? std::atoi(argv[1]) : 1;
    std::mt19937 rng(5); std::normal_distribution<double> N(0.0, 3.0); std::uniform_real_distribution<double> U(0, 1);
    const size_t n = 1u << 20; std::vector<int32_t> v(n), codes(n);
    for (auto& x : v) x = kind == 0 ? 0 : kind == 1 ? ((U(rng) < 0.9) ? 0 : (int32_t)std::lround(N(rng))) : kind == 2 ? (int32_t)std::lround(N(rng) * 20) : kind == 3 ? ((U(rng) < 0.999) ? 0 : (int32_t)std::lround(N(rng))) : (int32_t)std::lround(N(rng) * 5);
    for (size_t i = 0; i < n; ++i) { const uint32_t z = (static_cast<uint32_t>(v[i]) << 1) ^ static_cast<uint32_t>(v[i] >> 31); codes[i] = (int32_t)z; }
    const int dict = 8192; std::vector<unsigned> hist(dict, 0); for (int32_t c : codes) hist[c]++;
    int nz = 0; for (unsigned h : hist) nz += h > 0;
    std::vector<unsigned> hcl(dict, 0);
    { using P = std::pair<unsigned long long, int>; std::vector<std::vector<int>> members; std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
      for (int s = 0; s < dict; ++s) if (hist[s]) { members.push_back({s}); pq.push({hist[s], (int)members.size() - 1}); }
      while (pq.size() > 1) { P a = pq.top(); pq.pop(); P b = pq.top(); pq.pop(); for (int s : members[a.second]) hcl[s]++; for (int s : members[b.second]) hcl[s]++; std::vector<int> m = members[a.second]; m.insert(m.end(), members[b.second].begin(), members[b.second].end()); members.push_back(m); pq.push({a.first + b.first, (int)members.size() - 1}); } }
    double host_cost = 0; for (int s = 0; s < dict; ++s) host_cost += (double)hist[s] * hcl[s];
    std::printf("kind %d: distinct=%d host optimal cost=%.0f bits\n", kind, nz, host_cost);
    mgard_x::DeviceRuntime<mgard_x::HIP>::Initialize();
    mgard_x::Config cfg; cfg.lossless = mgard_x::lossless_type::Huffman; cfg.huff_dict_size = dict; cfg.huff_block_size = 20480; cfg.estimate_outlier_ratio = 1.0; cfg.log_level = 0;
    try {
        mgard_x::Array<1, std::int32_t, mgard_x::HIP> arr({(mgard_x::SIZE)n}); arr.load(codes.data());
        mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> comp;
        mgard_x::ComposedLosslessCompressor<std::int32_t, std::uint64_t, mgard_x::HIP> clc((mgard_x::SIZE)n, cfg);
        try { clc.Compress(arr, comp, 0); mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0); } catch (const std::exception& e) { std::printf("compress EXC: %s\n", e.what()); return 0; }
        auto& ws = clc.huffman.workspace;
        std::vector<uint64_t> cb(dict); std::vector<uint8_t> db(ws.decodebook_array.shape(0));
        mgard_x::MemoryManager<mgard_x::HIP>::Copy1D(cb.data(), ws.codebook_array.data(), dict, 0);
        mgard_x::MemoryManager<mgard_x::HIP>::Copy1D(db.data(), ws.decodebook_array.data(), db.size(), 0);
        mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
        const uint64_t* first = reinterpret_cast<const uint64_t*>(db.data()); const uint64_t* entry = first + 64;
        const unsigned* keys = reinterpret_cast<const unsigned*>(db.data() + sizeof(uint64_t) * 128);
        // codebook: len in top 8 bits, stored code = canonical ^ ((1<<len)-1)
        double dev_cost = 0, kraft = 0; size_t bad_len = 0, decode_fail = 0, zero_len_used = 0; unsigned dev_max = 0;
        for (int s = 0; s < dict; ++s) {
            if (!hist[s]) continue;
            const unsigned len = (unsigned)(cb[s] >> 56); const uint64_t stored = cb[s] & ((1ull << 56) - 1);
            if (len == 0 || len > 56) { bad_len++; continue; }
            dev_max = std::max(dev_max, len); dev_cost += (double)hist[s] * len; kraft += std::ldexp(1.0, -(int)len);
            // emulate Decode.hpp on the stored bits (MSB first)
            uint64_t vv = (stored >> (len - 1)) & 1ull; size_t l = 1, i = 0; bool ok = true;
            while (vv < first[l]) { ++i; if (i >= len) { ok = false; break; } vv = (vv << 1) | ((stored >> (len - 1 - i)) & 1ull); ++l; }
            if (!ok) { decode_fail++; continue; }
            const uint64_t idx = entry[l] + vv - first[l];
            if (l != len || idx >= (uint64_t)dict || keys[idx] != (unsigned)s) decode_fail++;
        }
        std::printf("device codebook: max len=%u cost=%.0f bits (host %.0f) kraft=%.6f bad_len=%zu decode-emulation failures=%zu of %d\n", dev_max, dev_cost, host_cost, kraft, bad_len, decode_fail, nz);
        std::printf("first[1..12]: "); for (int l = 1; l <= 12; ++l) std::printf("%llu ", (unsigned long long)first[l]); std::printf("\nentry[1..12]: "); for (int l = 1; l <= 12; ++l) std::printf("%llu ", (unsigned long long)entry[l]); std::printf("\n");
        int shown = 0; for (int s = 0; s < dict && shown < 8; ++s) if (hist[s]) { std::printf("  sym %d freq %u len %u stored 0x%llx\n", s, hist[s], (unsigned)(cb[s] >> 56), (unsigned long long)(cb[s] & ((1ull << 56) - 1))); shown++; }
        std::printf("compressed bytes=%zu\n", (size_t)comp.shape(0));
    } catch (const std::exception& e) { std::printf("EXC %s\n", e.what()); }
    return 0;
}
