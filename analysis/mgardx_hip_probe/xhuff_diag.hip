// Diagnose the MGARD-X HIP Huffman codebook stages: histogram, sort-by-key, code lengths (vs host computation).
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
    for (auto& x : v) x = kind == 0 ? 0 : kind == 1 ? ((U(rng) < 0.9) ? 0 : (int32_t)std::lround(N(rng))) : kind == 2 ? (int32_t)std::lround(N(rng) * 20) : ((U(rng) < 0.999) ? 0 : (int32_t)std::lround(N(rng)));
    for (size_t i = 0; i < n; ++i) { const uint32_t z = (static_cast<uint32_t>(v[i]) << 1) ^ static_cast<uint32_t>(v[i] >> 31); codes[i] = (int32_t)z; }
    const int dict = 8192; std::vector<unsigned> hist(dict, 0); for (int32_t c : codes) hist[c]++;
    int nz = 0; for (unsigned h : hist) nz += h > 0;
    // host Huffman code lengths
    std::vector<unsigned> hcl(dict, 0);
    { using P = std::pair<unsigned long long, int>; std::vector<std::vector<int>> members; std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
      for (int s = 0; s < dict; ++s) if (hist[s]) { members.push_back({s}); pq.push({hist[s], (int)members.size() - 1}); }
      while (pq.size() > 1) { P a = pq.top(); pq.pop(); P b = pq.top(); pq.pop(); for (int s : members[a.second]) hcl[s]++; for (int s : members[b.second]) hcl[s]++; std::vector<int> m = members[a.second]; m.insert(m.end(), members[b.second].begin(), members[b.second].end()); members.push_back(m); pq.push({a.first + b.first, (int)members.size() - 1}); } }
    unsigned host_max_cl = *std::max_element(hcl.begin(), hcl.end());
    std::printf("kind %d: n=%zu distinct=%d host max CL=%u\n", kind, n, nz, host_max_cl);
    mgard_x::DeviceRuntime<mgard_x::HIP>::Initialize();
    mgard_x::Config cfg; cfg.lossless = mgard_x::lossless_type::Huffman; cfg.huff_dict_size = dict; cfg.huff_block_size = 20480; cfg.estimate_outlier_ratio = 1.0; cfg.log_level = 0;
    try {
        mgard_x::Array<1, std::int32_t, mgard_x::HIP> arr({(mgard_x::SIZE)n}); arr.load(codes.data());
        mgard_x::Array<1, mgard_x::Byte, mgard_x::HIP> comp;
        mgard_x::ComposedLosslessCompressor<std::int32_t, std::uint64_t, mgard_x::HIP> clc((mgard_x::SIZE)n, cfg);
        try { clc.Compress(arr, comp, 0); mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0); std::printf("compress ok\n"); } catch (const std::exception& e) { std::printf("compress EXC: %s\n", e.what()); }
        auto& ws = clc.huffman.workspace;
        std::vector<unsigned> dh(dict), sf(dict), cl(dict); unsigned first_nz = 0;
        mgard_x::MemoryManager<mgard_x::HIP>::Copy1D(dh.data(), ws._d_freq_copy_array.data(), dict, 0);
        mgard_x::MemoryManager<mgard_x::HIP>::Copy1D(sf.data(), ws.freq_array.data(), dict, 0);
        mgard_x::MemoryManager<mgard_x::HIP>::Copy1D(cl.data(), ws.CL_array.data(), dict, 0);
        mgard_x::MemoryManager<mgard_x::HIP>::Copy1D(&first_nz, ws.first_nonzero_index_array.data(), 1, 0);
        // sorted qcodes live in the decodebook after 2*type_bw entries of H
        const size_t type_bw = 64; std::vector<uint8_t> db(ws.decodebook_array.shape(0));
        mgard_x::MemoryManager<mgard_x::HIP>::Copy1D(db.data(), ws.decodebook_array.data(), db.size(), 0);
        mgard_x::DeviceRuntime<mgard_x::HIP>::SyncQueue(0);
        const unsigned* qsorted = reinterpret_cast<const unsigned*>(db.data() + sizeof(uint64_t) * 2 * type_bw);
        size_t hist_mismatch = 0; for (int s = 0; s < dict; ++s) hist_mismatch += dh[s] != hist[s];
        unsigned long long sum_dev = 0, sum_host = 0; for (int s = 0; s < dict; ++s) { sum_dev += dh[s]; sum_host += hist[s]; }
        std::printf("device histogram: mismatching bins=%zu (sum dev %llu host %llu)\n", hist_mismatch, sum_dev, sum_host);
        bool sorted = true; for (int i = 1; i < dict; ++i) if (sf[i] < sf[i - 1]) { sorted = false; break; }
        size_t pair_mismatch = 0; for (int i = 0; i < dict; ++i) { const unsigned q = qsorted[i]; if (q >= (unsigned)dict || hist[q] != sf[i]) pair_mismatch++; }
        std::printf("sort-by-key: sorted=%s first_nonzero_index=%u (expected %d) pairing mismatches=%zu\n", sorted ? "yes" : "NO", first_nz, dict - nz, pair_mismatch);
        std::vector<unsigned> clnz(cl.begin(), cl.begin() + (dict - first_nz)); // CL for nz symbols (order reversed by GenerateCW)
        unsigned dev_max_cl = clnz.empty() ? 0 : *std::max_element(clnz.begin(), clnz.end());
        std::vector<unsigned> a(clnz), b; for (int s = 0; s < dict; ++s) if (hist[s]) b.push_back(hcl[s]); std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        std::printf("code lengths: device max=%u host max=%u multiset %s\n", dev_max_cl, host_max_cl, a == b ? "MATCH" : "DIFFER");
        std::printf("first 12 sorted (freq,qcode): "); for (int i = first_nz; i < std::min(dict, (int)first_nz + 12); ++i) std::printf("(%u,%u) ", sf[i], qsorted[i]); std::printf("\n");
        std::printf("first 12 device CL: "); for (int i = 0; i < std::min((int)clnz.size(), 12); ++i) std::printf("%u ", clnz[i]); std::printf("\n");
    } catch (const std::exception& e) { std::printf("EXC %s\n", e.what()); }
    return 0;
}
