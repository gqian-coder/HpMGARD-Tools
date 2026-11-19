#include "check.h"
#include <hpmgard/codec/Codec.h>
#include <random>
#include <string>
#include <vector>
using namespace hpmgard::codec;

static std::vector<int32_t> MakeStream(size_t n, int kind, std::mt19937& rng) {
    std::vector<int32_t> v(n);
    std::normal_distribution<double> N(0.0, 3.0);
    std::uniform_int_distribution<int32_t> big(-2000000000, 2000000000);
    std::uniform_real_distribution<double> U(0, 1);
    for (size_t i = 0; i < n; ++i) {
        switch (kind) {
        case 0: v[i] = 0; break;                                                        // all zero
        case 1: v[i] = (U(rng) < 0.9) ? 0 : static_cast<int32_t>(std::lround(N(rng))); break;   // sparse small
        case 2: v[i] = static_cast<int32_t>(std::lround(N(rng) * 1000)); break;               // wide
        case 3: v[i] = (U(rng) < 0.999) ? static_cast<int32_t>(std::lround(N(rng))) : big(rng); break;   // outliers
        default: v[i] = static_cast<int32_t>(i % 7) - 3; break;
        }
    }
    return v;
}

int main() {
    std::mt19937 rng(11);
    const std::vector<std::string> chains{"raw", "frbp", "huffman", "huffman:12", "raw|zstd:3", "frbp|zstd:3", "huffman|zstd:19", "huffman:8|zstd:1"};
    for (const std::string& spec : chains) {
        const std::vector<Stage> chain = ParseChain(spec);
        CHECK(!ChainString(chain).empty());
        for (size_t n : {size_t(0), size_t(1), size_t(31), size_t(32), size_t(33), size_t(10007), size_t(200000)}) {
            for (int kind = 0; kind < 5; ++kind) {
                const std::vector<int32_t> v = MakeStream(n, kind, rng);
                const std::vector<uint8_t> b = EncodeSymbols(v.data(), n, chain, 0.5);
                CHECK(PeekSymbolCount(b.data(), b.size()) == n);
                std::vector<int32_t> back;
                const size_t m = DecodeSymbols(b.data(), b.size(), back);
                CHECK(m == n);
                CHECK(back == v);
            }
        }
    }
    // extreme values (int32 limits) through every first stage
    {
        std::vector<int32_t> v{INT32_MIN, INT32_MAX, -1, 0, 1, INT32_MIN + 1, 123456789, -987654321};
        for (const std::string& spec : {"raw", "frbp", "huffman"}) {
            std::vector<uint8_t> b = EncodeSymbols(v.data(), v.size(), ParseChain(spec));
            std::vector<int32_t> back; DecodeSymbols(b.data(), b.size(), back);
            CHECK(back == v);
        }
    }
    // FRBP size formula: header 40 + rates (padded) + per nonzero block 4 + 4*rate
    {
        std::vector<int32_t> v(64, 0); v[40] = 5;   // block 0 all zero (rate 0), block 1 max 5 -> rate 3
        std::vector<uint8_t> b = FrbpEncode(v.data(), v.size(), 1.0);
        CHECK(b.size() == 40 + 4 + (4 + 4 * 3));
    }
    // Huffman: a stream with one distinct symbol and a stream that forces the escape path
    {
        std::vector<int32_t> one(1000, 42);
        std::vector<uint8_t> b = HuffmanEncode(one.data(), one.size(), 16);
        std::vector<int32_t> back(one.size()); HuffmanDecode(b.data(), b.size(), back.data(), back.size());
        CHECK(back == one);
        CHECK(b.size() < 200);   // ~1 bit per symbol
        std::vector<int32_t> esc(500); for (size_t i = 0; i < esc.size(); ++i) esc[i] = static_cast<int32_t>(i) * 100000 - 25000000;
        b = HuffmanEncode(esc.data(), esc.size(), 8);
        back.assign(esc.size(), 0); HuffmanDecode(b.data(), b.size(), back.data(), back.size());
        CHECK(back == esc);
    }
    // corrupt container is rejected
    {
        std::vector<int32_t> v = MakeStream(5000, 1, rng);
        std::vector<uint8_t> b = EncodeSymbols(v.data(), v.size(), ParseChain("huffman|zstd:3"));
        b[0] ^= 0xFF;
        std::vector<int32_t> back;
        CHECK_THROWS(DecodeSymbols(b.data(), b.size(), back));
    }
    CHECK_THROWS(ParseChain("zstd:3|huffman"));
    CHECK_THROWS(ParseChain("bogus"));
    return Report("test_codecs");
}
