// mgard_test: MGARD-X (SERIAL) compress/decompress a raw double array in a given shape.
// usage: mgard_test2 <raw.bin> <shape> <eb_rel_range> <s|inf> [levels|default] [label] [decoded_out.bin] [abs_tol override]
// Prints one CSV line: label,shape,eb,s,levels,n,tol_abs,range,comp_bytes,CR,max_err,rmse,rmse_over_range,t_comp_s,t_decomp_s
#include <mgard/compress_x.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 5) { std::cerr << "usage: mgard_test <raw.bin> <d0,d1,..> <eb_rel_range> <s|inf> [levels|default] [label]\n"; return 1; }
    const std::string path = argv[1];
    std::vector<mgard_x::SIZE> shape;
    { std::stringstream ss(argv[2]); std::string tok; while (std::getline(ss, tok, ',')) shape.push_back(std::stoul(tok)); }
    const double eb = std::atof(argv[3]);
    const double s = (std::string(argv[4]) == "inf") ? std::numeric_limits<double>::infinity() : std::atof(argv[4]);
    const std::string levels = argc > 5 ? argv[5] : "default";
    const std::string label = argc > 6 ? argv[6] : "run";

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "cannot open " << path << "\n"; return 2; }
    const size_t bytes = f.tellg(); f.seekg(0);
    const size_t n = bytes / sizeof(double);
    size_t prod = 1; for (auto d : shape) prod *= d;
    if (prod != n) { std::cerr << "shape product " << prod << " != n " << n << "\n"; return 3; }
    std::vector<double> data(n);
    f.read(reinterpret_cast<char *>(data.data()), bytes);

    double mn = data[0], mx = data[0];
    for (double v : data) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const double range = mx - mn;
    const double tol = (argc > 8) ? std::atof(argv[8]) : eb * range;  // optional absolute tolerance override

    mgard_x::Config cfg;
    cfg.dev_type = mgard_x::device_type::SERIAL;
    cfg.lossless = mgard_x::lossless_type::Huffman_Zstd;
    cfg.log_level = 0;
    if (levels != "default") cfg.num_global_refactoring_level = std::stoul(levels);

    void *comp = nullptr; size_t comp_size = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto st = mgard_x::compress(shape.size(), mgard_x::data_type::Double, shape, tol, s,
                                mgard_x::error_bound_type::ABS, data.data(), comp, comp_size, cfg, false);
    auto t1 = std::chrono::steady_clock::now();
    if (st != mgard_x::compress_status_type::Success) { std::cerr << "compress failed, status " << (int)st << "\n"; return 4; }

    void *dec = nullptr;
    auto t2 = std::chrono::steady_clock::now();
    st = mgard_x::decompress(comp, comp_size, dec, cfg, false);
    auto t3 = std::chrono::steady_clock::now();
    if (st != mgard_x::compress_status_type::Success) { std::cerr << "decompress failed, status " << (int)st << "\n"; return 5; }

    const double *d = static_cast<const double *>(dec);
    if (argc > 7 && std::string(argv[7]) != "-") { std::ofstream o(argv[7], std::ios::binary); o.write(reinterpret_cast<const char *>(d), n * sizeof(double)); }
    double maxerr = 0, sq = 0;
    for (size_t i = 0; i < n; ++i) { double e = std::fabs(d[i] - data[i]); if (e > maxerr) maxerr = e; sq += e * e; }
    const double rmse = std::sqrt(sq / n);
    const double tc = std::chrono::duration<double>(t1 - t0).count();
    const double td = std::chrono::duration<double>(t3 - t2).count();
    std::printf("%s,%s,%g,%s,%s,%zu,%.6g,%.6g,%zu,%.2f,%.6g,%.6g,%.3e,%.1f,%.1f\n", label.c_str(), argv[2], eb,
                argv[4], levels.c_str(), n, tol, range, comp_size, double(bytes) / comp_size, maxerr, rmse,
                rmse / range, tc, td);
    return 0;
}
