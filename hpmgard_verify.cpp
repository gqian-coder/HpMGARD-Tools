// hpmgard_verify: compare the double/float variables of two BP files block by block (the second may be
// a compressed file; ADIOS2 decodes it through the plugin). Reports max error, RMSE, RMS and range of the
// reference per variable and step, and the error relative to RMS and range.
//   hpmgard_verify reference.bp test.bp [--vars a,b] [--steps N] [--csv out.csv]
#include <adios2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

template <class T>
static void Compare(adios2::IO& ia, adios2::Engine& ra, adios2::IO& ib, adios2::Engine& rb, const std::string& name, size_t step, FILE* csv) {
    auto va = ia.InquireVariable<T>(name), vb = ib.InquireVariable<T>(name);
    if (!va || !vb) { std::cerr << "  " << name << ": missing in one file\n"; return; }
    const auto blocks = ra.BlocksInfo(va, step);
    const bool global = !va.Shape().empty();
    double maxerr = 0, sq = 0, ss = 0; double lo = 1e300, hi = -1e300; size_t n = 0;
    std::vector<T> a, b;
    for (size_t k = 0; k < blocks.size(); ++k) {
        size_t cnt = 1; for (auto c : blocks[k].Count) cnt *= c;
        a.resize(cnt); b.resize(cnt);
        va.SetStepSelection({step, 1}); vb.SetStepSelection({step, 1});
        if (global) { va.SetSelection({blocks[k].Start, blocks[k].Count}); vb.SetSelection({blocks[k].Start, blocks[k].Count}); }
        else { va.SetBlockSelection(k); vb.SetBlockSelection(k); }
        ra.Get(va, a.data(), adios2::Mode::Sync); rb.Get(vb, b.data(), adios2::Mode::Sync);
        for (size_t i = 0; i < cnt; ++i) {
            const double x = a[i], e = std::fabs(double(a[i]) - double(b[i]));
            maxerr = std::max(maxerr, e); sq += e * e; ss += x * x; lo = std::min(lo, x); hi = std::max(hi, x); ++n;
        }
    }
    const double rmse = n ? std::sqrt(sq / n) : 0, rms = n ? std::sqrt(ss / n) : 0, range = hi - lo;
    std::printf("  step %zu %-32s blocks %3zu  maxerr %.4e  rmse %.4e  maxerr/rms %.3e  rmse/rms %.3e  maxerr/range %.3e\n", step, name.c_str(), blocks.size(), maxerr, rmse,
                rms > 0 ? maxerr / rms : 0, rms > 0 ? rmse / rms : 0, range > 0 ? maxerr / range : 0);
    if (csv) std::fprintf(csv, "%zu,%s,%zu,%zu,%.6e,%.6e,%.6e,%.6e\n", step, name.c_str(), blocks.size(), n, maxerr, rmse, rms, range);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: hpmgard_verify reference.bp test.bp [--vars a,b] [--steps N] [--csv out.csv]\n"; return 2; }
    std::set<std::string> only; size_t max_steps = 1000000000; FILE* csv = nullptr;
    for (int i = 3; i + 1 < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--vars") { std::string cur; for (char c : std::string(argv[++i])) { if (c == ',') { only.insert(cur); cur.clear(); } else cur += c; } if (!cur.empty()) only.insert(cur); }
        else if (k == "--steps") max_steps = std::stoul(argv[++i]);
        else if (k == "--csv") { csv = std::fopen(argv[++i], "w"); std::fprintf(csv, "step,variable,blocks,count,maxerr,rmse,rms,range\n"); }
    }
    adios2::ADIOS ad;
    adios2::IO ia = ad.DeclareIO("a"), ib = ad.DeclareIO("b");
    adios2::Engine ra = ia.Open(argv[1], adios2::Mode::ReadRandomAccess), rb = ib.Open(argv[2], adios2::Mode::ReadRandomAccess);
    const size_t nsteps = std::min({max_steps, ra.Steps(), rb.Steps()});
    for (size_t step = 0; step < nsteps; ++step)
        for (const auto& kv : ia.AvailableVariables()) {
            if (!only.empty() && !only.count(kv.first)) continue;
            const std::string type = ia.VariableType(kv.first);
            if (type == "double") Compare<double>(ia, ra, ib, rb, kv.first, step, csv);
            else if (type == "float") Compare<float>(ia, ra, ib, rb, kv.first, step, csv);
        }
    if (csv) std::fclose(csv);
    ra.Close(); rb.Close();
    return 0;
}
