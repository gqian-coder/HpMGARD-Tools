// End-to-end test of the ADIOS2 plugin operator.
//   1. write mesh.bp (connectivity + coordinates of a small structured hex mesh);
//   2. write out.bp: the same mesh (passthrough) plus a p3 pressure field compressed with the dedup base
//      (mesh read from mesh.bp at compress time) and a 3-component velocity field with cellavg + frbp|zstd;
//   3. read out.bp back with plain ADIOS2: the plugin decodes self-contained (mesh names travel in the
//      header, the mesh is read from the file being read) and the pointwise bound is verified.
//   export ADIOS2_PLUGIN_PATH=<CompressHpMGARDOperator install>/lib64
//   hpmgard_adios_roundtrip [out.bp]
#include <adios2.h>
#include <hpmgard/Quadrature.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string file = argc > 1 ? argv[1] : "hpmgard_roundtrip_test.bp";
    const std::string meshfile = file + ".mesh.bp";
    const int nx = 10, ny = 8, nz = 6, p = 3, n = p + 1, nsp = n * n * n;
    const size_t ncells = size_t(nx) * ny * nz, npts = size_t(nx + 1) * (ny + 1) * (nz + 1);
    std::vector<int64_t> conn(ncells * 8); std::vector<double> pts(npts * 3);
    for (int k = 0; k <= nz; ++k) for (int j = 0; j <= ny; ++j) for (int i = 0; i <= nx; ++i) { const size_t v = (size_t(k) * (ny + 1) + j) * (nx + 1) + i; pts[3 * v] = i; pts[3 * v + 1] = j; pts[3 * v + 2] = k; }
    auto vid = [&](int i, int j, int k) { return static_cast<int64_t>((size_t(k) * (ny + 1) + j) * (nx + 1) + i); };
    size_t c = 0;
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i, ++c) {
        const int64_t v[8] = {vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)};
        for (int q = 0; q < 8; ++q) conn[c * 8 + q] = v[q];
    }
    const std::vector<double> xi = hpmgard::GaussLegendreNodes(n);
    std::vector<double> pres(ncells * nsp), vel(ncells * nsp * 3);
    std::mt19937 rng(5); std::normal_distribution<double> N(0, 1);
    c = 0;
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i, ++c)
        for (int kk = 0; kk < n; ++kk) for (int jj = 0; jj < n; ++jj) for (int ii = 0; ii < n; ++ii) {
            const double x = i + 0.5 * (xi[ii] + 1), y = j + 0.5 * (xi[jj] + 1), z = k + 0.5 * (xi[kk] + 1);
            const size_t sp = (kk * n + jj) * n + ii;
            pres[c * nsp + sp] = 1e5 + 3000 * std::sin(0.5 * x) * std::cos(0.4 * y) + 20 * z + 2 * N(rng);
            for (int v = 0; v < 3; ++v) vel[(c * nsp + sp) * 3 + v] = 100 * (v + 1) * std::cos(0.3 * x + v) + 5 * N(rng);
        }
    const double tol = 1e-3;
    int failures = 0;
    try {
        {   // 1. mesh file
            adios2::ADIOS ad; adios2::IO io = ad.DeclareIO("m");
            auto vconn = io.DefineVariable<int64_t>("vtkDGHex/conn", {}, {}, {ncells, 8});
            auto vpts = io.DefineVariable<double>("coordinates/points", {}, {}, {npts, 3});
            adios2::Engine w = io.Open(meshfile, adios2::Mode::Write);
            w.BeginStep(); w.Put(vconn, conn.data()); w.Put(vpts, pts.data()); w.EndStep(); w.Close();
        }
        {   // 2. compressed file with the mesh passed through
            adios2::ADIOS ad; adios2::IO io = ad.DeclareIO("w");
            auto vconn = io.DefineVariable<int64_t>("vtkDGHex/conn", {}, {}, {ncells, 8});
            auto vpts = io.DefineVariable<double>("coordinates/points", {}, {}, {npts, 3});
            auto vp = io.DefineVariable<double>("vtkDGHex/Pressure", {}, {}, {ncells, static_cast<size_t>(nsp)});
            auto vv = io.DefineVariable<double>("vtkDGHex/Velocity", {}, {}, {ncells, static_cast<size_t>(nsp) * 3});
            adios2::Params common{{"PluginName", "CompressHpMGARDOperator"}, {"PluginLibrary", "CompressHpMGARDOperator"},
                                  {"cell_type", "hexahedron"}, {"order", std::to_string(p)}, {"tolerance", std::to_string(tol)}, {"mode", "REL_L2"},
                                  {"blockid", "0"}, {"threshold", "1000"}, {"tuning", "fixed"}};
            adios2::Params pparams = common;
            pparams["meshfile"] = meshfile; pparams["connectivity_variable"] = "vtkDGHex/conn"; pparams["coordinates_variable"] = "coordinates/points";
            pparams["base_method"] = "dedup_sfc"; pparams["base_codec"] = "huffman|zstd:3"; pparams["residual_codec"] = "huffman|zstd:3";
            vp.AddOperation("plugin", pparams);
            adios2::Params vparams = common; vparams["base_method"] = "cellavg"; vparams["residual_codec"] = "frbp|zstd:3"; vparams["base_codec"] = "frbp|zstd:3";
            vv.AddOperation("plugin", vparams);
            adios2::Engine w = io.Open(file, adios2::Mode::Write);
            w.BeginStep(); w.Put(vconn, conn.data()); w.Put(vpts, pts.data()); w.Put(vp, pres.data()); w.Put(vv, vel.data()); w.EndStep(); w.Close();
        }
        {   // 3. read back (no parameters, no environment except the plugin path)
            adios2::ADIOS ad; adios2::IO io = ad.DeclareIO("r");
            adios2::Engine r = io.Open(file, adios2::Mode::ReadRandomAccess);
            auto vp = io.InquireVariable<double>("vtkDGHex/Pressure"); auto vv = io.InquireVariable<double>("vtkDGHex/Velocity");
            if (!vp || !vv) throw std::runtime_error("variables missing in the compressed file");
            std::vector<double> rp, rv;
            vp.SetBlockSelection(0); r.Get(vp, rp, adios2::Mode::Sync);
            vv.SetBlockSelection(0); r.Get(vv, rv, adios2::Mode::Sync);
            r.Close();
            auto check = [&](const char* name, const std::vector<double>& a, const std::vector<double>& b, int ncomp) {
                if (a.size() != b.size()) { std::printf("  %s: size mismatch %zu vs %zu\n", name, a.size(), b.size()); ++failures; return; }
                for (int v = 0; v < ncomp; ++v) {
                    double ss = 0, me = 0; size_t cnt = 0;
                    for (size_t i = v; i < a.size(); i += ncomp) { ss += a[i] * a[i]; ++cnt; me = std::max(me, std::fabs(a[i] - b[i])); }
                    const double bound = tol * std::sqrt(ss / cnt);
                    std::printf("  %s comp %d: max err %.4g, bound %.4g -> %s\n", name, v, me, bound, me <= bound * (1 + 1e-9) ? "OK" : "VIOLATED");
                    if (me > bound * (1 + 1e-9)) ++failures;
                }
            };
            check("Pressure", pres, rp, 1); check("Velocity", vel, rv, 3);
        }
    } catch (const std::exception& e) { std::printf("  exception: %s\n", e.what()); ++failures; }
    std::printf("hpmgard_adios_roundtrip: %s (%s)\n", failures ? "FAILED" : "PASSED", file.c_str());
    return failures ? 1 : 0;
}
