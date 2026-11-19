#include "check.h"
#include <hpmgard/hpmgard.h>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
using namespace hpmgard;

// Structured nx*ny*nz hex grid with Gauss p nodes per cell; smooth field + small noise; synthetic corners.
struct Synthetic {
    size_t ncells; int p, nsp, ncomp;
    std::vector<double> data;   // interleaved
    MeshCorners mesh;
};

static Synthetic Make(int nx, int ny, int nz, int p, int ncomp, double noise, std::mt19937& rng) {
    Synthetic s; s.p = p; s.ncomp = ncomp; s.nsp = (p + 1) * (p + 1) * (p + 1); s.ncells = size_t(nx) * ny * nz;
    const std::vector<double> xi = GaussLegendreNodes(p + 1);
    const int n = p + 1;
    s.data.resize(s.ncells * s.nsp * ncomp);
    s.mesh.ncorner = 8; s.mesh.vertex_id.resize(s.ncells * 8); s.mesh.xyz.resize(3 * s.ncells * 8);
    std::normal_distribution<double> N(0.0, 1.0);
    size_t c = 0;
    for (int cz = 0; cz < nz; ++cz) for (int cy = 0; cy < ny; ++cy) for (int cx = 0; cx < nx; ++cx, ++c) {
        for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
            const double x = cx + 0.5 * (xi[i] + 1), y = cy + 0.5 * (xi[j] + 1), z = cz + 0.5 * (xi[k] + 1);
            const size_t sp = (k * n + j) * n + i;
            for (int v = 0; v < ncomp; ++v) {
                const double f = 1000.0 * (v + 1) + 300.0 * std::sin(0.4 * x + 0.1 * v) * std::cos(0.3 * y) + 50.0 * z * z / (nz * nz) + noise * N(rng);
                s.data[(c * s.nsp + sp) * ncomp + v] = f;
            }
        }
        // corners in tensor order: (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),...
        for (int k = 0; k < 2; ++k) for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
            const size_t slot = c * 8 + (k * 2 + j) * 2 + i;
            const int gx = cx + i, gy = cy + j, gz = cz + k;
            s.mesh.vertex_id[slot] = (static_cast<int64_t>(gz) * (ny + 1) + gy) * (nx + 1) + gx;
            s.mesh.xyz[3 * slot] = gx; s.mesh.xyz[3 * slot + 1] = gy; s.mesh.xyz[3 * slot + 2] = gz;
        }
    }
    return s;
}

static double MaxErr(const std::vector<double>& a, const std::vector<double>& b) { double e = 0; for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i])); return e; }

int main() {
    std::mt19937 rng(3);
    Synthetic s = Make(12, 10, 8, 4, 1, 0.5, rng);
    FieldView f{s.data.data(), s.ncells, s.nsp, 1};
    BlockMeta meta; meta.blockid = 7; meta.dims = {s.ncells, static_cast<uint64_t>(s.nsp)};
    const FieldStats fs = ComputeStats(f);
    const double rms = ScaleFor(fs, s.ncells * s.nsp, Norm::RelL2)[0];
    std::printf("synthetic: %zu cells, p4, rms %.3f range %.3f\n", s.ncells, rms, fs.hi[0] - fs.lo[0]);

    struct Case { const char* name; CompressConfig cfg; bool mesh; };
    std::vector<Case> cases;
    // HPMGARD_TEST_DEVICE=hip compresses on the GPU and decodes on the CPU (cross-backend check)
    const char* dev_env = std::getenv("HPMGARD_TEST_DEVICE"); const std::string dev = dev_env ? dev_env : "cpu";
    std::printf("compress device: %s, decompress device: cpu\n", dev.c_str());
    auto base = [&]() { CompressConfig c; c.shape = ElemShape::HEX; c.p = 4; c.ncomp = 1; c.norm = Norm::RelL2; c.tolerance = 1e-3; c.threads = 4; c.device = dev; return c; };
    { Case k{"cellavg huffman|zstd", base(), false}; cases.push_back(k); }
    { Case k{"dedup frbp|zstd", base(), true}; k.cfg.base_method = BaseMethod::DedupSfc; k.cfg.base_codec = "frbp|zstd:3"; k.cfg.residual_codec = "frbp|zstd:3"; cases.push_back(k); }
    { Case k{"cell mean (p_base 0) frbp", base(), false}; k.cfg.base_order = 0; k.cfg.base_method = BaseMethod::Plain; k.cfg.base_codec = "frbp"; k.cfg.residual_codec = "frbp"; cases.push_back(k); }
    { Case k{"no split (p_base 4) raw|zstd", base(), false}; k.cfg.base_order = 4; k.cfg.base_method = BaseMethod::Plain; k.cfg.base_codec = "raw|zstd:3"; k.cfg.residual_codec = "raw|zstd:3"; cases.push_back(k); }
    { Case k{"p_base 2 closed, gauss-lobatto data", base(), false}; k.cfg.base_order = 2; k.cfg.node_set = NodeSet::GaussLobatto; cases.push_back(k); }
    { Case k{"abs tolerance", base(), false}; k.cfg.norm = Norm::Abs; k.cfg.tolerance = 0.7; cases.push_back(k); }
    { Case k{"rel range, base quantum 2", base(), true}; k.cfg.norm = Norm::RelRange; k.cfg.tolerance = 1e-4; k.cfg.base_quantum_ratio = 2.0; k.cfg.base_method = BaseMethod::DedupSfc; cases.push_back(k); }
    if (mgardx::Available()) {
        // MGARD-X lossless stages on the residual (>= 65536 symbols so the stage is really used; the GPU
        // path codes the device-resident symbols when the compress device is hip)
        for (const char* c : {"xhuff|zstd:3", "xhuff:3", "xrle", "xrans"}) { Case k{c, base(), false}; k.cfg.residual_codec = c; cases.push_back(k); }
        { Case k{"mgardx base codec", base(), false}; k.cfg.base_codec = "mgardx"; cases.push_back(k); }
        { Case k{"direct mgardx", base(), false}; k.cfg.direct_mgardx = true; cases.push_back(k); }
    }
    { Case k{"tuned", base(), true}; k.cfg.tune = true; k.cfg.tune_sample = 0.2; cases.push_back(k); }

    for (Case& k : cases) {
        CompressResult r = Compress(f, k.cfg, k.mesh ? &s.mesh : nullptr, meta);
        CHECK_THROWS(Compress(f, k.cfg, k.mesh ? &s.mesh : nullptr, BlockMeta{0, {1, 2}, 0, "", ""}));
        std::vector<double> out; BlockHeader h;
        Decompress(r.bytes.data(), r.bytes.size(), k.mesh ? &s.mesh : nullptr, out, &h, "cpu");
        CHECK(out.size() == s.data.size());
        const double tol_abs = h.tol_abs[0];
        const double err = MaxErr(s.data, out);
        CHECK(err <= tol_abs * (1 + 1e-9));
        CHECK(h.blockid == 7 && h.p == 4 && h.nsp == 125 && h.ncomp == 1);
        std::printf("  %-38s CR %7.2f  max err %.3e / tol %.3e  (%s)\n", k.name, r.stats.ratio(), err, tol_abs, r.stats.used.direct_mgardx ? "direct" : (r.stats.used.base_method == BaseMethod::DedupSfc ? "dedup" : "cellavg/plain"));
        if (k.cfg.tune) std::printf("%s", r.stats.tune_report.c_str());
    }
    // 3 interleaved components, float-typed caller, verbose header description
    {
        Synthetic s3 = Make(8, 8, 6, 3, 3, 0.2, rng);
        FieldView f3{s3.data.data(), s3.ncells, s3.nsp, 3};
        CompressConfig c = base(); c.p = 3; c.ncomp = 3; c.base_method = BaseMethod::DedupSfc;
        BlockMeta m3; m3.data_type = 1; m3.dims = {s3.ncells, static_cast<uint64_t>(s3.nsp) * 3}; m3.conn_var = "vtkDGHex/conn"; m3.coords_var = "coordinates/points";
        CompressResult r = Compress(f3, c, &s3.mesh, m3);
        std::vector<double> out; BlockHeader h;
        Decompress(r.bytes.data(), r.bytes.size(), &s3.mesh, out, &h);
        CHECK(h.ncomp == 3 && h.data_type == 1 && h.conn_var == "vtkDGHex/conn");
        for (int v = 0; v < 3; ++v) { double e = 0; for (size_t i = v; i < out.size(); i += 3) e = std::max(e, std::fabs(out[i] - s3.data[i])); CHECK(e <= h.tol_abs[v] * (1 + 1e-9)); }
        std::printf("%s", DescribeHeader(h).c_str());
        // wrong mesh -> rejected
        MeshCorners bad = s3.mesh; bad.vertex_id[5] += 1000000;
        std::vector<double> o2; CHECK_THROWS(Decompress(r.bytes.data(), r.bytes.size(), &bad, o2));
        // missing mesh -> rejected
        CHECK_THROWS(Decompress(r.bytes.data(), r.bytes.size(), nullptr, o2));
        // corrupted header -> rejected
        std::vector<uint8_t> bad_bytes = r.bytes; bad_bytes[40] ^= 0x5A;
        CHECK_THROWS(Decompress(bad_bytes.data(), bad_bytes.size(), &s3.mesh, o2));
    }
    // constant component and zero tolerance
    {
        std::vector<double> cst(500 * 27, 3.25);
        FieldView fc{cst.data(), 500, 27, 1};
        CompressConfig c = base(); c.p = 2;
        BlockMeta mc; mc.dims = {500, 27};
        CompressResult r = Compress(fc, c, nullptr, mc);
        std::vector<double> out; Decompress(r.bytes.data(), r.bytes.size(), nullptr, out);
        CHECK(out == cst); CHECK(r.bytes.size() < 400);
        Synthetic s2 = Make(4, 4, 4, 2, 1, 0.1, rng);
        FieldView f2{s2.data.data(), s2.ncells, s2.nsp, 1};
        CompressConfig z = base(); z.p = 2; z.norm = Norm::Abs; z.tolerance = 0.0;
        BlockMeta m2; m2.dims = {s2.ncells, static_cast<uint64_t>(s2.nsp)};
        r = Compress(f2, z, nullptr, m2);
        Decompress(r.bytes.data(), r.bytes.size(), nullptr, out);
        CHECK(out == s2.data);
    }
    return Report("test_compressor");
}
