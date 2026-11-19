// hpmgard_bench: compress a raw dump of one DG field with hpMGARD (fixed or tuned configuration) and,
// optionally, with MGARD-X 1-D at the same absolute bound; verify the error; print one CSV line.
//
//   hpmgard_bench --raw field.bin --ncells N --p 4 [--shape hexahedron] [--ncomp 1]
//                 --tol 1e-3 [--norm rel_l2|rel_range|abs]
//                 [--tune | --base-order 1 --base-method cellavg|dedup|plain --base-quantum 0.5
//                  --base-codec "huffman|zstd:3" --residual-codec "huffman|zstd:3" --node-set gauss|lobatto --warmup]
//                 [--conn conn.bin --points points.bin --nodes-per-cell 27]   (int64 / double raw dumps; enables dedup)
//                 [--mgardx-direct]  [--threads 16] [--label name] [--csv out.csv] [--verbose]
#include <hpmgard/hpmgard.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace hpmgard;
using Clock = std::chrono::steady_clock;
static double Sec(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

template <class T> static std::vector<T> ReadRaw(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t n = static_cast<size_t>(f.tellg()) / sizeof(T); f.seekg(0);
    std::vector<T> v(n); f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    return v;
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) { std::cerr << "unexpected argument " << k << "\n"; return 2; }
        k = k.substr(2);
        if (k == "tune" || k == "mgardx-direct" || k == "verbose") a[k] = "1";
        else { if (i + 1 >= argc) { std::cerr << "missing value for --" << k << "\n"; return 2; } a[k] = argv[++i]; }
    }
    auto get = [&](const std::string& k, const std::string& d) { auto it = a.find(k); return it == a.end() ? d : it->second; };
    if (!a.count("raw") || !a.count("ncells") || !a.count("p") || !a.count("tol")) { std::cerr << "usage: see header of hpmgard_bench.cpp\n"; return 2; }

    CompressConfig cfg;
    cfg.shape = ParseShape(get("shape", "hexahedron"));
    cfg.p = std::stoi(a["p"]); cfg.ncomp = std::stoi(get("ncomp", "1"));
    cfg.tolerance = std::stod(a["tol"]);
    const std::string norm = get("norm", "rel_l2");
    cfg.norm = norm == "abs" ? Norm::Abs : norm == "rel_range" ? Norm::RelRange : Norm::RelL2;
    cfg.node_set = get("node-set", "gauss") == "lobatto" ? NodeSet::GaussLobatto : NodeSet::GaussLegendre;
    cfg.base_order = std::stoi(get("base-order", "1"));
    const std::string bm = get("base-method", "cellavg");
    cfg.base_method = bm == "dedup" ? BaseMethod::DedupSfc : bm == "plain" ? BaseMethod::Plain : BaseMethod::CellAvg;
    cfg.base_quantum_ratio = std::stod(get("base-quantum", "0.5"));
    cfg.base_codec = get("base-codec", "huffman|zstd:3"); cfg.residual_codec = get("residual-codec", "huffman|zstd:3");
    cfg.tune = a.count("tune") > 0; cfg.tune_sample = std::stod(get("tune-sample", "0.03")); cfg.tune_codecs = get("tune-codecs", "huffman|zstd:3,raw|zstd:3,frbp|zstd:3");
    cfg.device = get("device", "auto"); cfg.threads = std::stoi(get("threads", "0")); cfg.verbose = a.count("verbose") > 0;
    cfg.allow_mgardx_direct = get("allow-mgardx-direct", "1") == "1";

    const std::vector<double> data = ReadRaw<double>(a["raw"]);
    const size_t ncells = std::stoull(a["ncells"]);
    const int nsp = NumSP(cfg.shape, cfg.p);
    if (data.size() != ncells * static_cast<size_t>(nsp) * cfg.ncomp) { std::cerr << "raw size " << data.size() << " != ncells*nsp*ncomp = " << ncells * nsp * cfg.ncomp << "\n"; return 3; }
    FieldView field{data.data(), ncells, nsp, cfg.ncomp};

    MeshCorners mesh; const MeshCorners* mesh_ptr = nullptr;
    if (a.count("conn") && a.count("points")) {
        const std::vector<int64_t> conn = ReadRaw<int64_t>(a["conn"]);
        const std::vector<double> pts = ReadRaw<double>(a["points"]);
        const int npc = std::stoi(get("nodes-per-cell", "27"));
        if (conn.size() != ncells * npc) { std::cerr << "conn size mismatch\n"; return 3; }
        const std::vector<int> perm = CornerPermutation(cfg.shape);
        const int nc = NumCorners(cfg.shape);
        mesh.ncorner = nc; mesh.vertex_id.resize(ncells * nc); mesh.xyz.resize(3 * ncells * nc);
        for (size_t c = 0; c < ncells; ++c) for (int k = 0; k < nc; ++k) {
            const int64_t v = conn[c * npc + perm[k]];
            mesh.vertex_id[c * nc + k] = v;
            for (int d = 0; d < 3; ++d) mesh.xyz[3 * (c * nc + k) + d] = pts[3 * v + d];
        }
        mesh_ptr = &mesh;
    }

    BlockMeta meta; meta.dims = {ncells, static_cast<uint64_t>(nsp) * cfg.ncomp};
    if (a.count("warmup")) { CompressResult w = Compress(field, cfg, mesh_ptr, meta); (void)w; }   // untimed run: kernel loading, workspaces
    const auto t0 = Clock::now();
    CompressResult r = Compress(field, cfg, mesh_ptr, meta);
    const double t_comp = Sec(t0, Clock::now());
    const auto t1 = Clock::now();
    std::vector<double> out; BlockHeader h;
    Decompress(r.bytes.data(), r.bytes.size(), mesh_ptr, out, &h, get("decode-device", cfg.device));
    const double t_dec = Sec(t1, Clock::now());
    double maxerr = 0, sq = 0; double maxrel = 0;
    for (int v = 0; v < cfg.ncomp; ++v) {
        double me = 0, s2 = 0; for (size_t i = v; i < out.size(); i += cfg.ncomp) { const double e = std::fabs(out[i] - data[i]); me = std::max(me, e); s2 += e * e; }
        maxerr = std::max(maxerr, me); sq += s2; maxrel = std::max(maxrel, h.tol_abs[v] > 0 ? me / h.tol_abs[v] : 0.0);
    }
    const double rmse = std::sqrt(sq / out.size());
    if (cfg.verbose) std::cerr << r.stats.tune_report << DescribeHeader(h);

    // MGARD-X direct reference at the same absolute bound (component 0 scale)
    double cr_mx_inf = 0, cr_mx_0 = 0, t_mx_inf = 0, t_mx_0 = 0, err_mx_inf = 0, err_mx_0 = 0;
    if (a.count("mgardx-direct") && mgardx::Available()) {
        std::vector<double> comp(ncells * nsp);
        size_t total_inf = 0, total_0 = 0;
        for (int v = 0; v < cfg.ncomp; ++v) {
            for (size_t i = 0; i < comp.size(); ++i) comp[i] = data[i * cfg.ncomp + v];
            const double tol = h.tol_abs[v];
            for (int mode = 0; mode < 2; ++mode) {
                const double s = mode == 0 ? std::numeric_limits<double>::infinity() : 0.0;
                const int mdev = get("mgardx-device", get("device", "auto")) == "hip" ? 1 : 0;
                try {
                    const auto ta = Clock::now();
                    std::vector<uint8_t> b = mgardx::Compress1D(comp.data(), comp.size(), tol, s, 0, mdev);
                    const double tc = Sec(ta, Clock::now());
                    std::vector<double> d; mgardx::Decompress(b.data(), b.size(), d, mdev);
                    double me = 0; for (size_t i = 0; i < comp.size(); ++i) me = std::max(me, std::fabs(d[i] - comp[i]));
                    if (mode == 0) { total_inf += b.size(); t_mx_inf += tc; err_mx_inf = std::max(err_mx_inf, me / tol); }
                    else { total_0 += b.size(); t_mx_0 += tc; err_mx_0 = std::max(err_mx_0, me / tol); }
                } catch (const std::exception& e) { std::cerr << "MGARD-X reference failed (device " << mdev << ", s=" << s << "): " << e.what() << "\n"; }
            }
        }
        cr_mx_inf = total_inf ? double(r.stats.raw_bytes) / total_inf : 0; cr_mx_0 = total_0 ? double(r.stats.raw_bytes) / total_0 : 0;
    }
    const std::string label = get("label", a["raw"]);
    const bool header = !a.count("csv") || !std::ifstream(a["csv"]).good();
    FILE* out_f = a.count("csv") ? std::fopen(a["csv"].c_str(), "a") : stdout;
    if (header) std::fprintf(out_f, "label,ncells,p,ncomp,norm,tol,tol_abs0,raw_bytes,hpm_bytes,hpm_CR,hpm_maxerr_over_tol,hpm_rmse,hpm_t_comp_s,hpm_t_dec_s,hpm_t_tune_s,base_order,base_method,base_quantum,base_codec,residual_codec,direct,mgardx_CR_sinf,mgardx_maxerr_over_tol_sinf,mgardx_t_sinf,mgardx_CR_s0,mgardx_maxerr_over_tol_s0,mgardx_t_s0,device,t_stats,t_refactor,t_base,t_residual,t_code\n");
    const CompressConfig& u = r.stats.used;
    std::fprintf(out_f, "%s,%zu,%d,%d,%s,%g,%g,%llu,%llu,%.3f,%.4f,%.6g,%.3f,%.3f,%.3f,%d,%s,%g,%s,%s,%d,%.3f,%.4f,%.3f,%.3f,%.4f,%.3f,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                 label.c_str(), ncells, cfg.p, cfg.ncomp, norm.c_str(), cfg.tolerance, h.tol_abs[0],
                 (unsigned long long)r.stats.raw_bytes, (unsigned long long)r.stats.compressed_bytes, r.stats.ratio(), maxrel, rmse, t_comp, t_dec, r.stats.t_tune,
                 u.base_order, u.base_method == BaseMethod::DedupSfc ? "dedup" : u.base_method == BaseMethod::CellAvg ? "cellavg" : "plain", u.base_quantum_ratio,
                 u.base_codec.c_str(), u.residual_codec.c_str(), u.direct_mgardx ? 1 : 0, cr_mx_inf, err_mx_inf, t_mx_inf, cr_mx_0, err_mx_0, t_mx_0,
                 h.writer.c_str(), r.stats.t_stats, r.stats.t_refactor, r.stats.t_base, r.stats.t_residual, r.stats.t_code);
    if (out_f != stdout) std::fclose(out_f);
    if (maxrel > 1.0 + 1e-9) { std::cerr << "ERROR: bound violated (max err / tol = " << maxrel << ")\n"; return 4; }
    return 0;
}
