// hpmgard_adios_compress: compress the DG cell attributes of a Fides cell_grid BP file with the
// CompressHpMGARDOperator plugin and pass everything else (mesh, attributes, time) through unchanged.
//
//   export ADIOS2_PLUGIN_PATH=<CompressHpMGARDOperator install>/lib64
//   hpmgard_adios_compress in.bp out.bp --tol 1e-3 [--mode REL_L2|REL_RANGE|ABS] [--vars a,b,c]
//        [--tuning auto|fixed] [--base-method cellavg|dedup_sfc|plain] [--base-order 1] [--base-quantum 0.5]
//        [--base-codec spec] [--residual-codec spec] [--tune-codecs list] [--threads N] [--steps N] [--no-mesh] [--verbose]
//
// Cell type and order come from the file's own attributes (`<name>/<celltype>/order`); the mesh for the
// dedup base is the input file itself (`shape/<celltype>/connectivity` and `shape/<celltype>/values`).
#include "adios_copy.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <type_traits>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::vector<std::string> Split(const std::string& s, char sep) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c == sep) { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

template <class T>
static void AccumulateStats(adios2::Variable<T> v, adios2::Engine& reader, size_t step, int ncomp,
                            std::vector<double>& ss, std::vector<double>& lo, std::vector<double>& hi, std::vector<size_t>& n) {
    const auto blocks = reader.BlocksInfo(v, step);
    std::vector<T> buf;
    for (size_t b = 0; b < blocks.size(); ++b) {
        size_t cnt = 1; for (auto c : blocks[b].Count) cnt *= c;
        buf.resize(cnt); v.SetStepSelection({step, 1});
        if (v.Shape().empty()) v.SetBlockSelection(b); else v.SetSelection({blocks[b].Start, blocks[b].Count});
        reader.Get(v, buf.data(), adios2::Mode::Sync);
        for (size_t i = 0; i < cnt; ++i) { const int c = static_cast<int>(i % ncomp); const double x = buf[i]; ss[c] += x * x; ++n[c]; lo[c] = std::min(lo[c], x); hi[c] = std::max(hi[c], x); }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: hpmgard_adios_compress in.bp out.bp --tol T [options]\n"; return 2; }
    const std::string in_file = argv[1], out_file = argv[2];
    std::map<std::string, std::string> a;
    for (int i = 3; i < argc; ++i) {
        std::string k = argv[i]; if (k.rfind("--", 0) != 0) { std::cerr << "unexpected " << k << "\n"; return 2; } k = k.substr(2);
        if (k == "no-mesh" || k == "verbose" || k == "block-norm") a[k] = "1"; else { if (i + 1 >= argc) { std::cerr << "missing value for --" << k << "\n"; return 2; } a[k] = argv[++i]; }
    }
    auto get = [&](const std::string& k, const std::string& d) { auto it = a.find(k); return it == a.end() ? d : it->second; };
    if (!a.count("tol")) { std::cerr << "--tol is required\n"; return 2; }
    const std::set<std::string> only = [&]() { std::set<std::string> s; for (auto& v : Split(get("vars", ""), ',')) s.insert(v); return s; }();
    const size_t max_steps = std::stoul(get("steps", "1000000000"));

    adios2::ADIOS ad;
    adios2::IO in = ad.DeclareIO("in"), out = ad.DeclareIO("out");
    adios2::Engine reader = in.Open(in_file, adios2::Mode::ReadRandomAccess);
    adios2::Engine writer = out.Open(out_file, adios2::Mode::Write);
    hpmt::CopyAttributes(in, out);
    const auto attrs = in.AvailableAttributes();
    auto attr_str = [&](const std::string& name) -> std::string { if (!attrs.count(name)) return ""; auto at = in.InquireAttribute<std::string>(name); return at ? at.Data()[0] : ""; };
    auto attr_int = [&](const std::string& name) -> int { if (!attrs.count(name)) return -1; auto at = in.InquireAttribute<int32_t>(name); return at ? at.Data()[0] : -1; };

    // discover compressible variables: "<celltype>/<name>" with attribute "<name>/<celltype>/order"
    struct Target { std::string var, name, cell_type; int order; std::string conn, coords; };
    std::vector<Target> targets; std::set<std::string> target_names;
    for (const auto& kv : in.AvailableVariables()) {
        const std::string& var = kv.first;
        const size_t slash = var.find('/');
        if (slash == std::string::npos) continue;
        const std::string ct = var.substr(0, slash), name = var.substr(slash + 1);
        if (ct.rfind("vtkDG", 0) != 0) continue;
        const std::string type = in.VariableType(var);
        if (type != "double" && type != "float") continue;
        const int order = attr_int(name + "/" + ct + "/order");
        if (order < 0) continue;
        if (!only.empty() && !only.count(var) && !only.count(name)) continue;
        Target t{var, name, ct, order, attr_str("shape/" + ct + "/connectivity"), attr_str("shape/" + ct + "/values")};
        targets.push_back(t); target_names.insert(var);
    }
    if (targets.empty()) { std::cerr << "no compressible vtkDG* variables found\n"; return 3; }
    std::cerr << "compressing " << targets.size() << " variables:"; for (auto& t : targets) std::cerr << " " << t.var << "(p" << t.order << ")"; std::cerr << "\n";

    adios2::Params common{{"PluginName", "CompressHpMGARDOperator"}, {"PluginLibrary", "CompressHpMGARDOperator"},
                          {"tolerance", a["tol"]}, {"mode", get("mode", "REL_L2")}, {"tuning", get("tuning", "auto")},
                          {"tuning_sample", get("tune-sample", "0.03")}, {"threads", get("threads", "0")}, {"threshold", get("threshold", "100000")}};
    for (const char* k : {"base-method", "base-order", "base-quantum", "base-codec", "residual-codec", "tune-codecs", "node-set"}) {
        if (a.count(k)) { std::string key = k; for (auto& c : key) if (c == '-') c = '_'; common[key] = a[k]; }
    }
    if (a.count("verbose")) common["verbose"] = "1";
    const bool use_mesh = !a.count("no-mesh");

    const bool block_norm = a.count("block-norm") > 0;   // legacy: let every block normalise its own tolerance
    const std::string mode = get("mode", "REL_L2");
    const double eb = std::stod(a["tol"]);
    // Variable-wide scale per component over all blocks of a step (RMS for REL_L2, range for REL_RANGE),
    // so that one absolute bound applies to every block of the variable.
    auto variable_tolerances = [&](const Target& t, size_t step) -> std::string {
        if (block_norm || mode == "ABS") return a["tol"];
        const std::string type = in.VariableType(t.var);
        const int ncomp = std::max(1, attr_int(t.name + "/components"));
        std::vector<double> ss(ncomp, 0.0), lo(ncomp, 1e300), hi(ncomp, -1e300); std::vector<size_t> n(ncomp, 0);
        if (type == "double") AccumulateStats<double>(in.InquireVariable<double>(t.var), reader, step, ncomp, ss, lo, hi, n);
        else AccumulateStats<float>(in.InquireVariable<float>(t.var), reader, step, ncomp, ss, lo, hi, n);
        std::string list;
        for (int c = 0; c < ncomp; ++c) {
            const double scale = (mode == "REL_RANGE") ? (hi[c] - lo[c]) : (n[c] ? std::sqrt(ss[c] / n[c]) : 0.0);
            char b[64]; std::snprintf(b, sizeof b, "%.17g", eb * scale); list += (c ? "," : "") + std::string(b);
        }
        return list;
    };

    const size_t nsteps = std::min(max_steps, reader.Steps());
    for (size_t step = 0; step < nsteps; ++step) {
        const auto t0 = Clock::now();
        writer.BeginStep();
        for (const auto& kv : in.AvailableVariables()) {
            const std::string& var = kv.first;
            if (!target_names.count(var)) { hpmt::CopyVariable(in, reader, out, writer, var, step); continue; }
            const Target& t = *std::find_if(targets.begin(), targets.end(), [&](const Target& x) { return x.var == var; });
            adios2::Params p = common;
            p["cell_type"] = t.cell_type; p["order"] = std::to_string(t.order);
            p["tolerance"] = variable_tolerances(t, step); if (!block_norm && mode != "ABS") p["mode"] = "ABS";
            if (use_mesh && !t.conn.empty() && !t.coords.empty()) { p["meshfile"] = in_file; p["connectivity_variable"] = t.conn; p["coordinates_variable"] = t.coords; }
            auto decorate_d = [&](adios2::Variable<double>& v, size_t block) { v.RemoveOperations(); adios2::Params q = p; q["blockid"] = std::to_string(block); v.AddOperation("plugin", q); };
            auto decorate_f = [&](adios2::Variable<float>& v, size_t block) { v.RemoveOperations(); adios2::Params q = p; q["blockid"] = std::to_string(block); v.AddOperation("plugin", q); };
            size_t nb = 0;
            if (in.VariableType(var) == "double") nb = hpmt::CopyVariableT<double>(in, reader, out, writer, var, step, decorate_d);
            else nb = hpmt::CopyVariableT<float>(in, reader, out, writer, var, step, decorate_f);
            std::cerr << "  step " << step << " " << var << ": " << nb << " block(s) compressed\n";
        }
        writer.EndStep();
        std::cerr << "step " << step << " done in " << std::chrono::duration<double>(Clock::now() - t0).count() << " s\n";
    }
    writer.Close(); reader.Close();
    return 0;
}
