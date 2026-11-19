// hpmgard_adios_decompress: copy a compressed BP file into a plain BP file; ADIOS2 runs the plugin's
// InverseOperate on read, so every variable comes out decoded.
//   export ADIOS2_PLUGIN_PATH=<CompressHpMGARDOperator install>/lib64
//   hpmgard_adios_decompress in.bp out.bp [--steps N]
#include "adios_copy.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: hpmgard_adios_decompress in.bp out.bp [--steps N]\n"; return 2; }
    size_t max_steps = 1000000000;
    for (int i = 3; i + 1 < argc; ++i) if (std::string(argv[i]) == "--steps") max_steps = std::stoul(argv[i + 1]);
    adios2::ADIOS ad;
    adios2::IO in = ad.DeclareIO("in"), out = ad.DeclareIO("out");
    adios2::Engine reader = in.Open(argv[1], adios2::Mode::ReadRandomAccess);
    adios2::Engine writer = out.Open(argv[2], adios2::Mode::Write);
    hpmt::CopyAttributes(in, out);
    const size_t nsteps = std::min(max_steps, reader.Steps());
    for (size_t step = 0; step < nsteps; ++step) {
        writer.BeginStep();
        for (const auto& kv : in.AvailableVariables()) hpmt::CopyVariable(in, reader, out, writer, kv.first, step);
        writer.EndStep();
        std::cerr << "step " << step << " decoded\n";
    }
    writer.Close(); reader.Close();
    return 0;
}
