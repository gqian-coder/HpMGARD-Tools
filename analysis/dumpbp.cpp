// dumpbp: dump one step of one ADIOS2 BP variable (double or int64) to a raw
// little-endian binary file. Handles global arrays (one block) and local arrays
// (all blocks concatenated, block extents written to <out>.blocks).
// usage: dumpbp <file.bp> <variable> <step> <out.bin>
#include <adios2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <class T>
static void Dump(adios2::IO &io, adios2::Engine &eng, const std::string &name,
                 size_t step, const std::string &out)
{
    adios2::Variable<T> var = io.InquireVariable<T>(name);
    if (!var) { std::cerr << "no variable " << name << "\n"; std::exit(2); }
    auto blocks = eng.BlocksInfo(var, step);
    std::ofstream of(out, std::ios::binary);
    std::ofstream bf(out + ".blocks");
    size_t total = 0;
    const bool global = !var.Shape().empty();
    if (global)
    {
        var.SetStepSelection({step, 1});
        auto shape = var.Shape();
        adios2::Dims start(shape.size(), 0);
        var.SetSelection({start, shape});
        size_t n = 1; for (auto s : shape) n *= s;
        std::vector<T> buf(n);
        eng.Get(var, buf.data(), adios2::Mode::Sync);
        of.write(reinterpret_cast<const char *>(buf.data()), n * sizeof(T));
        total = n;
        bf << "global";
        for (auto s : shape) bf << " " << s;
        bf << "\n";
    }
    else
    {
        for (size_t b = 0; b < blocks.size(); ++b)
        {
            var.SetStepSelection({step, 1});
            var.SetBlockSelection(b);
            size_t n = 1; for (auto c : blocks[b].Count) n *= c;
            std::vector<T> buf(n);
            eng.Get(var, buf.data(), adios2::Mode::Sync);
            of.write(reinterpret_cast<const char *>(buf.data()), n * sizeof(T));
            total += n;
            bf << "block " << b;
            for (auto c : blocks[b].Count) bf << " " << c;
            bf << "\n";
        }
    }
    std::cout << name << " step " << step << ": " << blocks.size() << " block(s), "
              << total << " values, " << total * sizeof(T) << " bytes -> " << out << "\n";
}

int main(int argc, char **argv)
{
    if (argc < 5) { std::cerr << "usage: dumpbp <file.bp> <variable> <step> <out.bin>\n"; return 1; }
    const std::string file = argv[1], name = argv[2], out = argv[4];
    const size_t step = std::strtoul(argv[3], nullptr, 10);
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("dump");
    adios2::Engine eng = io.Open(file, adios2::Mode::ReadRandomAccess);
    const std::string type = io.VariableType(name);
    if (type == "double") Dump<double>(io, eng, name, step, out);
    else if (type == "int64_t") Dump<int64_t>(io, eng, name, step, out);
    else if (type == "int32_t") Dump<int32_t>(io, eng, name, step, out);
    else if (type == "float") Dump<float>(io, eng, name, step, out);
    else { std::cerr << "unsupported type " << type << "\n"; return 3; }
    eng.Close();
    return 0;
}
