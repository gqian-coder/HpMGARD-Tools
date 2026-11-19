// Helpers shared by the ADIOS2 drivers: copy attributes and variables (all types, all blocks) between
// two BP files, step by step. Operators attached to a variable on the output side are applied by ADIOS2
// at Put time; on the input side ADIOS2 decodes transparently (the plugin must be on ADIOS2_PLUGIN_PATH).
#pragma once
#include <adios2.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace hpmt {

#define HPMT_FOREACH_TYPE(MACRO) \
    MACRO(int8_t) MACRO(int16_t) MACRO(int32_t) MACRO(int64_t) MACRO(uint8_t) MACRO(uint16_t) MACRO(uint32_t) MACRO(uint64_t) \
    MACRO(float) MACRO(double) MACRO(long double) MACRO(std::complex<float>) MACRO(std::complex<double>)

inline void CopyAttributes(adios2::IO& in, adios2::IO& out) {
    for (const auto& kv : in.AvailableAttributes()) {
        const std::string& name = kv.first;
        const std::string type = in.AttributeType(name);
        if (out.InquireAttribute<std::string>(name)) continue;
        if (type == "string") {
            auto a = in.InquireAttribute<std::string>(name);
            const auto d = a.Data();
            if (d.size() == 1) out.DefineAttribute<std::string>(name, d[0]); else out.DefineAttribute<std::string>(name, d.data(), d.size());
            continue;
        }
#define HPMT_COPY_ATTR(T) \
        if (type == adios2::GetType<T>()) { auto a = in.InquireAttribute<T>(name); const auto d = a.Data(); if (d.size() == 1) out.DefineAttribute<T>(name, d[0]); else out.DefineAttribute<T>(name, d.data(), d.size()); continue; }
        HPMT_FOREACH_TYPE(HPMT_COPY_ATTR)
#undef HPMT_COPY_ATTR
        std::cerr << "warning: attribute " << name << " of type " << type << " not copied\n";
    }
}

// Copies every block of `name` for the current step. `decorate(var)` may attach operations to the
// output variable (called once after definition). Returns the number of blocks copied.
template <class T>
size_t CopyVariableT(adios2::IO& in, adios2::Engine& reader, adios2::IO& out, adios2::Engine& writer, const std::string& name,
                     size_t step, const std::function<void(adios2::Variable<T>&, size_t)>& decorate) {
    adios2::Variable<T> vin = in.InquireVariable<T>(name);
    if (!vin) return 0;
    adios2::Variable<T> vout = out.InquireVariable<T>(name);
    if (vin.ShapeID() == adios2::ShapeID::GlobalValue) {   // single value per step (e.g. time)
        if (!vout) vout = out.DefineVariable<T>(name);
        T v{}; vin.SetStepSelection({step, 1}); reader.Get(vin, &v, adios2::Mode::Sync);
        writer.Put(vout, v, adios2::Mode::Sync);
        return 1;
    }
    const auto blocks = reader.BlocksInfo(vin, step);
    const bool global = !vin.Shape().empty();
    if (!vout) {
        if (global) vout = out.DefineVariable<T>(name, vin.Shape(), blocks.empty() ? adios2::Dims(vin.Shape().size(), 0) : blocks[0].Start, blocks.empty() ? vin.Shape() : blocks[0].Count);
        else if (vin.ShapeID() == adios2::ShapeID::LocalValue) vout = out.DefineVariable<T>(name, {adios2::LocalValueDim});
        else vout = out.DefineVariable<T>(name, {}, {}, blocks.empty() ? adios2::Dims{1} : blocks[0].Count);
    }
    std::vector<T> buf;
    for (size_t b = 0; b < blocks.size(); ++b) {
        vin.SetStepSelection({step, 1});
        size_t n = 1; for (auto c : blocks[b].Count) n *= c;
        if (vin.ShapeID() == adios2::ShapeID::LocalValue) n = 1;
        buf.resize(n);
        if (global) vin.SetSelection({blocks[b].Start, blocks[b].Count}); else vin.SetBlockSelection(b);
        reader.Get(vin, buf.data(), adios2::Mode::Sync);
        if (global) vout.SetSelection({blocks[b].Start, blocks[b].Count});
        else if (vin.ShapeID() != adios2::ShapeID::LocalValue) vout.SetSelection({{}, blocks[b].Count});
        if (decorate) decorate(vout, b);
        writer.Put(vout, buf.data(), adios2::Mode::Sync);
    }
    return blocks.size();
}

// Type-dispatching copy with no decoration.
inline size_t CopyVariable(adios2::IO& in, adios2::Engine& reader, adios2::IO& out, adios2::Engine& writer, const std::string& name, size_t step) {
    const std::string type = in.VariableType(name);
    if (type == "string") {
        auto v = in.InquireVariable<std::string>(name); std::string s; v.SetStepSelection({step, 1}); reader.Get(v, s, adios2::Mode::Sync);
        auto o = out.InquireVariable<std::string>(name); if (!o) o = out.DefineVariable<std::string>(name);
        writer.Put(o, s, adios2::Mode::Sync); return 1;
    }
#define HPMT_COPY_VAR(T) if (type == adios2::GetType<T>()) return CopyVariableT<T>(in, reader, out, writer, name, step, nullptr);
    HPMT_FOREACH_TYPE(HPMT_COPY_VAR)
#undef HPMT_COPY_VAR
    std::cerr << "warning: variable " << name << " of type " << type << " not copied\n";
    return 0;
}

} // namespace hpmt
