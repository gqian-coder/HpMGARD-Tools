#include "check.h"
#include <hpmgard/Operators.h>
#include <hpmgard/Quadrature.h>
#include <random>
using namespace hpmgard;

// Sample a polynomial given by coefficients c[i][j][k] of x^i y^j z^k at the tensor Gauss nodes of order p.
static std::vector<double> SampleHex(int p, const std::vector<double>& coef, int deg, size_t ncells, std::mt19937& rng) {
    const std::vector<double> x = GaussLegendreNodes(p + 1);
    const int n = p + 1;
    std::vector<double> u(ncells * n * n * n);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (size_t c = 0; c < ncells; ++c) {
        const double shift = U(rng);   // per-cell offset keeps cells distinct
        for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
            double v = shift;
            for (int a = 0; a <= deg; ++a) for (int b = 0; b <= deg; ++b) for (int d = 0; d <= deg; ++d)
                v += coef[(a * (deg + 1) + b) * (deg + 1) + d] * std::pow(x[i], a) * std::pow(x[j], b) * std::pow(x[k], d);
            u[((c * n + k) * n + j) * n + i] = v;
        }
    }
    return u;
}

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const size_t ncells = 50;
    // 1. a trilinear polynomial (degree <= 1 per axis) survives p4 -> p1 -> p4 exactly, Gauss nodes, closed base
    {
        std::vector<double> coef(8); for (auto& c : coef) c = U(rng);
        const std::vector<double> u = SampleHex(4, coef, 1, ncells, rng);
        OperatorSet ops = BuildOperators(ElemShape::HEX, 4, 1, NodeSet::GaussLegendre, BaseRepresentation::Closed);
        CHECK(ops.base_rep == BaseRepresentation::Closed); CHECK(ops.nsp_base == 8); CHECK(ops.nsp_data == 125);
        CHECK_NEAR(ops.amp_inf, 1.0, 1e-12);
        std::vector<double> base(ncells * 8), back(u.size());
        ops.Restrict(u.data(), ncells, base.data()); ops.Prolong(base.data(), ncells, back.data());
        double err = 0; for (size_t i = 0; i < u.size(); ++i) err = std::max(err, std::fabs(u[i] - back[i]));
        CHECK(err < 1e-12);
        // the stored base values are the corner values of the polynomial
        OperatorSet dn = BuildOperators(ElemShape::HEX, 4, 1, NodeSet::GaussLegendre, BaseRepresentation::DataNodes);
        CHECK_NEAR(dn.amp_inf, 3.8665628944393693, 1e-9);
        std::vector<double> base2(ncells * 8), back2(u.size());
        dn.Restrict(u.data(), ncells, base2.data()); dn.Prolong(base2.data(), ncells, back2.data());
        err = 0; for (size_t i = 0; i < u.size(); ++i) err = std::max(err, std::fabs(u[i] - back2[i]));
        CHECK(err < 1e-12);
        // same polynomial, different representation: corner values differ from Gauss p1 values
        double diff = 0; for (size_t i = 0; i < base.size(); ++i) diff = std::max(diff, std::fabs(base[i] - base2[i]));
        CHECK(diff > 1e-3);
    }
    // 2. a degree-4 field is NOT reproduced by p1 (sanity), and the Gauss restriction is the L2 projection:
    //    residual is orthogonal to the trilinear space under the Gauss quadrature weights
    {
        std::vector<double> coef(125); for (auto& c : coef) c = U(rng);
        const std::vector<double> u = SampleHex(4, coef, 4, ncells, rng);
        OperatorSet ops = BuildOperators(ElemShape::HEX, 4, 1, NodeSet::GaussLegendre, BaseRepresentation::Closed);
        std::vector<double> base(ncells * 8), back(u.size());
        ops.Restrict(u.data(), ncells, base.data()); ops.Prolong(base.data(), ncells, back.data());
        const std::vector<double> x = GaussLegendreNodes(5), w = GaussLegendreWeights(5);
        double maxres = 0, maxorth = 0;
        for (size_t c = 0; c < ncells; ++c) {
            // inner products of the residual with 1, x, y, z, xy, ... (trilinear basis) must vanish
            for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int d = 0; d < 2; ++d) {
                double ip = 0;
                for (int k = 0; k < 5; ++k) for (int j = 0; j < 5; ++j) for (int i = 0; i < 5; ++i) {
                    const size_t idx = ((c * 5 + k) * 5 + j) * 5 + i;
                    const double r = u[idx] - back[idx]; maxres = std::max(maxres, std::fabs(r));
                    ip += w[i] * w[j] * w[k] * r * std::pow(x[i], a) * std::pow(x[j], b) * std::pow(x[k], d);
                }
                maxorth = std::max(maxorth, std::fabs(ip));
            }
        }
        CHECK(maxres > 1e-3); CHECK(maxorth < 1e-12);
    }
    // 3. data sampled at Gauss nodes but processed with Gauss-Lobatto operators: a linear field is no longer exact
    {
        std::vector<double> coef(8); for (auto& c : coef) c = U(rng);
        const std::vector<double> u = SampleHex(2, coef, 1, ncells, rng);
        OperatorSet wrong = BuildOperators(ElemShape::HEX, 2, 1, NodeSet::GaussLobatto, BaseRepresentation::Closed);
        OperatorSet right = BuildOperators(ElemShape::HEX, 2, 1, NodeSet::GaussLegendre, BaseRepresentation::Closed);
        std::vector<double> b(ncells * 8), back(u.size());
        wrong.Restrict(u.data(), ncells, b.data()); wrong.Prolong(b.data(), ncells, back.data());
        double e_wrong = 0; for (size_t i = 0; i < u.size(); ++i) e_wrong = std::max(e_wrong, std::fabs(u[i] - back[i]));
        right.Restrict(u.data(), ncells, b.data()); right.Prolong(b.data(), ncells, back.data());
        double e_right = 0; for (size_t i = 0; i < u.size(); ++i) e_right = std::max(e_right, std::fabs(u[i] - back[i]));
        CHECK(e_right < 1e-12); CHECK(e_wrong > 1e-3);
    }
    // 4. cell-mean base (p_base = 0) and no-split (p_base = p)
    {
        std::vector<double> coef(27); for (auto& c : coef) c = U(rng);
        const std::vector<double> u = SampleHex(2, coef, 2, ncells, rng);
        OperatorSet m = BuildOperators(ElemShape::HEX, 2, 0, NodeSet::GaussLegendre, BaseRepresentation::Closed);
        CHECK(m.nsp_base == 1); CHECK_NEAR(m.amp_inf, 1.0, 1e-14);
        std::vector<double> base(ncells);
        m.Restrict(u.data(), ncells, base.data());
        const std::vector<double> w = GaussLegendreWeights(3);
        double ref = 0; for (int k = 0; k < 3; ++k) for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) ref += w[i] * w[j] * w[k] * u[(k * 3 + j) * 3 + i] / 8.0;
        CHECK_NEAR(base[0], ref, 1e-13);
        OperatorSet id = BuildOperators(ElemShape::HEX, 2, 2, NodeSet::GaussLegendre, BaseRepresentation::Closed);
        std::vector<double> b2(u.size()), back(u.size());
        id.Restrict(u.data(), ncells, b2.data()); id.Prolong(b2.data(), ncells, back.data());
        double e = 0; for (size_t i = 0; i < u.size(); ++i) e = std::max(e, std::fabs(u[i] - b2[i]) + std::fabs(u[i] - back[i]));
        CHECK(e < 1e-13);
    }
    // 5. simplex tables: R P = I for TET, PRISM, PYRAMID (p -> p_base) via the ported hpMusic matrices
    for (ElemShape s : {ElemShape::TET, ElemShape::PRISM, ElemShape::PYRAMID}) {
        for (int p = 2; p <= 4; ++p) {
            OperatorSet o = BuildOperators(s, p, 1, NodeSet::GaussLegendre, BaseRepresentation::Closed);
            CHECK(o.base_rep == BaseRepresentation::DataNodes); CHECK(!o.tensor);
            const std::vector<double> RP = MatMul(o.Rfull, o.nsp_base, o.nsp_data, o.Pfull, o.nsp_data, o.nsp_base);
            double e = 0; for (int i = 0; i < o.nsp_base; ++i) for (int j = 0; j < o.nsp_base; ++j) e = std::max(e, std::fabs(RP[i * o.nsp_base + j] - (i == j ? 1.0 : 0.0)));
            CHECK(e < 1e-10);
            CHECK(o.amp_inf >= 1.0);
        }
    }
    return Report("test_operators");
}
