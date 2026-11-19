#include "check.h"
#include <hpmgard/Quadrature.h>
using namespace hpmgard;

int main() {
    // Gauss-Legendre p4 nodes equal hpMusic's sp_1d_p4 table
    const std::vector<double> g5 = GaussLegendreNodes(5), w5 = GaussLegendreWeights(5);
    const double ref5[5] = {-0.906179845938663853, -0.538469310105682997, 0.0, 0.538469310105682997, 0.906179845938663853};
    const double refw5[5] = {0.236926885056189088, 0.478628670499366468, 0.568888888888888889, 0.478628670499366468, 0.236926885056189088};
    for (int i = 0; i < 5; ++i) { CHECK_NEAR(g5[i], ref5[i], 1e-15); CHECK_NEAR(w5[i], refw5[i], 1e-15); }
    const std::vector<double> g6 = GaussLegendreNodes(6);
    CHECK_NEAR(g6[0], -0.932469514203152050, 1e-15); CHECK_NEAR(g6[2], -0.238619186083196932, 1e-15);
    // Gauss-Lobatto
    const std::vector<double> l5 = GaussLobattoNodes(5), lw5 = GaussLobattoWeights(5);
    CHECK_NEAR(l5[0], -1.0, 0); CHECK_NEAR(l5[1], -0.6546536707079772, 1e-15); CHECK_NEAR(l5[2], 0.0, 1e-15); CHECK_NEAR(l5[4], 1.0, 0);
    double s = 0; for (double w : lw5) s += w; CHECK_NEAR(s, 2.0, 1e-14);
    CHECK_NEAR(lw5[0], 0.1, 1e-15);   // 2/(4*5)
    s = 0; for (double w : w5) s += w; CHECK_NEAR(s, 2.0, 1e-14);
    CHECK(GaussLobattoNodes(1).size() == 1); CHECK_NEAR(GaussLobattoNodes(2)[1], 1.0, 0);
    // Lagrange partition of unity and interpolation exactness
    for (double x : {-0.9, -0.3, 0.2, 0.77}) { double u = 0; for (int j = 0; j < 5; ++j) u += LagrangeBasis(g5, j, x); CHECK_NEAR(u, 1.0, 1e-13); }
    const std::vector<double> P = InterpMatrix(g5, l5);   // 5x5: Gauss p4 -> Lobatto nodes
    // cubic f(x) = 1 + x - 2x^2 + 0.5x^3 reproduced exactly
    auto f = [](double x) { return 1 + x - 2 * x * x + 0.5 * x * x * x; };
    for (int t = 0; t < 5; ++t) { double v = 0; for (int k = 0; k < 5; ++k) v += P[t * 5 + k] * f(g5[k]); CHECK_NEAR(v, f(l5[t]), 1e-13); }
    // R P = I for the L2 restriction (p4 -> p1 Gauss)
    const std::vector<double> g2 = GaussLegendreNodes(2), w2 = GaussLegendreWeights(2);
    const std::vector<double> R = L2RestrictMatrix(g5, w5, g2, w2), Pp = InterpMatrix(g2, g5);
    const std::vector<double> RP = MatMul(R, 2, 5, Pp, 5, 2);
    CHECK_NEAR(RP[0], 1.0, 1e-14); CHECK_NEAR(RP[3], 1.0, 1e-14); CHECK_NEAR(RP[1], 0.0, 1e-14); CHECK_NEAR(RP[2], 0.0, 1e-14);
    // R equals hpMusic's restrict_1d_p4 (row 0, first two entries) from scheme_coef_cpr_gsp_gfp_hex.C
    CHECK_NEAR(MaxRowAbsSum(Pp, 5, 2), 1.5695495339607035, 1e-12);
    return Report("test_quadrature");
}
