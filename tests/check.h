// Minimal test helpers (no framework dependency).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static int g_checks = 0, g_failures = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failures; std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_NEAR(a, b, tol) do { ++g_checks; const double _a = (a), _b = (b); if (!(std::fabs(_a - _b) <= (tol))) { ++g_failures; std::printf("  FAIL %s:%d: %s = %.17g vs %s = %.17g (tol %g)\n", __FILE__, __LINE__, #a, _a, #b, _b, (double)(tol)); } } while (0)
#define CHECK_THROWS(expr) do { ++g_checks; bool _t = false; try { expr; } catch (const std::exception&) { _t = true; } if (!_t) { ++g_failures; std::printf("  FAIL %s:%d: expected exception from %s\n", __FILE__, __LINE__, #expr); } } while (0)
static int Report(const char* name) {
    std::printf("%s: %d checks, %d failures\n", name, g_checks, g_failures);
    return g_failures ? 1 : 0;
}
