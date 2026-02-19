# pragma version 0.4.3
# pragma optimize gas

# =============================================================================
# StableSwap (n=2), D=1, fixed-point WAD=1e18
#
# Goal:
#   Given amplification A and target marginal price p_target = -dx/dy,
#   compute portfolio value in x-units:
#     V = x + p_target * y
#   where (x, y) lies on the StableSwap invariant.
#
# Notation:
#   A_eff := A_raw / A_PRECISION
#   y := y/D, with D=1 normalization
#   g(y) := p(y) - p_target
#
# 1) Invariant and x(y)
#   For n=2, D=1:
#     4*A_eff*(x + y) + 1 = 4*A_eff + 1/(4*x*y)
#   Rearranged:
#     4*A_eff*x^2 + (4*A_eff*(y-1) + 1)*x - 1/(4*y) = 0
#   With b1 = 4*A_eff*(y-1)+1:
#     x(y) = (-b1 + sqrt(b1^2 + 4*A_eff/y)) / (8*A_eff)
#
# 2) Marginal price p(y)
#   From implicit differentiation of F(x,y)=0:
#     p(y) = -dx/dy
#          = (4*A_eff + 1/(4*x*y^2)) / (4*A_eff + 1/(4*x^2*y))
#
# 3) Value at fixed y
#     V(y) = x(y) + p_target * y
#
# 4) Root equation solved by numeric method
#     g(y) = p(y) - p_target = 0
#   On the relevant branch p(y) is monotone decreasing in y, so bracketing works.
#
# 5) Mapping used by this implementation
#   b1   = WAD + (4*A_raw*(y - WAD))/A_PRECISION
#   rad2 = b1^2 + (4*A_raw*WAD^3)/(A_PRECISION*y)
#   x    = ((-b1 + sqrt(rad2)) * A_PRECISION) / (8*A_raw)
#   p    = ((4*A_raw*x/A_PRECISION + WAD^3/(4*y^2)) * WAD) /
#          (4*A_raw*x/A_PRECISION + WAD^3/(4*x*y))
#
# 6) Symmetry for p_target < 1
#   Solve reciprocal branch at p_inv ~= WAD^2 / p_target, then map back:
#     V(p_target) = p_target * V(p_inv)
#     (x, y) at p_target is (y, x) at p_inv.
#
# 7) Method used here: safeguarded Brent-style updates on g(y)
#   Keep bracket [lo, hi] with g(lo) >= 0 >= g(hi), try secant candidate:
#     y_sec = hi - g(hi) * (hi - lo) / (g(hi) - g(lo))
#   Accept secant step only if it stays well inside bracket; otherwise fallback
#   to midpoint. This combines fast progress of secant with bisection safety.
# =============================================================================
WAD: constant(uint256) = 10**18
WAD2: constant(uint256) = WAD * WAD
WAD3: constant(uint256) = WAD2 * WAD

A_PRECISION: constant(uint256) = 10**4
MAX_A: constant(uint256) = 100_000
MAX_A_PRECISION: constant(uint256) = 10_000
MAX_A_RAW: constant(uint256) = MAX_A * MAX_A_PRECISION

BISECT_STEPS: constant(uint256) = 7
BRENT_STEPS: constant(uint256) = 64
PRICE_TOL_REL: constant(uint256) = 10**6


@internal
@pure
def _x_from_y(A_raw: uint256, y: uint256) -> uint256:
    # Invariant quadratic in x:
    #   4A*x^2 + (4A*(y-1)+1)*x - 1/(4y) = 0
    # Positive root:
    #   x(y) = (-b1 + sqrt(b1^2 + 4A/y)) / (8A), b1 = 1 - 4A*(1-y)
    #
    # Error bound for fixed-point rounding (absolute, in output wei):
    #   x*     = ((sqrt(b^2 + t) - b) * A_PRECISION) / (8*A_raw)      (exact real)
    #   b      = WAD - (4*A_raw*(WAD-y))/A_PRECISION
    #   t      = (4*A_raw*WAD^3)/(A_PRECISION*y)
    #   b_hat  = WAD - floor(4*A_raw*(WAD-y)/A_PRECISION), |b_hat-b| < 1
    #   t_hat  = floor(t),                                    |t_hat-t| < 1
    #   r_hat  = floor(sqrt(b_hat^2 + t_hat))
    #   x_hat  = floor(((r_hat - b_hat) * A_PRECISION)/(8*A_raw))
    # Using |d sqrt(b^2+t)/db| <= 1 and |d sqrt(b^2+t)/dt| = 1/(2*sqrt(b^2+t)) << 1:
    #   |r_hat - sqrt(b^2+t)| < 2
    # Therefore:
    #   |x_hat - x*| < 1 + (3*A_PRECISION)/(8*A_raw)
    #   => for A_raw >= 1:            |x_hat - x*| < 3751 wei
    #   => for A_raw >= A_PRECISION:  |x_hat - x*| < 2 wei
    b1: int256 = convert(WAD, int256) - convert(4 * A_raw * (WAD - y) // A_PRECISION, int256)

    abs_b1: uint256 = convert(abs(b1), uint256)
    term: uint256 = unsafe_div(4 * A_raw * WAD3, A_PRECISION * y)
    rad: int256 = convert(isqrt(abs_b1**2 + term), int256)
    if rad <= b1:
        return 0

    return (convert(rad - b1, uint256) * A_PRECISION) // (8 * A_raw)


@internal
@pure
def _p_from_y(A_raw: uint256, y: uint256) -> uint256:
    # p(y) = -dx/dy:
    #   p(y) = (4A + 1/(4*x*y^2)) / (4A + 1/(4*x^2*y))
    #        = (4A*x + 1/(4*y^2)) / (4A*x + 1/(4*x*y))
    #
    # Error propagation (absolute, in output wei):
    #   p*   = WAD * (N / D), with:
    #          N = a + u,  D = a + v
    #          a = (4*A_raw/A_PRECISION) * x*
    #          u = WAD^3/(4*y^2)
    #          v = WAD^3/(4*x* y)
    #   p_hat uses x_hat from _x_from_y and floor divisions.
    #
    # Let E_x = |x_hat - x*| from _x_from_y bound, alpha = 4*A_raw/A_PRECISION,
    # beta = WAD^3/(4*y). For x* > E_x:
    #   |delta_a| <= alpha * E_x + 1
    #   |delta_u| < 1
    #   |delta_v| <= beta * E_x / (x* * (x* - E_x)) + 1
    #
    # Define:
    #   deltaN = |delta_a| + |delta_u|
    #   deltaD = |delta_a| + |delta_v|
    # Then for deltaD < D:
    #   |p_hat - p*| <= 1 + WAD * (deltaN * D + N * deltaD) / (D * (D - deltaD))
    #
    # Equivalent relative form (first-order):
    #   |p_hat - p*| / p* ~= deltaN / N + deltaD / D + 1 / p*
    #
    # Empirical examples on solver domain y in [WAD/10^5, WAD/2+1]
    # (dense y-sweep, high-precision reference; illustrative, not a proof):
    #   A_eff = 1      (A_raw = 1 * A_PRECISION):       |p_hat - p*| <= ~6.3e3 wei
    #   A_eff = 200    (A_raw = 200 * A_PRECISION):     |p_hat - p*| <= ~4.1e3 wei
    #   A_eff = 10_000 (A_raw = 10_000 * A_PRECISION):  |p_hat - p*| <= ~1.3e4 wei
    #   Relative error in all those sweeps is about 1e-18.
    x: uint256 = self._x_from_y(A_raw, y)
    if x == 0:
        return max_value(uint256)

    term4A: uint256 = (4 * A_raw * x) // A_PRECISION
    return unsafe_div(
        (term4A + unsafe_div(WAD3, 4 * y * y)) * WAD,
        term4A + unsafe_div(WAD3, 4 * x * y),
    )


@internal
@pure
def _y_from_brent(A_raw: uint256, p: uint256) -> uint256:
    # Brent-style root finder for g(y)=p(y)-p_target on [lo, hi].
    assert p >= WAD
    lo: uint256 = WAD // 10**5
    hi: uint256 = WAD // 2 + 1

    plo: uint256 = self._p_from_y(A_raw, lo)
    phi: uint256 = self._p_from_y(A_raw, hi)

    # Warmup bisection by sign of g(mid).
    for _: uint256 in range(BISECT_STEPS):
        mid: uint256 = unsafe_div(unsafe_add(lo, hi), 2)
        pm: uint256 = self._p_from_y(A_raw, mid)

        if pm > p:
            lo = mid
            plo = pm
        else:
            hi = mid
            phi = pm

    p_i: int256 = convert(p, int256)
    tol_abs: uint256 = unsafe_div(p, PRICE_TOL_REL)
    g_lo: int256 = convert(plo, int256) - p_i
    g_hi: int256 = convert(phi, int256) - p_i

    if convert(abs(g_lo), uint256) <= tol_abs:
        return lo
    if convert(abs(g_hi), uint256) <= tol_abs:
        return hi

    for _: uint256 in range(BRENT_STEPS):
        span: uint256 = unsafe_sub(hi, lo)
        if span <= 1:
            break

        y: uint256 = unsafe_div(unsafe_add(lo, hi), 2)

        # Secant candidate from endpoints; keep midpoint fallback.
        dg: int256 = g_hi - g_lo
        if dg != 0:
            y_i: int256 = convert(hi, int256) - unsafe_div(g_hi * (convert(hi, int256) - convert(lo, int256)), dg)
            if convert(lo, int256) < y_i and y_i < convert(hi, int256):
                y_sec: uint256 = convert(y_i, uint256)
                # Reject endpoint-clinging secant steps; otherwise they can
                # make very small progress when |g_lo| >> |g_hi|.
                if unsafe_add(lo, span // 8) < y_sec and y_sec < unsafe_sub(hi, span // 8):
                    y = y_sec

        py: uint256 = self._p_from_y(A_raw, y)
        gy: int256 = convert(py, int256) - p_i

        if py > p:
            lo = y
            plo = py
            g_lo = gy
        else:
            hi = y
            phi = py
            g_hi = gy

        if convert(abs(gy), uint256) <= tol_abs:
            break

    if unsafe_add(plo, phi) >= 2 * p:
        return hi
    return lo


@internal
@pure
def _get_x_y(A_raw: uint256, p: uint256) -> (uint256, uint256):
    assert A_raw > 0
    assert A_raw <= MAX_A_RAW
    assert p != 0

    if p < WAD:
        p_inv: uint256 = unsafe_div(WAD2 + p // 2, p)
        y_inv: uint256 = self._y_from_brent(A_raw, p_inv)
        x_inv: uint256 = self._x_from_y(A_raw, y_inv)
        return y_inv, x_inv

    y: uint256 = self._y_from_brent(A_raw, p)
    x: uint256 = self._x_from_y(A_raw, y)
    return x, y


@internal
@pure
def _portfolio_value_brent(A_raw: uint256, p: uint256) -> uint256:
    x: uint256 = 0
    y: uint256 = 0
    x, y = self._get_x_y(A_raw, p)
    return x + p * y // WAD


@external
@pure
def portfolio_value(_A_raw: uint256, _p: uint256) -> uint256:
    return self._portfolio_value_brent(_A_raw, _p)
