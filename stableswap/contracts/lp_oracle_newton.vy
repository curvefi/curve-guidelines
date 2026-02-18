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
# 7) Method used here: safeguarded quasi-Newton on g(y)
#   We still solve g(y)=0 but approximate local derivative numerically:
#     g'(y) ~ (g(y) - g_prev) / (y - y_prev)
#   Newton-like step:
#     y_new = y - g(y) / g'(y)
#           = y - g(y) * (y - y_prev) / (g(y) - g_prev)
#   If y_new leaves bracket (lo, hi), fallback to midpoint and keep bracketing.
# =============================================================================
WAD: constant(uint256) = 10**18
WAD2: constant(uint256) = WAD * WAD
WAD3: constant(uint256) = WAD2 * WAD
A_PRECISION: constant(uint256) = 10**4
MAX_A: constant(uint256) = 100_000
MAX_A_PRECISION: constant(uint256) = 10_000
MAX_A_RAW: constant(uint256) = MAX_A * MAX_A_PRECISION
BISECT_STEPS: constant(uint256) = 7
NEWTON_STEPS: constant(uint256) = 64
PRICE_TOL: constant(uint256) = 10**7


@internal
@pure
def _x_from_y(A_raw: uint256, y: uint256) -> uint256:
    # Invariant quadratic in x:
    #   4A*x^2 + (4A*(y-1)+1)*x - 1/(4y) = 0
    # Positive root:
    #   x(y) = (-b1 + sqrt(b1^2 + 4A/y)) / (8A), b1 = 1 - 4A*(1-y)
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
def _y_from_newton(A_raw: uint256, p: uint256) -> uint256:
    # Solve g(y)=0 where g(y)=p(y)-p_target.
    assert p >= WAD
    lo: uint256 = WAD // 10**5
    hi: uint256 = WAD // 2 + 1

    plo: uint256 = self._p_from_y(A_raw, lo)
    phi: uint256 = self._p_from_y(A_raw, hi)

    # Warmup bisection to tighten bracket before quasi-Newton updates.
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

    y: int256 = convert(unsafe_div(unsafe_add(lo, hi), 2), int256)
    gy: int256 = convert(self._p_from_y(A_raw, convert(y, uint256)), int256) - p_i

    y_prev: int256 = convert(lo, int256)
    g_prev: int256 = convert(plo, int256) - p_i

    for _: uint256 in range(NEWTON_STEPS):
        if convert(abs(gy), uint256) <= PRICE_TOL or unsafe_sub(hi, lo) <= 1:
            break

        y_new: uint256 = unsafe_div(unsafe_add(lo, hi), 2)

        dg: int256 = gy - g_prev
        ds: int256 = y - y_prev
        if dg != 0:
            y_new_i: int256 = y - unsafe_div(gy * ds, dg)
            if convert(lo, int256) < y_new_i and y_new_i < convert(hi, int256):
                y_new = convert(y_new_i, uint256)

        p_new: uint256 = self._p_from_y(A_raw, y_new)
        g_new: int256 = convert(p_new, int256) - p_i

        if p_new > p:
            lo = y_new
            plo = p_new
        else:
            hi = y_new
            phi = p_new

        y_prev = y
        g_prev = gy
        y = convert(y_new, int256)
        gy = g_new

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
        y_inv: uint256 = self._y_from_newton(A_raw, p_inv)
        x_inv: uint256 = self._x_from_y(A_raw, y_inv)
        return y_inv, x_inv

    y: uint256 = self._y_from_newton(A_raw, p)
    x: uint256 = self._x_from_y(A_raw, y)
    return x, y


@internal
@pure
def _portfolio_value_newton(A_raw: uint256, p: uint256) -> uint256:
    x: uint256 = 0
    y: uint256 = 0
    x, y = self._get_x_y(A_raw, p)
    return x + p * y // WAD


@external
@pure
def portfolio_value(_A_raw: uint256, _p: uint256) -> uint256:
    return self._portfolio_value_newton(_A_raw, _p)
