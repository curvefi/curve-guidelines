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
#   s := y (because D=1 normalization)
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
# =============================================================================
WAD: constant(uint256) = 10**18
WAD2: constant(uint256) = WAD * WAD
WAD3: constant(uint256) = WAD2 * WAD

A_PRECISION: constant(uint256) = 10**4
MAX_A: constant(uint256) = 100_000
MAX_A_PRECISION: constant(uint256) = 10_000
MAX_A_RAW: constant(uint256) = MAX_A * MAX_A_PRECISION

BISECT_STEPS: constant(uint256) = 7
SECANT_STEPS: constant(uint256) = 64
# Relative price tolerance in WAD-scale:
#   |p(y_hat)-p_target| / p_target <= 1 / PRICE_TOL_REL
PRICE_TOL_REL: constant(uint256) = 10**6
# Error notation used below:
#   eps_p_abs := |p(y_hat) - p_target|          (WAD-scaled absolute price error)
#   eps_p_rel := eps_p_abs / p_target           (relative price error)
#   eps_V     := |V(y_hat) - V(y_star)|         (WAD-scaled value error)
#
# For tolerance stop:
#   eps_p_rel <= 1 / PRICE_TOL_REL


@internal
@pure
def _x_from_y(A_raw: uint256, y: uint256) -> uint256:
    # x(s) from invariant quadratic:
    #   4A*x^2 + (4A*(y-1)+1)*x - 1/(4y) = 0
    #   x(y) = (-b1 + sqrt(b1^2 + 4A/y)) / (8A), b1 = 4A*(y-1)+1 = 1-4A*(1-y)
    # y<=1.
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
    if rad <= b1:  # extra safety
        return 0

    return (convert(rad - b1, uint256) * A_PRECISION) // (8 * A_raw)


@internal
@pure
def _p_from_y(A_raw: uint256, y: uint256) -> uint256:
    # p(y) = -dx/dy from implicit differentiation:
    #   p(y) = (4A + 1/(4*x*y^2)) / (4A + 1/(4*x^2*y))
    #        = (4A*x + 1/(4*y^2)) / (4A*x + 1/(4*x*y))
    x: uint256 = self._x_from_y(A_raw, y)
    if x == 0:
        return max_value(uint256)

    term4AP: uint256 = (4 * A_raw * x) // A_PRECISION
    return unsafe_div(
        (term4AP + unsafe_div(WAD3, 4 * y * y)) * WAD,
        term4AP + unsafe_div(WAD3, 4 * x * y),
    )


@internal
@pure
def _y_from_secant(A_raw: uint256, p: uint256) -> uint256:
    # Solve g(y)=0 where:
    #   g(y) = p(y) - p_target
    #   p_target = p
    assert p >= WAD
    lo: uint256 = WAD // 10**5  # y for p ~ 5000 and A=100_000
    hi: uint256 = WAD // 2 + 1  # y for p = 1

    plo: uint256 = self._p_from_y(A_raw, lo)
    phi: uint256 = self._p_from_y(A_raw, hi)

    # Warmup bisection by sign of g(mid):
    #   p(y) > p_target => g(y)>0 => move lo up
    #   p(y) < p_target => g(y)<0 => move hi down
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
    y0: int256 = convert(lo, int256)
    g0: int256 = convert(plo, int256) - p_i
    y1: int256 = convert(hi, int256)
    g1: int256 = convert(phi, int256) - p_i
    # g0 = g(y0), g1 = g(y1)

    if convert(abs(g0), uint256) <= tol_abs:
        return lo
    if convert(abs(g1), uint256) <= tol_abs:
        return hi

    # Secant step on g(y):
    #   y2 = y1 - g1 * (y1 - y0) / (g1 - g0)
    # Then safeguard y2 to stay inside (lo, hi) using mid point as fallback.
    for _: uint256 in range(SECANT_STEPS):
        y2: uint256 = unsafe_div(unsafe_add(lo, hi), 2)
        dg: int256 = g1 - g0

        if dg != 0:
            new_y2: int256 = y1 - unsafe_div(g1 * (y1 - y0), dg)
            if convert(lo, int256) < new_y2 and new_y2 < convert(hi, int256):
                # boundaries are positive, hence safe to convert
                y2 = convert(new_y2, uint256)

        p2: uint256 = self._p_from_y(A_raw, y2)
        g2: int256 = convert(p2, int256) - p_i

        if p2 > p:
            lo = y2
            plo = p2
        else:
            hi = y2
            phi = p2

        y0 = y1
        g0 = g1
        y1 = convert(y2, int256)
        g1 = g2

        if convert(abs(g2), uint256) <= tol_abs or unsafe_sub(hi, lo) <= 1:
            break

    # Final endpoint selection picks the closer bracket edge, so if this path
    # exits by bracket width the endpoint price error is bounded by half-span:
    #   eps_p_abs <= (p(lo) - p(hi)) / 2
    # If loop exits by tolerance:
    #   eps_p_abs <= p_target / PRICE_TOL_REL
    # Combined conservative bound:
    #   eps_p_abs <= max(p_target / PRICE_TOL_REL, (p(lo) - p(hi))/2)
    #
    # Value error from price error (mean-value bound, V'(y)=p_target-p(y)):
    #   eps_V <= eps_p_abs * |y_hat - y*| / WAD
    # and when hi-lo<=1:
    #   eps_V <= eps_p_abs / WAD
    if unsafe_add(plo, phi) >= 2 * p:  # using plo >= phi
        return hi
    return lo


@internal
@pure
def _get_x_y(A_raw: uint256, p: uint256) -> (uint256, uint256):
    assert A_raw > 0
    assert A_raw <= MAX_A_RAW
    assert p != 0

    # For p < 1 solve reciprocal branch and map back by symmetry.
    if p < WAD:
        p_inv: uint256 = unsafe_div(WAD2 + p // 2, p)
        y_inv: uint256 = self._y_from_secant(A_raw, p_inv)
        x_inv: uint256 = self._x_from_y(A_raw, y_inv)
        return y_inv, x_inv

    y: uint256 = self._y_from_secant(A_raw, p)
    x: uint256 = self._x_from_y(A_raw, y)
    return x, y


@internal
@pure
def _portfolio_value_secant(A_raw: uint256, p: uint256) -> uint256:
    x: uint256 = 0
    y: uint256 = 0
    x, y = self._get_x_y(A_raw, p)
    return x + p * y // WAD


@external
@pure
def portfolio_value(_A_raw: uint256, _p: uint256) -> uint256:
    return self._portfolio_value_secant(_A_raw, _p)
