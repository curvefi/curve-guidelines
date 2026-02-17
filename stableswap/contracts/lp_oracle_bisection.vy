# pragma version 0.4.3


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
#   s := y (because D=1 normalization), with sP = floor(s * WAD)
#   xP := floor(x * WAD)
#   g(s) := p(s) - p_target
#
# 1) Invariant and x(s)
#   For n=2, D=1:
#     4*A_eff*(x + y) + 1 = 4*A_eff + 1/(4*x*y)
#   Rearranged:
#     4*A_eff*x^2 + (4*A_eff*(y-1) + 1)*x - 1/(4*y) = 0
#   With y=s and b1 = 4*A_eff*(s-1)+1:
#     x(s) = (-b1 + sqrt(b1^2 + 4*A_eff/s)) / (8*A_eff)
#
# 2) Marginal price p(s)
#   From implicit differentiation of F(x,y)=0:
#     p(s) = -dx/dy
#          = (4*A_eff + 1/(4*x*s^2)) / (4*A_eff + 1/(4*x^2*s))
#
# 3) Value at fixed s
#     V(s) = x(s) + p_target * s
#
# 4) Root equation solved by numeric method
#     g(s) = p(s) - p_target = 0
#   On the relevant branch p(s) is monotone decreasing in s, so bracketing works.
#
# 5) Fixed-point mapping used by this implementation
#   b1P   = WAD + (4*A_raw*(sP - WAD))/A_PRECISION
#   radP2 = b1P^2 + (4*A_raw*WAD^3)/(A_PRECISION*sP)
#   xP    = ((-b1P + sqrt(radP2)) * A_PRECISION) / (8*A_raw)
#   term4 = (4*A_raw*WAD)/A_PRECISION
#   pP    = ((term4 + WAD^4/(4*xP*sP^2)) * WAD) / (term4 + WAD^4/(4*xP^2*sP))
#
# 6) Symmetry for p_target < 1
#   Solve reciprocal branch at p_inv ~= WAD^2 / p_target, then map back:
#     V(p_target) = p_target * V(p_inv)
#     (x, y) at p_target is swap of (x, y) at p_inv.
# =============================================================================
WAD: constant(uint256) = 10**18
WAD3: constant(uint256) = WAD * WAD * WAD
WAD4: constant(uint256) = WAD * WAD * WAD * WAD
A_PRECISION: constant(uint256) = 100
MAX_A: constant(uint256) = 100_000
MAX_A_PRECISION: constant(uint256) = 10_000
MAX_A_RAW: constant(uint256) = MAX_A * MAX_A_PRECISION
BISECTION_ITERS: constant(uint256) = 80


@internal
@pure
def _abs_diff(a: uint256, b: uint256) -> uint256:
    if a >= b:
        return unsafe_sub(a, b)
    return unsafe_sub(b, a)


@internal
@pure
def _inv_price(p: uint256) -> uint256:
    assert p != 0
    return unsafe_div(WAD * WAD + p // 2, p)


@internal
@pure
def _assert_inputs(A_raw: uint256, p: uint256):
    assert A_PRECISION <= MAX_A_PRECISION
    assert A_raw > 0
    assert A_raw <= MAX_A_RAW
    assert p != 0


@internal
@pure
def _x_from_s(A_raw: uint256, sP: uint256) -> uint256:
    # x(s) from invariant quadratic:
    #   4A*x^2 + (4A*(s-1)+1)*x - 1/(4s) = 0
    #   x(s) = (-b1 + sqrt(b1^2 + 4A/s)) / (8A), b1 = 4A*(s-1)+1
    # Here sP is WAD-scaled s, and all arithmetic stays in fixed-point.
    delta: int256 = convert(sP, int256) - convert(WAD, int256)

    b1_term_abs: uint256 = (4 * A_raw * convert(abs(delta), uint256)) // A_PRECISION
    b1P: int256 = convert(WAD, int256)
    if delta >= 0:
        b1P += convert(b1_term_abs, int256)
    else:
        b1P -= convert(b1_term_abs, int256)

    b1sq: uint256 = convert(abs(b1P), uint256) * convert(abs(b1P), uint256)
    term: uint256 = unsafe_div(4 * A_raw * WAD3, A_PRECISION * sP)
    radP2: uint256 = b1sq + term
    sqrtP: uint256 = isqrt(radP2)

    num: int256 = -b1P + convert(sqrtP, int256)
    if num <= 0:
        return 0

    return convert((num * convert(A_PRECISION, int256)) // convert(8 * A_raw, int256), uint256)


@internal
@pure
def _p_from_s(A_raw: uint256, sP: uint256) -> uint256:
    # p(s) = -dx/dy from implicit differentiation:
    #   p(s) = (4A + 1/(4*x*s^2)) / (4A + 1/(4*x^2*s))
    # with x = x(s).
    xP: uint256 = self._x_from_s(A_raw, sP)
    if xP == 0:
        return max_value(uint256)

    term4AP: uint256 = (4 * A_raw * WAD) // A_PRECISION
    return unsafe_div(
        (term4AP + unsafe_div(WAD4, 4 * xP * sP * sP)) * WAD,
        term4AP + unsafe_div(WAD4, 4 * xP * xP * sP),
    )


@internal
@pure
def _value_from_s(A_raw: uint256, p: uint256, sP: uint256) -> uint256:
    # Portfolio value at y=s:
    #   V(s) = x(s) + p_target * s
    return self._x_from_s(A_raw, sP) + (p * sP) // WAD


@internal
@pure
def _s_from_bisection(A_raw: uint256, p: uint256) -> uint256:
    # Root finding for:
    #   g(s) = p(s) - p_target
    # Target equation:
    #   g(s*) = 0  <=>  p(s*) = p_target
    # Monotonicity on bracket [lo, hi]:
    #   g(mid) > 0 (pm > p) => s too small => move lo = mid
    #   g(mid) < 0 (pm < p) => s too large => move hi = mid
    lo: uint256 = 1
    hi: uint256 = WAD - 1

    for _: uint256 in range(BISECTION_ITERS):
        mid: uint256 = unsafe_div(unsafe_add(lo, hi), 2)
        pm: uint256 = self._p_from_s(A_raw, mid)

        # p(s) decreases with s
        if pm > p:
            lo = mid
        else:
            hi = mid

        if unsafe_sub(hi, lo) <= 1:
            break

    plo: uint256 = self._p_from_s(A_raw, lo)
    phi: uint256 = self._p_from_s(A_raw, hi)

    if self._abs_diff(phi, p) < self._abs_diff(plo, p):
        return hi
    return lo


@internal
@pure
def _portfolio_value_bisection(A_raw: uint256, p: uint256) -> uint256:
    self._assert_inputs(A_raw, p)
    # For p < 1 solve reciprocal branch and map back by symmetry.
    if p < WAD:
        p_inv: uint256 = self._inv_price(p)
        return (p * self._value_from_s(A_raw, p_inv, self._s_from_bisection(A_raw, p_inv))) // WAD

    sP: uint256 = self._s_from_bisection(A_raw, p)
    return self._value_from_s(A_raw, p, sP)


@external
@pure
def portfolio_value(_A_raw: uint256, _p: uint256) -> uint256:
    return self._portfolio_value_bisection(_A_raw, _p)


@internal
@pure
def _get_x_y(_A_raw: uint256, _p: uint256) -> (uint256, uint256):
    if _p < WAD:
        p_inv: uint256 = self._inv_price(_p)
        y_inv: uint256 = self._s_from_bisection(_A_raw, p_inv)
        x_inv: uint256 = self._x_from_s(_A_raw, y_inv)
        return y_inv, x_inv

    y: uint256 = self._s_from_bisection(_A_raw, _p)
    x: uint256 = self._x_from_s(_A_raw, y)
    return x, y
