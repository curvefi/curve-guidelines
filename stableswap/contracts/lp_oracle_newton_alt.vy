# pragma version 0.4.3
# pragma optimize gas

# =============================================================================
# Alternative LP-oracle implementation for StableSwap (n=2, D=1).
#
# Same problem, same answer as `lp_oracle_2.vy`, but solved with:
#
#   1. A closed-form initial guess from the high-A asymptotic.
#      In the A -> infinity regime with p > 1, the curve hugs x + y = 1, so
#      x ~ 1 and the marginal-price condition
#           16 A_eff (p-1) x^2 y^2 = x - p y
#      reduces, for y << 1, to
#           16 A_eff (p-1) y^2 = 1
#      giving the leading-order estimate
#           y_0 = 1 / (4 * sqrt(A_eff * (p - 1))).
#      For Curve's typical A_eff in [10, 10_000] this is already accurate to
#      a few percent on the dominant branch.
#
#   2. Newton's method on g(y) = p(y) - p_target = 0.
#      Quadratic convergence: a 1%-accurate seed lands below 10^-12 in
#      2 iterations.
#
#   3. Closed-form derivative p'(y) via implicit differentiation:
#           p'(y) = -2 (x^2 - p x y + p^2 y^2) / (x y^2 (16 A_eff x^2 y + 1)).
#      No bisection, no inner Newton, no sqrt in the derivative itself.
#
#   4. A bracket [lo, hi] is maintained alongside Newton. If a Newton step
#      would leave the bracket (overshoot near a near-stationary point), we
#      fall back to a bisection midpoint. This makes the iteration globally
#      convergent without sacrificing local quadratic speed.
#
# Per-call cost: 1 sqrt for the initial guess + ~2 inner iterations
# (each: 1 sqrt + arithmetic). Typically ~3x fewer iterations than the
# bisection in `lp_oracle_2.vy`.
#
# Geometric reading: by the Envelope Theorem, dV/dp = y* at every p. So
# y is the natural variable to refine, and Newton on y uses the local
# slope of p(y) -- which is exactly the local curvature of the invariant.
# =============================================================================

WAD: constant(uint256) = 10**18
WAD2: constant(uint256) = WAD * WAD
WAD3: constant(uint256) = WAD2 * WAD

A_PRECISION: constant(uint256) = 10**4
MAX_A: constant(uint256) = 100_000
MAX_A_RAW: constant(uint256) = MAX_A * A_PRECISION

MAX_NEWTON_ITERS: constant(uint256) = 64  # high cap; typically only 2-3 Newton
                                          # steps fire. The remaining budget is
                                          # consumed only when the bracket
                                          # fallback engages on extreme inputs
                                          # (very low A + extreme p), where the
                                          # loop reverts to pure bisection.
PRICE_TOL_REL: constant(uint256) = 10**6  # 0.01 bps

# Safe bounds on p and x for Newton's derivative formula. The derivative
# expression has terms like p*p*y*y and 16*A_raw*x*x*y; if p or x escape
# the unit-square regime (only happens at very low A with extreme p),
# those products overflow uint256. We detect this and fall back to a
# bisection step.
#
# 10**20 = 100 * WAD covers any realistic LP-oracle input (p in [0.01, 100]
# of coin-0 units, x in similar range). Outside this range, the contract
# silently switches to bisection, so the call still succeeds — just at the
# original library's gas cost.
SAFE_P_MAX: constant(uint256) = 10**20
SAFE_X_MAX: constant(uint256) = 10**20


@internal
@pure
def _x_from_y(A_raw: uint256, y: uint256) -> uint256:
    # Positive root of 4A x^2 + (4A(y-1) + 1) x - 1/(4y) = 0.
    b1: int256 = convert(WAD, int256) - convert(4 * A_raw * (WAD - y) // A_PRECISION, int256)
    abs_b1: uint256 = convert(abs(b1), uint256)
    term: uint256 = unsafe_div(4 * A_raw * WAD3, A_PRECISION * y)
    rad: int256 = convert(isqrt(abs_b1**2 + term), int256)
    if rad <= b1:
        return 0
    return (convert(rad - b1, uint256) * A_PRECISION) // (8 * A_raw)


@internal
@pure
def _p_from_y(A_raw: uint256, x: uint256, y: uint256) -> uint256:
    # p(y) = (4A x + 1/(4 y^2)) / (4A x + 1/(4 x y)),  WAD-scaled.
    if x == 0:
        return max_value(uint256)
    term4A: uint256 = (4 * A_raw * x) // A_PRECISION
    return unsafe_div(
        (term4A + unsafe_div(WAD3, 4 * y * y)) * WAD,
        term4A + unsafe_div(WAD3, 4 * x * y),
    )


@internal
@pure
def _p_prime_abs(A_raw: uint256, x: uint256, y: uint256, p: uint256) -> uint256:
    # Returns |dp/dy| at (x, y), WAD-scaled.
    #
    # Derivation: implicit diff of F(x, y) = 0 with p = F_y / F_x.
    # Using F_y/F_x = p (so b = p a) and the Hessian of F:
    #
    #   p'(y) = -2 (x^2 - p x y + p^2 y^2) / (x y^2 (16 A_eff x^2 y + 1))
    #
    # The numerator is x^2 - p x y + p^2 y^2 = (x - p y / 2)^2 + 3 p^2 y^2 / 4,
    # always positive, so we drop the sign and return the magnitude.

    # Numerator N at WAD^2 scaling (i.e. N_real * WAD^2 = WAD^2-units integer)
    #   x*x is already WAD^2-scaled.
    #   p*x*y is WAD^3-scaled; divide by WAD.
    #   p^2*y^2 is WAD^4-scaled; divide by WAD^2.
    xx: uint256 = x * x
    pxy: uint256 = unsafe_div(p * x * y, WAD)
    p2y2: uint256 = unsafe_div(p * p * y * y, WAD2)
    if xx + p2y2 <= pxy:
        # Analytically impossible (AM-GM gives xx + p2y2 >= 2*pxy >= pxy).
        # Guard against integer-rounding pathology.
        return 0
    n_w2: uint256 = xx + p2y2 - pxy

    # Denominator at WAD^2 scaling: D = x*y^2 * (16 A_eff x^2 y + 1).
    #   bracket = 16 A_eff x^2 y + 1, dimensionless
    #          = 16 A_raw x^2 y / (A_PRECISION * WAD^3) + 1, integer.
    #   xy2_w2 = x*y^2 in WAD^2-scaled units = (x*y*y) / WAD.
    bracket: uint256 = (16 * A_raw * x * x * y) // (A_PRECISION * WAD3) + 1
    xy2_w2: uint256 = unsafe_div(x * y * y, WAD)
    d_w2: uint256 = xy2_w2 * bracket
    if d_w2 == 0:
        return 0

    # |p'(y)| * WAD = 2 * (n_w2 / WAD^2) * WAD / (d_w2 / WAD^2)
    #              = 2 * n_w2 * WAD / d_w2.
    return unsafe_div(2 * n_w2 * WAD, d_w2)


@internal
@pure
def _y_initial_guess(A_raw: uint256, p: uint256) -> uint256:
    # High-A asymptotic initial guess on the p >= WAD branch.
    #
    # y_0_real = 1 / (4 * sqrt(A_eff * (p_real - 1)))
    #          = sqrt(A_PRECISION / (16 * A_raw * (p_real - 1)))
    # In WAD-scaled integer:
    #   y_0_int = y_0_real * WAD = sqrt(WAD^3 * A_PRECISION / (16 * A_raw * diff))
    # where diff = p - WAD (WAD-scaled).
    #
    # Capped at WAD/2 when the asymptotic gives a value outside the
    # bisection branch (happens for p extremely close to WAD).
    if p <= WAD:
        return WAD // 2
    diff: uint256 = p - WAD
    denom: uint256 = 16 * A_raw * diff
    if denom == 0:
        return WAD // 2
    y0_sq: uint256 = unsafe_div(WAD3 * A_PRECISION, denom)
    y0: uint256 = isqrt(y0_sq)
    if y0 >= WAD // 2:
        return WAD // 2
    if y0 == 0:
        return 1
    return y0


@internal
@pure
def _y_newton(A_raw: uint256, p: uint256) -> uint256:
    # Bracketed Newton on g(y) = p(y) - p_target.
    #
    # The bracket [lo, hi] is maintained the same way bisection would:
    # we know p is monotonically decreasing on (0, WAD/2], so a Newton step
    # that lands outside [lo, hi] is replaced by a bisection midpoint.
    # This makes the iteration globally convergent.
    assert p >= WAD

    lo: uint256 = 1
    hi: uint256 = WAD // 2 + 1
    y: uint256 = self._y_initial_guess(A_raw, p)
    # Ensure y starts strictly inside the bracket.
    if y <= lo:
        y = lo + 1
    if y >= hi:
        y = hi - 1

    tol_abs: uint256 = unsafe_div(p, PRICE_TOL_REL)

    for _: uint256 in range(MAX_NEWTON_ITERS):
        x: uint256 = self._x_from_y(A_raw, y)
        if x == 0:
            # Numerical edge; fall back to mid-bracket and retry.
            y = unsafe_div(lo + hi, 2)
            continue

        pm: uint256 = self._p_from_y(A_raw, x, y)

        # Convergence check (price tolerance).
        if pm > p:
            if unsafe_sub(pm, p) <= tol_abs:
                return y
            lo = y  # pm above target -> y too small
        else:
            if unsafe_sub(p, pm) <= tol_abs:
                return y
            hi = y  # pm below target -> y too large

        if unsafe_sub(hi, lo) <= 1:
            return hi

        # If the curve point is in the extreme regime (very low A + y near
        # boundary), the derivative formula's p*p*y*y and 16*A_raw*x*x*y
        # terms can overflow uint256. Detect this and use a bisection step;
        # the bracket guarantees convergence regardless.
        y_new: uint256 = 0
        if pm > SAFE_P_MAX or x > SAFE_X_MAX:
            y_new = unsafe_div(lo + hi, 2)
        else:
            # Newton step y_new = y - g(y)/g'(y), with g'(y) = -|p'(y)| < 0.
            ppy: uint256 = self._p_prime_abs(A_raw, x, y, pm)
            if ppy == 0:
                y_new = unsafe_div(lo + hi, 2)
            elif pm > p:
                # Need to move y up (toward smaller p).
                delta: uint256 = unsafe_div((pm - p) * WAD, ppy)
                y_new = y + delta
            else:
                # Need to move y down (toward larger p).
                delta: uint256 = unsafe_div((p - pm) * WAD, ppy)
                if delta >= y:
                    # Newton overshoots into negative; bisect instead.
                    y_new = unsafe_div(lo + hi, 2)
                else:
                    y_new = y - delta

        # Bisection fallback if Newton overshoots the bracket.
        if y_new <= lo or y_new >= hi:
            y_new = unsafe_div(lo + hi, 2)
        y = y_new

    return y


@internal
@pure
def _get_x_y(A_raw: uint256, p: uint256) -> (uint256, uint256):
    assert A_raw > 0
    assert A_raw <= MAX_A_RAW
    assert p != 0

    if p < WAD:
        p_inv: uint256 = unsafe_div(WAD2 + p // 2, p)
        y_inv: uint256 = self._y_newton(A_raw, p_inv)
        x_inv: uint256 = self._x_from_y(A_raw, y_inv)
        return y_inv, x_inv

    y: uint256 = self._y_newton(A_raw, p)
    x: uint256 = self._x_from_y(A_raw, y)
    return x, y


@internal
@pure
def _portfolio_value(A_raw: uint256, p: uint256) -> uint256:
    x: uint256 = 0
    y: uint256 = 0
    x, y = self._get_x_y(A_raw, p)
    return x + p * y // WAD


@external
@pure
def portfolio_value(_A_raw: uint256, _p: uint256) -> uint256:
    return self._portfolio_value(_A_raw, _p)
