import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Simulations
    Python simulations for LP price formulas.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import pandas as pd
    import math
    from copy import copy
    from stableswap.simulation import StableSwap
    return StableSwap, copy, math, mo, px


@app.cell
def _(mo):
    n = mo.ui.number(start=2, stop=8, step=1, value=2, label="n")
    A = mo.ui.number(start=1, stop=100_000, step=1, value=200, label="A")

    D = 1_000_000 * 10 ** 18
    return A, D, n


@app.cell
def _(A, mo):
    mo.md(f"""{A}  \n""")
    return


@app.cell
def _(A, D, StableSwap, copy, mo, n, portfolio_value_secant, px):
    # Assuming p = price_oracle
    MAX_P_FACTOR = 2.5  # Factor to limit price changes
    dx = D // 10_000
    dy = 0

    pool = StableSwap(A.value * n.value ** (n.value - 1), D, n.value, fee=0)
    p = pool.get_p()
    while p[0] / 10 ** 18 < MAX_P_FACTOR and 10 ** 18 / p[0] < MAX_P_FACTOR:
        dy = pool.exchange(0, 1, dx)
        p = pool.get_p()
    pool.exchange(1, 0, dy)  # go 1 step back
    p = pool.get_p()

    points = []

    while p[0] / 10 ** 18 < MAX_P_FACTOR and 10 ** 18 / p[0] < MAX_P_FACTOR:
        pool.exchange(1, 0, dx)
        p = pool.get_p()
        points.append({
            "balances": copy(pool.x),
            "prices": [10 ** 18] + copy(p),
            "vp": pool.get_virtual_price(),
            "total_supply": copy(pool.tokens),
        })

    def real_lp_price(point):
        return sum([b * p // 10 ** 18 for p, b in zip(point["prices"], point["balances"])]) * 10**18 // point["total_supply"] / 10**18

    def simplified_lp_price(point):
        min_p = min(point["prices"])
        return min_p * point["vp"] // 10 ** 18 / 10 ** 18

    def ai_analytic_lp_price(point):
        assert n.value == 2
        return portfolio_value_secant(A.value, int(point["prices"][1])) * point["vp"] // 10**18 / 10**18

    plot = mo.ui.plotly(
        px.line(
            [{
                "price": point["prices"][1] / 10 ** 18,
                "Real": real_lp_price(point),
                "Simplified": simplified_lp_price(point),
                "ai_formula": ai_analytic_lp_price(point),
            } for point in points],
            x="price",
            y=["Real", "Simplified", "ai_formula"],
            title="LP Price",
        )
    )

    plot
    return


@app.cell
def _(math):
    Q  = 10**18
    Q2 = Q * Q
    Q3 = Q2 * Q
    Q4 = Q2 * Q2

    U256_MAX = 2**256 - 1

    # Строгая безопасная граница для A, чтобы radicand не мог переполнить uint256 в худшем случае.
    A_SAFE_MAX = 84945683565681204819  # ~8.49e19

    def _isqrt(n: int) -> int:
        return int(math.isqrt(n))

    def _idiv0(a: int, b: int) -> int:
        """EVM-like signed division: trunc toward zero."""
        if b == 0:
            raise ZeroDivisionError
        sign = -1 if ((a < 0) ^ (b < 0)) else 1
        return sign * (abs(a) // abs(b))

    def _x_from_s(A: int, sQ: int) -> int:
        """
        x(s) in Q fixed-point, D=1.
        x = (-b1 + sqrt(b1^2 + 4A/s)) / (8A)
        b1 = 4A(s-1)+1
        All computed in fixed-point:
          b1Q = b1*Q
          radQ2 = (b1^2 + 4A/s) * Q^2 = b1Q^2 + (4A*Q^3)//sQ
        """
        b1Q = 4 * A * (sQ - Q) + Q                # signed
        b1sq = b1Q * b1Q                           # ok with A<=A_SAFE_MAX
        term = (4 * A * Q3) // sQ                  # floor(4A*Q^3/s)
        radQ2 = b1sq + term
        sqrtQ = _isqrt(radQ2)
        num = -b1Q + sqrtQ                         # signed
        if num <= 0:
            return 0
        xQ = num // (8 * A)
        return xQ if xQ > 0 else 0

    def _p_from_s(A: int, sQ: int) -> int:
        """
        p(s) = -dx/dy in Q fixed-point, D=1.
        p = (4A + 1/(4 x s^2)) / (4A + 1/(4 x^2 s))
        inv1Q = Q^4 / (4*xQ*sQ^2)
        inv2Q = Q^4 / (4*xQ^2*sQ)
        pQ = (num/den) * Q
        """
        xQ = _x_from_s(A, sQ)
        if xQ == 0:
            return U256_MAX

        term4AQ = 4 * A * Q
        s2 = sQ * sQ
        x2 = xQ * xQ

        inv1Q = Q4 // (4 * xQ * s2)
        inv2Q = Q4 // (4 * x2 * sQ)

        numQ = term4AQ + inv1Q
        denQ = term4AQ + inv2Q

        # Если хочешь железно избегать mul overflow в EVM — заменишь на q/r-трюк.
        return (numQ * Q) // denQ

    def _value_from_s(A: int, p: int, sQ: int) -> int:
        """V = x + p*y, y=s (D=1). All in Q."""
        xQ = _x_from_s(A, sQ)
        return xQ + (p * sQ) // Q

    def portfolio_value_bisection(A: int, p: int, *, iters: int = 80) -> int:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")
        if p < 0:
            p = -p

        lo, hi = 1, Q - 1

        for _ in range(iters):
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            # p(s) decreases with s
            if pm > p:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1:
                break

        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)
        sQ = lo if abs(int(plo) - int(p)) <= abs(int(phi) - int(p)) else hi

        return _value_from_s(A, p, sQ)

    def portfolio_value_secant(A: int, p: int, *, bisect_steps: int = 10, secant_steps: int = 14) -> int:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")
        if p < 0:
            p = -p

        lo, hi = 1, Q - 1
        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)

        # немного поджать скобку (ускоряет секущую)
        for _ in range(bisect_steps):
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            if pm > p:
                lo, plo = mid, pm
            else:
                hi, phi = mid, pm
            if hi - lo <= 1:
                break

        # g(s) = p(s) - p
        s0, g0 = lo, int(plo) - int(p)
        s1, g1 = hi, int(phi) - int(p)

        # если уже достаточно близко
        if abs(g0) <= 2:
            sQ = s0
            return _value_from_s(A, p, sQ)
        if abs(g1) <= 2:
            sQ = s1
            return _value_from_s(A, p, sQ)

        sQ = (lo + hi) // 2

        for _ in range(secant_steps):
            dg = g1 - g0
            if dg == 0:
                s2 = (lo + hi) // 2
            else:
                s2 = s1 - _idiv0(g1 * (s1 - s0), dg)

            if not (lo < s2 < hi):
                s2 = (lo + hi) // 2

            p2 = _p_from_s(A, s2)
            g2 = int(p2) - int(p)

            # shrink bracket
            if p2 > p:
                lo, plo = s2, p2
            else:
                hi, phi = s2, p2

            s0, g0 = s1, g1
            s1, g1 = s2, g2
            sQ = s2

            if hi - lo <= 1 or abs(g2) <= 2:
                break

        # best endpoint
        sQ = lo if abs(int(plo) - int(p)) <= abs(int(phi) - int(p)) else hi
        return _value_from_s(A, p, sQ)

    def portfolio_value_newton(A: int, p: int, *, bisect_steps: int = 10, newton_steps: int = 12) -> int:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")
        if p < 0:
            p = -p

        # bracket
        lo, hi = 1, Q - 1
        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)

        # tighten bracket a bit
        for _ in range(bisect_steps):
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            if pm > p:
                lo, plo = mid, pm
            else:
                hi, phi = mid, pm
            if hi - lo <= 1:
                break

        # стартовая точка внутри
        s = (lo + hi) // 2
        ps = _p_from_s(A, s)
        gs = int(ps) - int(p)

        # предыдущая точка для наклона
        s_prev = lo
        g_prev = int(plo) - int(p)

        for _ in range(newton_steps):
            if abs(gs) <= 2 or hi - lo <= 1:
                break

            dg = gs - g_prev
            ds = s - s_prev

            if dg == 0:
                s_new = (lo + hi) // 2
            else:
                # Newton-like step: s - g * ds/dg
                s_new = s - _idiv0(gs * ds, dg)

            # safeguard
            if not (lo < s_new < hi):
                s_new = (lo + hi) // 2

            p_new = _p_from_s(A, s_new)
            g_new = int(p_new) - int(p)

            # update bracket by monotonicity
            if p_new > p:
                lo, plo = s_new, p_new
            else:
                hi, phi = s_new, p_new

            s_prev, g_prev = s, gs
            s, gs = s_new, g_new

        # best endpoint
        sQ = lo if abs(int(plo) - int(p)) <= abs(int(phi) - int(p)) else hi
        return _value_from_s(A, p, sQ)
    return (portfolio_value_secant,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Comparison
    Comparison of Vyper implementations for LP price oracles.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
