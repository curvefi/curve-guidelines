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
    return StableSwap, copy, math, mo, np, pd, px


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
def _(
    A,
    D,
    StableSwap,
    copy,
    mo,
    n,
    portfolio_value_bisection,
    portfolio_value_newton,
    portfolio_value_secant,
    px,
):
    # Assuming p = price_oracle
    MAX_P_FACTOR = 10  # Factor to limit price changes
    dx = D // 10_000
    dy = 0

    pool = StableSwap(A.value, D, n.value, fee=0)
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

    def converge_lp_price(point, method):
        assert n.value == 2
        return method(A.value, int(point["prices"][1])) * point["vp"] // 10**18 / 10**18

    plot = mo.ui.plotly(
        px.line(
            [{
                "price": point["prices"][1] / 10 ** 18,
                "Real": real_lp_price(point),
                "Simplified": simplified_lp_price(point),
                "bisection": converge_lp_price(point, portfolio_value_bisection),
                "secant": converge_lp_price(point, portfolio_value_secant),
                "newton": converge_lp_price(point, portfolio_value_newton),
            } for point in points],
            x="price",
            y=["Real", "Simplified", "bisection", "secant", "newton"],
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

    def _inv_price_Q(p: int) -> int:
        # round-to-nearest reciprocal: p_inv = round(Q^2 / p)
        # (only integer ops)
        return (Q2 + p // 2) // p

    def _mul_div_Q(a: int, b: int) -> int:
        # floor(a*b/Q) without any extra assumptions
        return (a * b) // Q


    def portfolio_value_bisection(A: int, p: int, *, iters: int = 80) -> int:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")

        # symmetry: solve only for p >= 1, map back for p < 1
        if p < Q:
            p_inv = _inv_price_Q(p)
            V_inv = portfolio_value_bisection(A, p_inv, iters=iters)   # value in "y units"
            return _mul_div_Q(p, V_inv)                                # back to "x units"

        lo, hi = 1, Q - 1
        for _ in range(iters):
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
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


    def portfolio_value_secant(A: int, p: int, *, bisect_steps: int = 10, secant_steps: int = 256) -> int:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")

        if p < Q:
            p_inv = _inv_price_Q(p)
            V_inv = portfolio_value_secant(A, p_inv, bisect_steps=bisect_steps, secant_steps=secant_steps)
            return _mul_div_Q(p, V_inv)

        lo, hi = 1, Q - 1
        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)

        for _ in range(bisect_steps):
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            if pm > p:
                lo, plo = mid, pm
            else:
                hi, phi = mid, pm
            if hi - lo <= 1:
                break

        s0, g0 = lo, int(plo) - int(p)
        s1, g1 = hi, int(phi) - int(p)

        if abs(g0) <= 2:
            return _value_from_s(A, p, s0)
        if abs(g1) <= 2:
            return _value_from_s(A, p, s1)

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

            if p2 > p:
                lo, plo = s2, p2
            else:
                hi, phi = s2, p2

            s0, g0 = s1, g1
            s1, g1 = s2, g2
            sQ = s2

            if hi - lo <= 1 or abs(g2) <= 2:
                break

        sQ = lo if abs(int(plo) - int(p)) <= abs(int(phi) - int(p)) else hi
        return _value_from_s(A, p, sQ)


    def portfolio_value_newton(A: int, p: int, *, bisect_steps: int = 10, newton_steps: int = 256) -> int:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")

        if p < Q:
            p_inv = _inv_price_Q(p)
            V_inv = portfolio_value_newton(A, p_inv, bisect_steps=bisect_steps, newton_steps=newton_steps)
            return _mul_div_Q(p, V_inv)

        lo, hi = 1, Q - 1
        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)

        for _ in range(bisect_steps):
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            if pm > p:
                lo, plo = mid, pm
            else:
                hi, phi = mid, pm
            if hi - lo <= 1:
                break

        s = (lo + hi) // 2
        ps = _p_from_s(A, s)
        gs = int(ps) - int(p)

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
                s_new = s - _idiv0(gs * ds, dg)

            if not (lo < s_new < hi):
                s_new = (lo + hi) // 2

            p_new = _p_from_s(A, s_new)
            g_new = int(p_new) - int(p)

            if p_new > p:
                lo, plo = s_new, p_new
            else:
                hi, phi = s_new, p_new

            s_prev, g_prev = s, gs
            s, gs = s_new, g_new

        sQ = lo if abs(int(plo) - int(p)) <= abs(int(phi) - int(p)) else hi
        return _value_from_s(A, p, sQ)
    return (
        portfolio_value_bisection,
        portfolio_value_newton,
        portfolio_value_secant,
    )


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
def _(mo):
    comparison_samples = mo.ui.number(start=10, stop=10_000, step=10, value=200, label="samples")
    comparison_seed = mo.ui.number(start=0, stop=1_000_000, step=1, value=42, label="seed")
    comparison_a_min = mo.ui.number(start=1.0, stop=100_000, step=0.0001, value=9, label="A min")
    comparison_a_max = mo.ui.number(start=1.0, stop=100_000, step=1, value=10_000, label="A max")
    comparison_p_min = mo.ui.number(start=0.01, stop=100.0, step=0.01, value=0.1, label="p min (x WAD)")
    comparison_p_max = mo.ui.number(start=0.1, stop=10.0, step=0.01, value=8.0, label="p max (x WAD)")
    mo.vstack(
        [
            mo.hstack([comparison_samples, comparison_seed]),
            mo.hstack([comparison_a_min, comparison_p_min]),
            mo.hstack([comparison_a_max, comparison_p_max]),
        ]
    )
    return (
        comparison_a_max,
        comparison_a_min,
        comparison_p_max,
        comparison_p_min,
        comparison_samples,
        comparison_seed,
    )


@app.cell
def _(
    comparison_a_max,
    comparison_a_min,
    comparison_p_max,
    comparison_p_min,
    comparison_samples,
    comparison_seed,
    np,
    pd,
):
    import os

    # Keep titanoboa cache local when notebook is run in restricted environments.
    os.environ.setdefault("XDG_CACHE_HOME", ".cache")
    import boa

    WAD = 10**18
    A_PRECISION = 100

    def stats(oracle, params: dict, method: str):
        A_eff = int(params["A"])
        p_int = int(params["p_int"])
        A_raw_local = A_eff * A_PRECISION

        boa.env.reset_gas_used()
        x_int, y_int = oracle.internal._get_x_y(A_raw_local, p_int)
        gas_xy = boa.env.get_gas_used()

        # boa.env.reset_gas_used()
        # oracle_value = int(oracle.portfolio_value(A_raw_local, p_int))
        # gas_portfolio_value = int(boa.env.get_gas_used())

        # Invariant in D=1 coordinates:
        # (4A(x+y-1)+1) * 4xy = 1
        # Here x,y are WAD-scaled, so compare scaled sides:
        # ((4A(x+y-WAD)+WAD) * 4xy) ?= WAD^3
        lhs = (4 * A_eff * (x_int + y_int - WAD) + WAD) * 4 * x_int * y_int
        rhs = WAD * WAD * WAD
        inv_eq_abs_err_raw = abs(lhs - rhs)
        inv_eq_abs_err = inv_eq_abs_err_raw / rhs
        inv_eq_rel = inv_eq_abs_err_raw / rhs

        # Price convergence check: p_hat = -dx/dy computed from (x, y)
        term4a_wad = 4 * A_eff * WAD
        inv1 = (WAD ** 4) // (4 * x_int * y_int * y_int)
        inv2 = (WAD ** 4) // (4 * x_int * x_int * y_int)
        p_hat = ((term4a_wad + inv1) * WAD) // (term4a_wad + inv2)
        price_abs_err_raw = abs(p_hat - p_int)
        price_abs_err = price_abs_err_raw / WAD
        price_rel = price_abs_err_raw / p_int

        return {
            "method": method,
            "x": x_int,
            "y": y_int,
            "value": x_int + (p_int * y_int) // WAD,
            "inv_eq_abs_err": inv_eq_abs_err,
            "inv_eq_rel": inv_eq_rel,
            "p_hat": p_hat,
            "price_abs_err": price_abs_err,
            "price_rel": price_rel,
            "gas_xy": gas_xy,
            # "gas_portfolio_value": gas_portfolio_value,
        }

    def storage_ref_gas_price_oracle(pool) -> int:
        pool.set(A_PRECISION, WAD)
        boa.env.reset_gas_used()
        _ = pool.price_oracle()
        return boa.env.get_gas_used()

    oracles = {
        "bisection": boa.load("stableswap/contracts/lp_oracle_bisection.vy"),
        "secant": boa.load("stableswap/contracts/lp_oracle_secant.vy"),
        "newton": boa.load("stableswap/contracts/lp_oracle_newton.vy"),
        "brent": boa.load("stableswap/contracts/lp_oracle_brent.vy"),
    }
    pool_ref = boa.load("stableswap/contracts/StableSwapMock.vy")
    boa.env.enable_gas_profiling()
    gas_storage_ref = storage_ref_gas_price_oracle(pool_ref)

    rng = np.random.default_rng(int(comparison_seed.value))
    a_values = rng.integers(
        int(comparison_a_min.value),
        int(comparison_a_max.value) + 1,
        size=int(comparison_samples.value),
    )
    p_values = rng.integers(
        int(float(comparison_p_min.value) * WAD),
        int(float(comparison_p_max.value) * WAD) + 1,
        size=int(comparison_samples.value),
        dtype=np.int64,
    )

    rows = []
    for A_eff, p_raw in zip(a_values, p_values):
        p_int = int(p_raw)
        for method, oracle in oracles.items():
            row = stats(
                oracle,
                {
                    "A": int(A_eff),
                    "p_int": p_int,
                },
                method=method,
            )
            row.update({"A": int(A_eff), "price": p_int / WAD, "p_int": p_int})
            rows.append(row)

    comparison_df = pd.DataFrame(rows)

    summary_df = (
        comparison_df.groupby("method", as_index=False)
        .agg(
            samples=("method", "size"),
            max_inv_eq_rel=("inv_eq_rel", "max"),
            mean_inv_eq_rel=("inv_eq_rel", "mean"),
            max_inv_eq_abs_err=("inv_eq_abs_err", "max"),
            max_price_rel=("price_rel", "max"),
            mean_price_rel=("price_rel", "mean"),
            max_price_abs_err=("price_abs_err", "max"),
            mean_gas=("gas_xy", "mean"),
            max_gas=("gas_xy", "max"),
        )
        .sort_values("method")
    )
    summary_display_df = summary_df.copy()
    for col in [
        "max_inv_eq_rel",
        "mean_inv_eq_rel",
        "max_inv_eq_abs_err",
        "max_price_rel",
        "mean_price_rel",
        "max_price_abs_err",
    ]:
        summary_display_df[col] = summary_display_df[col].map(lambda v: f"{v:.2e}")

    metrics_display_df = comparison_df[
        [
            "method",
            "A",
            "price",
            "x",
            "y",
            "inv_eq_abs_err",
            "inv_eq_rel",
            "price_abs_err",
            "price_rel",
            "gas_xy",
        ]
    ].copy()
    for col in ["inv_eq_abs_err", "inv_eq_rel", "price_abs_err", "price_rel"]:
        metrics_display_df[col] = metrics_display_df[col].map(lambda v: f"{v:.2e}")
    return (
        comparison_df,
        gas_storage_ref,
        metrics_display_df,
        summary_display_df,
    )


@app.cell
def _(
    comparison_df,
    gas_storage_ref,
    metrics_display_df,
    mo,
    px,
    summary_display_df,
):
    summary_md = mo.md(
        f"""**How to compare convergence**  
    `oracle.internal.xy(A, p)` returns `(x, y)` for each implementation.  
    Errors are measured by direct substitution into StableSwap formulas:  
    - invariant equality: `|(4A(x+y-1)+1)*4*x*y - 1|`  
    - price convergence (`-dx/dy`): `|p_hat(x,y) - p|`  
    - gas reference (single storage read `StableSwapMock.price_oracle()`): `{gas_storage_ref}`  
    """
    )

    err_fig = px.line(
        comparison_df.sort_values(["method", "price"]),
        x="price",
        y="inv_eq_rel",
        color="method",
        title="Invariant equality relative error by method",
    )
    err_fig.update_yaxes(tickformat=".2e")
    err_plot = mo.ui.plotly(err_fig)

    price_fig = px.line(
        comparison_df.sort_values(["method", "price"]),
        x="price",
        y="price_rel",
        color="method",
        title="Price convergence (-dx/dy) relative error by method",
    )
    price_fig.update_yaxes(tickformat=".2e")
    price_plot = mo.ui.plotly(price_fig)

    gas_plot = mo.ui.plotly(
        px.scatter(
            comparison_df,
            x="price",
            y="gas_xy",
            color="method",
            title="Gas vs price (xy), color=method",
        )
    )

    summary_table = mo.ui.table(summary_display_df)

    metrics_table = mo.ui.table(metrics_display_df)
    mo.vstack([summary_md, summary_table, err_plot, price_plot, gas_plot, metrics_table])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
