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
    from decimal import Decimal
    from stableswap.simulation import StableSwap
    return Decimal, StableSwap, copy, math, mo, np, pd, px


@app.cell
def _(mo):
    n = mo.ui.number(start=2, stop=8, step=1, value=2, label="n")
    A = mo.ui.number(start=1, stop=100_000, step=1, value=200, label="A")

    D = 1_000_000 * 10 ** 18
    return A, D, n


@app.cell
def _(mo, n):
    rates = mo.ui.array(
        [
            mo.ui.number(
                start=0.000001,
                step=0.000001,
                value=1.0,
                label=f"rate[{i}]",
            )
            for i in range(n.value)
        ],
        label="rates",
    )
    return (rates,)


@app.cell
def _(A, mo, n, rates):
    mo.vstack(
        [
            mo.hstack([n, A], justify="start"),
            rates,
        ]
    )
    return


@app.cell
def _(
    A,
    D,
    Decimal,
    StableSwap,
    copy,
    mo,
    n,
    pd,
    portfolio_value_bisection,
    portfolio_value_newton,
    portfolio_value_newton_alt,
    portfolio_value_secant,
    px,
    rates,
):
    # Assuming p = price_oracle
    MAX_P_FACTOR = 10  # Factor to limit price changes
    dx = D // 10_000
    dy = 0
    rate_values = [
        int(Decimal(str(rate)) * 10**18)
        for rate in rates.value
    ]

    def get_underlying_prices(pool):
        return [10 ** 18] + [int(_p) for _p in pool.get_p()]

    def get_prices(pool, underlying_prices):
        base_rate = pool.p[0]
        return [
            u_price * rate // base_rate
            for u_price, rate in zip(underlying_prices, pool.p)
        ]

    def get_relative_prices(prices):
        return [price * 10 ** 18 // prices[0] for price in prices]

    pool = StableSwap(A.value, D, n.value, p=rate_values, fee=0)
    underlying_prices = get_underlying_prices(pool)
    while (
        underlying_prices[1] / 10 ** 18 < MAX_P_FACTOR
        and 10 ** 18 / underlying_prices[1] < MAX_P_FACTOR
    ):
        dy = pool.exchange(0, 1, dx)
        underlying_prices = get_underlying_prices(pool)
    if dy > 0:
        pool.exchange(1, 0, dy)  # go 1 step back
    underlying_prices = get_underlying_prices(pool)

    points = []

    while (
        underlying_prices[1] / 10 ** 18 < MAX_P_FACTOR
        and 10 ** 18 / underlying_prices[1] < MAX_P_FACTOR
    ):
        pool.exchange(1, 0, dx)
        underlying_prices = get_underlying_prices(pool)
        prices_with_rates = get_prices(pool, underlying_prices)
        relative_prices = get_relative_prices(prices_with_rates)
        points.append({
            "balances": copy(pool.x),
            "underlying_prices": copy(underlying_prices),
            "prices": copy(relative_prices),
            "rates": copy(pool.p),
            "vp": pool.get_virtual_price(),
            "total_supply": copy(pool.tokens),
        })

    def real_lp_price(point):
        return (
            sum([b * p for p, b in zip(point["prices"], point["balances"])])
            // point["total_supply"]
            / 10**18
        )

    def simplified_lp_price(point):
        min_p = min(point["underlying_prices"]) * 10**18 // point["rates"][0]
        return min_p * point["vp"] // 10 ** 18 / 10 ** 18

    def converge_lp_price_result(point, method):
        value, iterations = method(A.value, int(point["underlying_prices"][1]))
        return value * point["vp"] // point["rates"][0] / 10**18, iterations

    data = []
    y_columns = ["Real", "Simplified", "bisection", "secant", "newton", "newton_alt"]
    iteration_columns = [
        "iterations_bisection",
        "iterations_secant",
        "iterations_newton",
        "iterations_newton_alt",
    ]

    for point in points:
        lp_row = {
            "price": point["prices"][1] / 10 ** 18,
            "underlying_price": point["underlying_prices"][1] / 10 ** 18,
            "Real": real_lp_price(point),
            "Simplified": simplified_lp_price(point),
        }
        for i, balance in enumerate(point["balances"]):
            lp_row[f"balance_{i}"] = balance / 10 ** 18
        bisection_price, bisection_iterations = converge_lp_price_result(point, portfolio_value_bisection)
        secant_price, secant_iterations = converge_lp_price_result(point, portfolio_value_secant)
        newton_price, newton_iterations = converge_lp_price_result(point, portfolio_value_newton)
        newton_alt_price, newton_alt_iterations = converge_lp_price_result(point, portfolio_value_newton_alt)
        lp_row.update(
            {
                "bisection": bisection_price,
                "secant": secant_price,
                "newton": newton_price,
                "newton_alt": newton_alt_price,
                "iterations_bisection": bisection_iterations,
                "iterations_secant": secant_iterations,
                "iterations_newton": newton_iterations,
                "iterations_newton_alt": newton_alt_iterations,
            }
        )
        data.append(lp_row)

    plot_df = pd.DataFrame(data)
    anchor_row = (
        plot_df[plot_df["underlying_price"] >= 1]
        .sort_values("underlying_price")
        .head(1)
    )
    anchor_simplified = (
        float(anchor_row["Simplified"].iloc[0])
        if not anchor_row.empty
        else None
    )

    fig = px.line(
        plot_df,
        x="underlying_price",
        y=y_columns,
        title="LP Price vs Underlying price",
        custom_data=["underlying_price", "price", *[f"balance_{i}" for i in range(n.value)]],
        labels={
            "underlying_price": "Underlying price",
            "price": "Price with rates",
            "value": "LP Price",
        },
    )
    if anchor_simplified is not None:
        fig.add_hline(
            y=anchor_simplified,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{anchor_simplified:.6f}",
            annotation_position="top right",
        )
    fig.update_xaxes(title="Underlying price")
    fig.update_traces(hovertemplate=(
        "Series: %{fullData.name}<br>"
        + "Underlying price: %{customdata[0]:.6f}<br>"
        + "LP Price: %{y:.6f}<br>"
        + "Price with rates: %{customdata[1]:.6f}"
        + "".join(
            f"<br>Balance[{i}]: %{{customdata[{i + 2}]:.6f}}"
            for i in range(n.value)
        )
        + "<extra></extra>"
    ))

    plot = mo.ui.plotly(fig)
    iter_fig = px.line(
        plot_df,
        x="underlying_price",
        y=iteration_columns,
        title="Iterations to abs_tol=1e-6 by method",
        labels={
            "underlying_price": "Underlying price",
            "value": "Iterations",
        },
    )
    iter_plot = mo.ui.plotly(iter_fig)
    simulation_view = mo.vstack([plot, iter_plot])

    simulation_view
    return


@app.cell
def _(math):
    Q  = 10**18
    Q2 = Q * Q
    Q3 = Q2 * Q
    Q4 = Q2 * Q2

    U256_MAX = 2**256 - 1

    A_SAFE_MAX = 84945683565681204819  # ~8.49e19
    ABS_TOL_Q = 10**12  # 1e-6 in Q-scaled price units

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


    def portfolio_value_bisection(A: int, p: int, *, abs_tol: int = ABS_TOL_Q, iters: int = 80) -> tuple[int, int]:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")

        if p < Q:
            p_inv = _inv_price_Q(p)
            value, iterations = portfolio_value_bisection(A, p_inv, abs_tol=abs_tol, iters=iters)
            return _mul_div_Q(p, value), iterations

        lo, hi = 1, Q - 1
        iterations = 0
        for _ in range(iters):
            iterations += 1
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            if abs(int(pm) - int(p)) <= abs_tol:
                return _value_from_s(A, p, mid), iterations
            if pm > p:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1:
                break

        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)
        sQ = lo if abs(int(plo) - int(p)) <= abs(int(phi) - int(p)) else hi
        return _value_from_s(A, p, sQ), iterations


    def portfolio_value_secant(
        A: int,
        p: int,
        *,
        abs_tol: int = ABS_TOL_Q,
        bisect_steps: int = 10,
        secant_steps: int = 256,
    ) -> tuple[int, int]:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")

        if p < Q:
            p_inv = _inv_price_Q(p)
            value, iterations = portfolio_value_secant(
                A,
                p_inv,
                abs_tol=abs_tol,
                bisect_steps=bisect_steps,
                secant_steps=secant_steps,
            )
            return _mul_div_Q(p, value), iterations

        lo, hi = 1, Q - 1
        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)
        iterations = 0

        for _ in range(bisect_steps):
            iterations += 1
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            if abs(int(pm) - int(p)) <= abs_tol:
                return _value_from_s(A, p, mid), iterations
            if pm > p:
                lo, plo = mid, pm
            else:
                hi, phi = mid, pm
            if hi - lo <= 1:
                break

        s0, g0 = lo, int(plo) - int(p)
        s1, g1 = hi, int(phi) - int(p)

        if abs(g0) <= abs_tol:
            return _value_from_s(A, p, s0), iterations
        if abs(g1) <= abs_tol:
            return _value_from_s(A, p, s1), iterations

        for _ in range(secant_steps):
            iterations += 1
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

            if abs(g2) <= abs_tol:
                return _value_from_s(A, p, s2), iterations
            if hi - lo <= 1:
                break

        sQ = lo if abs(int(plo) - int(p)) <= abs(int(phi) - int(p)) else hi
        return _value_from_s(A, p, sQ), iterations


    def portfolio_value_newton(
        A: int,
        p: int,
        *,
        abs_tol: int = ABS_TOL_Q,
        bisect_steps: int = 10,
        newton_steps: int = 256,
    ) -> tuple[int, int]:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")

        if p < Q:
            p_inv = _inv_price_Q(p)
            value, iterations = portfolio_value_newton(
                A,
                p_inv,
                abs_tol=abs_tol,
                bisect_steps=bisect_steps,
                newton_steps=newton_steps,
            )
            return _mul_div_Q(p, value), iterations

        lo, hi = 1, Q - 1
        plo = _p_from_s(A, lo)
        phi = _p_from_s(A, hi)
        iterations = 0

        for _ in range(bisect_steps):
            iterations += 1
            mid = (lo + hi) // 2
            pm = _p_from_s(A, mid)
            if abs(int(pm) - int(p)) <= abs_tol:
                return _value_from_s(A, p, mid), iterations
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
            if abs(gs) <= abs_tol:
                return _value_from_s(A, p, s), iterations
            if hi - lo <= 1:
                break

            iterations += 1

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
        return _value_from_s(A, p, sQ), iterations


    def _p_prime_abs(A: int, xQ: int, sQ: int, pQ: int) -> int:
        xx = xQ * xQ
        pxy = (pQ * xQ * sQ) // Q
        p2y2 = (pQ * pQ * sQ * sQ) // Q2
        if xx + p2y2 <= pxy:
            return 0
        n_w2 = xx + p2y2 - pxy
        bracket = (16 * A * xQ * xQ * sQ) // Q3 + 1
        xy2_w2 = (xQ * sQ * sQ) // Q
        d_w2 = xy2_w2 * bracket
        if d_w2 == 0:
            return 0
        return (2 * n_w2 * Q) // d_w2


    def _y_initial_guess_newton_alt(A: int, p: int) -> int:
        if p <= Q:
            return Q // 2
        diff = p - Q
        denom = 16 * A * diff
        if denom == 0:
            return Q // 2
        y0_sq = Q3 // denom
        y0 = _isqrt(y0_sq)
        if y0 >= Q // 2:
            return Q // 2
        if y0 == 0:
            return 1
        return y0


    def _y_from_newton_alt(A: int, p: int, *, abs_tol: int = ABS_TOL_Q) -> dict:
        MAX_NEWTON_ITERS = 64
        lo = 1
        hi = Q // 2 + 1
        sQ = _y_initial_guess_newton_alt(A, p)
        if sQ <= lo:
            sQ = lo + 1
        if sQ >= hi:
            sQ = hi - 1

        iterations = 0

        for _ in range(MAX_NEWTON_ITERS):
            iterations += 1
            xQ = _x_from_s(A, sQ)
            if xQ == 0:
                sQ = (lo + hi) // 2
                continue

            pm = _p_from_s(A, sQ)
            if pm > p:
                if pm - p <= abs_tol:
                    return {
                        "sQ": sQ,
                        "solver_iterations": iterations,
                    }
                lo = sQ
            else:
                if p - pm <= abs_tol:
                    return {
                        "sQ": sQ,
                        "solver_iterations": iterations,
                    }
                hi = sQ

            if hi - lo <= 1:
                return {
                    "sQ": hi,
                    "solver_iterations": iterations,
                }

            ppy = _p_prime_abs(A, xQ, sQ, pm)
            if ppy == 0:
                s_new = (lo + hi) // 2
            elif pm > p:
                delta = ((pm - p) * Q) // ppy
                s_new = sQ + delta
            else:
                delta = ((p - pm) * Q) // ppy
                if delta >= sQ:
                    s_new = (lo + hi) // 2
                else:
                    s_new = sQ - delta

            if s_new <= lo or s_new >= hi:
                s_new = (lo + hi) // 2

            sQ = s_new

        return {
            "sQ": sQ,
            "solver_iterations": iterations,
        }


    def portfolio_value_newton_alt(A: int, p: int, *, abs_tol: int = ABS_TOL_Q) -> tuple[int, int]:
        if A <= 0 or A > A_SAFE_MAX:
            raise ValueError(f"A must be in [1, {A_SAFE_MAX}]")
        if p == 0:
            raise ValueError("p!=0")

        reciprocal = p < Q
        solve_p = _inv_price_Q(p) if reciprocal else p
        details = _y_from_newton_alt(A, solve_p, abs_tol=abs_tol)
        value = _value_from_s(A, solve_p, details["sQ"])
        if reciprocal:
            value = _mul_div_Q(p, value)

        return value, details["solver_iterations"]
    return (
        portfolio_value_bisection,
        portfolio_value_newton,
        portfolio_value_newton_alt,
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
    comparison_a_min = mo.ui.number(start=0.01, stop=100_000, step=0.0001, value=1, label="A min")
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
    import boa

    WAD = 10**18
    A_PRECISION = 10**4

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
        "newton_alt": boa.load("stableswap/contracts/lp_oracle_newton_alt.vy"),
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
