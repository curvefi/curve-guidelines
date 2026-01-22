import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Oraclized StableSwap
    Oraclized StableSwap is a pool variant where the target price of one asset is scaled by an
    external oracle. Pools like stETH/ETH work this way: ETH is the base asset, and the
    staking token trades around the oracle price.
    """
    )
    return


@app.cell
def _(controls, d_view, mo, profit_view):
    max_update_description = mo.md(
        r"""
    ## Maximum update per block
    This study concerns a **single** oracle update.

    **Attack scenario (sandwich around the update):**  
    - before the update: the attacker trades in the favorable direction;  
    - oracle update: the price of token1 changes;  
    - after the update: the reverse trade closes the position.  

    **Conclusion:** to make a single update non-arbitrageable, the oracle change should be **no more than two fees** (≈ `2 * fee`).
    """
    )
    mo.accordion(
        {
            "Maximum update per block": mo.vstack(
                [
                    max_update_description,
                    controls,
                    profit_view,
                    d_view,
                ]
            )
        }
    )
    return


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    from stableswap.simulation import StableSwap
    return StableSwap, go, mo


@app.cell
def _(mo):
    A = mo.ui.number(start=10, stop=200_000, step=10, value=200, label="A")
    fee_pct = mo.ui.number(start=0.01, stop=0.1, step=0.001, value=0.01, label="Fee, %")
    trade_steps = mo.ui.slider(start=11, stop=81, step=2, value=11, label="Trade points (log, internal)")
    max_change_pct = mo.ui.slider(start=0.03, stop=3.0, step=0.01, value=0.05, label="Max |oracle change|, %", show_value=True)
    initial_price_steps = mo.ui.slider(start=11, stop=81, step=2, value=11, label="Initial price points (internal)")
    steps = mo.ui.slider(start=41, stop=201, step=10, value=41, label="Oracle points (horizontal)")
    return A, fee_pct, initial_price_steps, max_change_pct, steps, trade_steps


@app.cell
def _(A, fee_pct, initial_price_steps, max_change_pct, mo, steps, trade_steps):
    pool_controls = mo.hstack([A, fee_pct], justify="start", wrap=True)
    oracle_controls = mo.hstack([max_change_pct], justify="start", wrap=True)
    grid_controls = mo.hstack([initial_price_steps, steps, trade_steps], justify="start", wrap=True)
    controls = mo.vstack([pool_controls, oracle_controls, grid_controls])
    return (controls,)


@app.cell
def _(A, fee_pct):
    MIN_TRADE_SHARE = 1e-6
    MAX_TRADE_SHARE = 0.999
    A_int = int(A.value)
    D_wei = int(1_000_000 * 10**18)
    fee_int = int(fee_pct.value * 10**8)  # percent to 1e10 onchain scale
    params = {
        "A": A_int,
        "D": D_wei,
        "fee": fee_int,
    }
    return A_int, D_wei, MAX_TRADE_SHARE, MIN_TRADE_SHARE, fee_int, params


@app.cell
def _(
    A_int,
    D_wei,
    MAX_TRADE_SHARE,
    MIN_TRADE_SHARE,
    fee_int,
    initial_prices,
    price_points_pct,
    trade_shares,
):
    info = {
        "A": A_int,
        "Total liquidity": f"{D_wei/1e18:,.0f} tokens",
        "Fee": f"{fee_int/1e8:.4f}%",
        "Initial price range": f"{initial_prices[0]:.2f}..{initial_prices[-1]:.2f} ({len(initial_prices)} pts)",
        "Trade share range": (
            f"{MIN_TRADE_SHARE*100:.4f}%..{MAX_TRADE_SHARE*100:.2f}% "
            f"({len(trade_shares)} pts, log)"
        ),
        "Grid": f"{len(initial_prices)} x {len(price_points_pct)} (price x oracle)",
    }
    return


@app.cell
def _(
    MAX_TRADE_SHARE,
    MIN_TRADE_SHARE,
    StableSwap,
    fee_int,
    initial_price_steps,
    max_change_pct,
    params,
    steps,
    trade_steps,
):
    from typing import List, Optional, Tuple
    import math

    def value_at_prices(amounts: List[int], prices: List[int]) -> int:
        return sum(x * p // 10**18 for x, p in zip(amounts, prices))

    def pool_value(pool: "StableSwap") -> int:
        return value_at_prices(pool.x, pool.p)

    def input_for_target_output(
        pool: "StableSwap",
        i: int,
        j: int,
        target_dy: int,
        max_dx: int,
    ) -> Optional[int]:
        if pool.dy(i, j, max_dx) < target_dy:
            return None
        low, high = 0, max_dx
        for _ in range(90):
            mid = (low + high) // 2
            if pool.dy(i, j, mid) >= target_dy:
                high = mid
            else:
                low = mid
        return high

    def simulate_strategy(
        p0: List[int],
        new_p: List[int],
        base_val: int,
        base_D: int,
        trade_share: float,
        direction: int,
        reverse_mode: str,
    ) -> Optional[dict]:
        pool = StableSwap(params["A"], params["D"], 2, p=p0.copy(), fee=params["fee"])

        other = 1 - direction
        holdings = [0, 0]
        dx = int(pool.x[direction] * trade_share)
        dy = pool.exchange(direction, other, dx)
        holdings[direction] -= dx
        holdings[other] += dy

        pool.p = new_p
        if reverse_mode == "all":
            back = pool.exchange(other, direction, dy)
            holdings[direction] += back
            holdings[other] -= dy
        else:
            needed = input_for_target_output(pool, other, direction, dx, dy)
            if needed is None:
                return None
            back = pool.exchange(other, direction, needed)
            holdings[direction] += back
            holdings[other] -= needed

        profit_value = value_at_prices(holdings, new_p)
        final_val = pool_value(pool)
        final_D = pool.D()
        profit_ratio = profit_value / base_val
        pool_loss_ratio = (base_val - final_val) / base_val
        d_delta_ratio = (final_D - base_D) / base_D
        return {
            "profit": profit_ratio,
            "pool_loss": pool_loss_ratio,
            "D_delta": d_delta_ratio,
            "direction": direction,
            "reverse_mode": reverse_mode,
            "trade_share": trade_share,
            "trade_share_pct": trade_share * 100,
        }

    def logspace(start: float, stop: float, num: int) -> List[float]:
        if num <= 1:
            return [start]
        log_start = math.log10(start)
        log_stop = math.log10(stop)
        step = (log_stop - log_start) / (num - 1)
        return [10 ** (log_start + i * step) for i in range(num)]

    def linspace(start: float, stop: float, num: int) -> List[float]:
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

    def build_series() -> Tuple[List[float], List[float], List[List[Optional[float]]], List[List[Optional[float]]], List[float], Optional[dict], Optional[dict], List[Optional[float]], List[Optional[float]]]:
        spread_pct = max(max_change_pct.value, 2 * (fee_int / 1e8))
        spread = spread_pct / 100
        price_points = []
        for i in range(int(steps.value)):
            rel = -spread + 2 * spread * i / max(1, steps.value - 1)
            price_points.append(rel)
        selected = max_change_pct.value / 100
        if 0 < selected <= spread:
            price_points.extend([-selected, selected])
        price_points = sorted({round(val, 18) for val in price_points})
        price_points_pct = [val * 100 for val in price_points]

        trade_shares = logspace(
            MIN_TRADE_SHARE,
            MAX_TRADE_SHARE,
            int(trade_steps.value),
        )
        anchors = [1e-4, 0.9, 0.95, 0.99, 0.999]
        for anchor in anchors:
            if MIN_TRADE_SHARE <= anchor <= MAX_TRADE_SHARE:
                trade_shares.append(anchor)
        trade_shares = sorted({round(val, 18) for val in trade_shares})

        initial_prices = linspace(0.5, 2.0, int(initial_price_steps.value))
        initial_prices.append(1.0)
        initial_prices = sorted({round(val, 6) for val in initial_prices})

        profit_grid = []
        d_grid = []
        best_profit = None
        max_loss = None
        for initial_price in initial_prices:
            profit_row = []
            d_row = []
            for price_change in price_points:
                p0 = [10**18, int(initial_price * 10**18)]
                new_p = [10**18, int(initial_price * (1 + price_change) * 10**18)]
                base = StableSwap(params["A"], params["D"], 2, p=p0.copy(), fee=params["fee"])
                base.p = new_p
                base_val = pool_value(base)
                base_D = base.D()

                best_profit_point = None
                min_d_point = None
                for trade_share in trade_shares:
                    for direction in (0, 1):
                        for reverse_mode in ("all", "target"):
                            res = simulate_strategy(
                                p0,
                                new_p,
                                base_val,
                                base_D,
                                trade_share,
                                direction,
                                reverse_mode,
                            )
                            if res is None:
                                continue
                            res["price_change_pct"] = price_change * 100
                            res["initial_price"] = initial_price
                            if best_profit_point is None or res["profit"] > best_profit_point:
                                best_profit_point = res["profit"]
                            if min_d_point is None or res["D_delta"] < min_d_point:
                                min_d_point = res["D_delta"]
                            if best_profit is None or res["profit"] > best_profit["profit"]:
                                best_profit = res.copy()
                            if max_loss is None or res["pool_loss"] > max_loss["pool_loss"]:
                                max_loss = res.copy()

                profit_row.append(best_profit_point)
                d_row.append(min_d_point)
            profit_grid.append(profit_row)
            d_grid.append(d_row)
        profit_line = []
        d_line = []
        for idx in range(len(price_points)):
            profit_vals = [row[idx] for row in profit_grid if row[idx] is not None]
            d_vals = [row[idx] for row in d_grid if row[idx] is not None]
            profit_line.append(max(profit_vals) if profit_vals else None)
            d_line.append(min(d_vals) if d_vals else None)
        return (
            price_points_pct,
            initial_prices,
            profit_grid,
            d_grid,
            trade_shares,
            best_profit,
            max_loss,
            profit_line,
            d_line,
        )

    price_points_pct, initial_prices, profit_grid, d_grid, trade_shares, best_profit, max_loss, profit_line, d_line = build_series()
    return d_line, initial_prices, price_points_pct, profit_line, trade_shares


@app.cell
def _(go, mo, price_points_pct, profit_line):
    profit_line_pct = [None if val is None else val * 100 for val in profit_line]
    profit_fig = go.Figure(
        data=go.Scatter(
            x=price_points_pct,
            y=profit_line_pct,
            mode="lines",
            line={"width": 2},
        )
    )
    profit_fig.update_layout(
        title="Max attacker profit (% of pool)",
        xaxis_title="Oracle change, %",
        yaxis_title="Profit (% of pool)",
    )
    profit_view = mo.ui.plotly(profit_fig)
    return (profit_view,)


@app.cell
def _(d_line, go, mo, price_points_pct):
    d_line_pct = [None if val is None else val * 100 for val in d_line]
    d_fig = go.Figure(
        data=go.Scatter(
            x=price_points_pct,
            y=d_line_pct,
            mode="lines",
            line={"width": 2},
        )
    )
    d_fig.update_layout(
        title="Min ΔD (% of D)",
        xaxis_title="Oracle change, %",
        yaxis_title="ΔD (% of D)",
    )
    d_view = mo.ui.plotly(d_fig)
    return (d_view,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
