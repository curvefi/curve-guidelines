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
def _(
    apy_controls,
    apy_description,
    apy_view,
    controls,
    d_view,
    gradual_update_description,
    gradual_view,
    max_update_description,
    mo,
    profit_view,
):
    mo.accordion(
        {
            "Maximum update per block": mo.vstack(
                [
                    max_update_description,
                    controls,
                    profit_view,
                    d_view,
                ]
            ),
            "Gradual oracle updates (2 * fee per block)": mo.vstack(
                [
                    gradual_update_description,
                    controls,
                    gradual_view,
                ]
            ),
            "How much is oracle change?": mo.vstack(
                [
                    apy_description,
                    apy_controls,
                    apy_view,
                ]
            ),
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
    mo.md(r"""## Maximum update per block""")
    return


@app.cell
def _(mo):
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
    return (max_update_description,)


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
    return MAX_TRADE_SHARE, MIN_TRADE_SHARE, fee_int, params


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

    def build_series() -> Tuple[List[float], List[float], List[List[Optional[float]]], List[List[Optional[float]]], List[List[Optional[float]]], List[float], Optional[dict], Optional[dict], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
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
        sandwich_loss_grid = []
        best_profit = None
        max_loss = None
        for initial_price in initial_prices:
            profit_row = []
            d_row = []
            sandwich_loss_row = []
            for price_change in price_points:
                p0 = [10**18, int(initial_price * 10**18)]
                new_p = [10**18, int(initial_price * (1 + price_change) * 10**18)]
                base = StableSwap(params["A"], params["D"], 2, p=p0.copy(), fee=params["fee"])
                base.p = new_p
                base_val = pool_value(base)
                base_D = base.D()

                best_profit_point = None
                min_d_point = None
                max_loss_point = None
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
                            if max_loss_point is None or res["pool_loss"] > max_loss_point:
                                max_loss_point = res["pool_loss"]
                            if best_profit is None or res["profit"] > best_profit["profit"]:
                                best_profit = res.copy()
                            if max_loss is None or res["pool_loss"] > max_loss["pool_loss"]:
                                max_loss = res.copy()

                profit_row.append(best_profit_point)
                d_row.append(min_d_point)
                sandwich_loss_row.append(max_loss_point)
            profit_grid.append(profit_row)
            d_grid.append(d_row)
            sandwich_loss_grid.append(sandwich_loss_row)
        profit_line = []
        d_line = []
        sandwich_loss_line = []
        for idx in range(len(price_points)):
            profit_vals = [row[idx] for row in profit_grid if row[idx] is not None]
            d_vals = [row[idx] for row in d_grid if row[idx] is not None]
            loss_vals = [row[idx] for row in sandwich_loss_grid if row[idx] is not None]
            profit_line.append(max(profit_vals) if profit_vals else None)
            d_line.append(min(d_vals) if d_vals else None)
            sandwich_loss_line.append(max(loss_vals) if loss_vals else None)
        return (
            price_points_pct,
            initial_prices,
            profit_grid,
            d_grid,
            sandwich_loss_grid,
            trade_shares,
            best_profit,
            max_loss,
            profit_line,
            d_line,
            sandwich_loss_line,
        )

    price_points_pct, initial_prices, profit_grid, d_grid, sandwich_loss_grid, trade_shares, best_profit, max_loss, profit_line, d_line, sandwich_loss_line = build_series()
    return d_line, initial_prices, price_points_pct, profit_line, sandwich_loss_line


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
def _(mo):
    mo.md(r"""## Gradual oracle updates""")
    return


@app.cell
def _(mo):
    gradual_update_description = mo.md(
        r"""
    ## Gradual oracle updates (2 * fee per block)
    This study assumes the true price jumps to a new value and stays there,
    while the pool oracle can move only by **2 * fee** per block.
    After each oracle step, the pool is arbitraged to the true price using
    the latest oracle value.

    The chart shows the **maximum pool loss** across initial prices,
    measured as the value shortfall vs holding the initial balances at the
    final true price.
    It also compares against the **single-update sandwich loss** from the
    maximum-update study.
    """
    )
    return (gradual_update_description,)


@app.cell
def _(
    MAX_TRADE_SHARE,
    StableSwap,
    fee_int,
    initial_prices,
    params,
    price_points_pct,
):
    import typing as t

    def value_at_prices_gradual(amounts: t.List[int], prices: t.List[int]) -> int:
        return sum(x * p // 10**18 for x, p in zip(amounts, prices))

    def price_after_trade(pool: "StableSwap", i: int, j: int, dx: int) -> float:
        if dx <= 0:
            return pool.get_p()[0]
        trial = StableSwap(
            pool.A,
            pool.x.copy(),
            pool.n,
            p=pool.p.copy(),
            fee=pool.fee,
        )
        trial.exchange(i, j, dx)
        return trial.get_p()[0]

    def gradual_trade_profit(pool: "StableSwap", i: int, j: int, dx: int, true_price: int) -> int:
        if dx <= 0:
            return 0
        dy = pool.dy(i, j, dx)
        if i == 0:
            return dy * true_price // 10**18 - dx
        return dy - dx * true_price // 10**18

    def arbitrage_to_price(
        pool: "StableSwap",
        target_internal: float,
        true_price: int,
        max_iter: int = 60,
        tol: float = 1e-10,
    ) -> None:
        current = pool.get_p()[0]
        if target_internal <= 0:
            return
        if abs(current - target_internal) / target_internal <= tol:
            return
        if current < target_internal:
            i, j = 0, 1
        else:
            i, j = 1, 0
        max_dx = int(pool.x[i] * MAX_TRADE_SHARE)
        if max_dx <= 0:
            return
        price_max = price_after_trade(pool, i, j, max_dx)
        if current < target_internal:
            reaches_target = price_max >= target_internal
        else:
            reaches_target = price_max <= target_internal
        if reaches_target:
            low, high = 0, max_dx
            for _ in range(max_iter):
                mid = (low + high) // 2
                price_mid = price_after_trade(pool, i, j, mid)
                if current < target_internal:
                    if price_mid < target_internal:
                        low = mid
                    else:
                        high = mid
                else:
                    if price_mid > target_internal:
                        low = mid
                    else:
                        high = mid
            dx_target = high
        else:
            dx_target = max_dx
        if dx_target <= 0:
            return
        if gradual_trade_profit(pool, i, j, dx_target, true_price) <= 0:
            low, high = 0, dx_target
            for _ in range(max_iter):
                mid = (low + high) // 2
                if gradual_trade_profit(pool, i, j, mid, true_price) >= 0:
                    low = mid
                else:
                    high = mid
            dx_target = low
        if dx_target > 0:
            pool.exchange(i, j, dx_target)

    def simulate_gradual_update(initial_price: float, price_change: float) -> t.Optional[float]:
        p_start = int(initial_price * 10**18)
        p_true = int(initial_price * (1 + price_change) * 10**18)
        pool = StableSwap(params["A"], params["D"], 2, p=[10**18, p_start], fee=params["fee"])
        base_val = value_at_prices_gradual(pool.x, [10**18, p_true])
        if p_true == p_start:
            return 0.0
        step_pct = 2 * (fee_int / 1e8)
        if step_pct <= 0:
            pool.p = [10**18, p_true]
            arbitrage_to_price(pool, 1e18, p_true)
            final_val = value_at_prices_gradual(pool.x, [10**18, p_true])
            return (base_val - final_val) / base_val
        step_ratio = step_pct / 100
        max_steps = int(abs((p_true - p_start) / p_start) / step_ratio) + 5
        p_oracle = p_start
        for _ in range(max_steps):
            if p_oracle == p_true:
                break
            remaining_ratio = (p_true - p_oracle) / p_oracle
            if abs(remaining_ratio) <= step_ratio:
                p_oracle = p_true
            else:
                p_oracle = int(p_oracle * (1 + step_ratio * (1 if remaining_ratio > 0 else -1)))
            pool.p = [10**18, p_oracle]
            # Normalize the true price by the current oracle price.
            target_internal = p_true * 1e18 / p_oracle
            arbitrage_to_price(pool, target_internal, p_true)
        final_val = value_at_prices_gradual(pool.x, [10**18, p_true])
        return (base_val - final_val) / base_val

    price_changes = [pct / 100 for pct in price_points_pct]
    loss_grid = []
    for initial_price in initial_prices:
        row = []
        for price_change in price_changes:
            row.append(simulate_gradual_update(initial_price, price_change))
        loss_grid.append(row)

    loss_line = []
    for idx in range(len(price_changes)):
        vals = [row[idx] for row in loss_grid if row[idx] is not None]
        loss_line.append(max(vals) if vals else None)
    return (loss_line,)


@app.cell
def _(go, loss_line, mo, price_points_pct, sandwich_loss_line):
    loss_line_pct = [None if val is None else val * 100 for val in loss_line]
    sandwich_loss_line_pct = [None if val is None else val * 100 for val in sandwich_loss_line]
    loss_fig = go.Figure(
        data=[
            go.Scatter(
                x=price_points_pct,
                y=loss_line_pct,
                mode="lines",
                line={"width": 2},
                name="Gradual updates",
            ),
            go.Scatter(
                x=price_points_pct,
                y=sandwich_loss_line_pct,
                mode="lines",
                line={"width": 2, "dash": "dash"},
                name="Single-update sandwich",
            ),
        ]
    )
    loss_fig.update_layout(
        title="Pool loss with gradual oracle updates (% of pool)",
        xaxis_title="Oracle change, %",
        yaxis_title="Loss (% of pool)",
    )
    gradual_view = mo.ui.plotly(loss_fig)
    return (gradual_view,)


@app.cell
def _(mo):
    mo.md(r"""## How much is oracle change""")
    return


@app.cell
def _(mo):
    apy_description = mo.md(
        r"""
    ## How much is oracle change?
    This calculator converts **oracle change per update** into annualized yield
    (APY) using per-period compounding for intuition.

    The window for oracle change is the **update time** (default 12 seconds).
    Vault implementations may not update smoothly, so changes can be accrued
    over a longer period (e.g., periodic accruals like TradFi).
    This can also happen during blockchain outages or delayed updates.
    """
    )
    return (apy_description,)


@app.cell
def _(mo):
    block_seconds = mo.ui.number(
        start=1,
        stop=15_552_000,
        step=1,
        value=12,
        label="Update time, sec",
    )
    apy_max_change_pct = mo.ui.number(
        start=0.0001,
        stop=0.2,
        step=0.0001,
        value=0.02,
        label="Max oracle change per update, %",
    )
    apy_controls = mo.hstack([block_seconds, apy_max_change_pct], justify="start", wrap=True)
    return apy_controls, apy_max_change_pct, block_seconds


@app.cell
def _(apy_max_change_pct, block_seconds, mo):
    import math as apy_math

    apy_seconds_per_year = 365 * 24 * 60 * 60
    apy_max_change_value = float(apy_max_change_pct.value)
    apy_step = apy_max_change_value / 4 if apy_max_change_value else 0.0
    apy_change_points = [
        -4 * apy_step,
        -3 * apy_step,
        -2 * apy_step,
        -1 * apy_step,
        0.0,
        1 * apy_step,
        2 * apy_step,
        3 * apy_step,
        4 * apy_step,
    ]

    apy_block_time = float(block_seconds.value)
    apy_periods = [
        ("custom update time", apy_block_time),
        ("1 month", 30 * 24 * 60 * 60),
        ("1 week", 7 * 24 * 60 * 60),
        ("1 day", 24 * 60 * 60),
        ("1 hour", 60 * 60),
        ("1 minute", 60),
        ("12 sec", 12),
    ]
    apy_rows = []
    for apy_label, apy_seconds in apy_periods:
        if apy_seconds <= 0:
            apy_row_values = [None for _ in apy_change_points]
        else:
            apy_periods_per_year = apy_seconds_per_year / apy_seconds
            apy_row_values = []
            for apy_change in apy_change_points:
                apy_rate = apy_change / 100
                if apy_rate <= -1:
                    apy_row_values.append(None)
                    continue
                apy_log_growth = apy_math.log1p(apy_rate) * apy_periods_per_year
                if apy_log_growth > 700:
                    apy_row_values.append(None)
                    continue
                apy_row_values.append(apy_math.expm1(apy_log_growth) * 100)
        apy_rows.append((apy_label, apy_row_values))

    def apy_format_pct(value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"
        return f"{text}%"

    change_labels = [apy_format_pct(val) for val in apy_change_points]
    header = "| Metric | " + " | ".join(change_labels) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(change_labels)) + " |"
    apy_lines = [header, separator]
    for apy_label, apy_row_values in apy_rows:
        apy_labels = [
            "n/a" if apy_value is None else f"{apy_value:.6g}%"
            for apy_value in apy_row_values
        ]
        apy_lines.append("| " + apy_label + " | " + " | ".join(apy_labels) + " |")

    apy_view = mo.md("\n".join(apy_lines))
    return (apy_view,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
