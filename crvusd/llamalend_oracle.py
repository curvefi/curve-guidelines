import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import math

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, math, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
    # LlamaLend oracle requirements and sizing

    An oracle for LLAMMA is part of the liquidation mechanism, not only a price
    display. It should move slowly enough for soft-liquidation and
    de-liquidation to traverse bands, while still converging before bad debt can
    accumulate. The [sDOLA-long2 incident](https://gov.curve.finance/t/llamalend-sdola-long2-post-mortem/11020)
    is a concrete example of why a discontinuous collateral rate is dangerous.

    This notebook is a first-pass parameter screen. Final settings still require
    historical replay, manipulation tests and a market fork with the exact
    controller, LLAMMA, oracle wrappers and liquidity sources.
    """
    )
    return


@app.cell
def _(mo):
    current_ltv = mo.ui.slider(
        start=1, stop=99, step=0.5, value=70, label="Current LTV, %"
    )
    liquidation_ltv = mo.ui.slider(
        start=1, stop=99, step=0.5, value=85, label="Liquidation LTV, %"
    )
    adverse_shock = mo.ui.slider(
        start=0.1, stop=90, step=0.1, value=30, label="Raw adverse price shock, %"
    )
    shock_direction = mo.ui.dropdown(
        options={"Price down": -1, "Price up": 1},
        value="Price down",
        label="Shock direction",
    )
    response_hours = mo.ui.number(
        start=0.1, stop=168, step=0.1, value=6, label="Borrower / keeper response, h"
    )
    amm_A = mo.ui.number(start=2, stop=1000, step=1, value=100, label="LLAMMA A")
    update_minutes = mo.ui.number(
        start=0.1, stop=1440, step=0.1, value=15, label="Oracle update interval, min"
    )
    max_bands = mo.ui.number(
        start=0.1, stop=20, step=0.1, value=1, label="Max bands per first update"
    )
    ema_time = mo.ui.number(
        start=1, stop=20_000, step=1, value=480, label="EMA e-folding time, min"
    )
    ema_controls = mo.vstack(
        [
            mo.hstack(
                [
                    current_ltv,
                    liquidation_ltv,
                    adverse_shock,
                    shock_direction,
                    response_hours,
                ],
                justify="start",
                wrap=True,
            ),
            mo.hstack(
                [amm_A, update_minutes, max_bands, ema_time],
                justify="start",
                wrap=True,
            ),
        ]
    )
    ema_controls
    return (
        adverse_shock,
        amm_A,
        current_ltv,
        ema_time,
        liquidation_ltv,
        max_bands,
        response_hours,
        shock_direction,
        update_minutes,
    )


@app.cell
def _(math):
    def required_ema_time(shock, allowed_move, horizon_minutes):
        if allowed_move <= 0:
            return math.inf
        if shock <= allowed_move:
            return 0.0
        return -horizon_minutes / math.log(1.0 - allowed_move / shock)
    return (required_ema_time,)


@app.cell
def _(
    adverse_shock,
    amm_A,
    current_ltv,
    ema_time,
    go,
    liquidation_ltv,
    make_subplots,
    math,
    max_bands,
    mo,
    np,
    required_ema_time,
    response_hours,
    shock_direction,
    update_minutes,
):
    ltv_0 = current_ltv.value / 100
    ltv_limit = liquidation_ltv.value / 100
    shock = adverse_shock.value / 100
    direction = shock_direction.value
    response = response_hours.value * 60
    tau = ema_time.value

    ltv_move_budget = max(0.0, 1.0 - ltv_0 / ltv_limit)
    ltv_tau = (
        required_ema_time(shock, ltv_move_budget, response)
        if direction < 0
        else 0.0
    )
    band_log_width = math.log(amm_A.value / (amm_A.value - 1))
    band_move_budget = (
        1.0 - math.exp(-max_bands.value * band_log_width)
        if direction < 0
        else math.exp(max_bands.value * band_log_width) - 1.0
    )
    band_tau = required_ema_time(shock, band_move_budget, update_minutes.value)
    recommended_tau = max(ltv_tau, band_tau)

    horizon = max(response, 4 * tau, 12 * update_minutes.value)
    times = np.linspace(0, horizon, 400)
    raw_price = np.full_like(times, 1.0 + direction * shock)
    raw_price[0] = 1.0
    oracle_price = 1.0 + direction * shock * (1.0 - np.exp(-times / tau))
    oracle_ltv = ltv_0 / oracle_price

    ema_figure = make_subplots(specs=[[{"secondary_y": True}]])
    ema_figure.add_trace(
        go.Scatter(x=times / 60, y=raw_price, name="Raw price"), secondary_y=False
    )
    ema_figure.add_trace(
        go.Scatter(x=times / 60, y=oracle_price, name="EMA price"), secondary_y=False
    )
    ema_figure.add_trace(
        go.Scatter(x=times / 60, y=100 * oracle_ltv, name="Oracle LTV"),
        secondary_y=True,
    )
    ema_figure.add_hline(
        y=100 * ltv_limit,
        line_dash="dash",
        line_color="red",
        annotation_text="Liquidation LTV",
        secondary_y=True,
    )
    ema_figure.update_xaxes(title_text="Hours after a step shock")
    ema_figure.update_yaxes(title_text="Relative price", secondary_y=False)
    ema_figure.update_yaxes(title_text="LTV, %", secondary_y=True)

    selected_safe = tau >= recommended_tau
    first_update_price = 1 + direction * shock * (
        1 - np.exp(-update_minutes.value / tau)
    )
    first_update_bands = abs(math.log(first_update_price)) / band_log_width
    recommendation = (
        "No finite EMA time protects an already liquidatable position."
        if math.isinf(recommended_tau)
        else f"At least **{recommended_tau:.0f} min** ({recommended_tau / 60:.2f} h)"
    )
    mo.vstack(
        [
            mo.md(
                rf"""
                ## EMA sizing

                For a step from $p_0$ to $p_1$, the model uses the same e-folding
                convention as Curve EMA oracles:

                $$p(t)=p_1+(p_0-p_1)e^{{-t/\tau}}.$$

                Shock direction: **{'down' if direction < 0 else 'up'}**

                Required $\tau$ from the LTV response window: **{ltv_tau:.0f} min**

                Required $\tau$ from the first-update band limit: **{band_tau:.0f} min**

                Combined screening recommendation: {recommendation}

                Selected $\tau$: **{tau:.0f} min** — **{'PASS' if selected_safe else 'TOO FAST'}**

                Bands crossed by the first update: **{first_update_bands:.2f}**

                The band estimate uses $\Delta\ln p=\ln(A/(A-1))$. The LTV rule
                limits the observed adverse move during the response window to
                $1-LTV_0/LTV_{{liq}}$.
                """
            ),
            ema_figure,
            mo.callout(
                "Smoothing is a trade-off: increasing τ helps orderly band traversal "
                "but keeps a stale favorable price longer after a real collapse. Test "
                "both downward and upward jumps; ERC-4626 or rebasing collateral can "
                "be unsafe in the upward direction as well.",
                kind="warn" if not selected_safe else "info",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    collateral_usd_0 = mo.ui.number(
        start=0.0001, stop=1_000_000, step=0.01, value=100, label="Collateral USD now"
    )
    collateral_usd_1 = mo.ui.number(
        start=0.0001,
        stop=1_000_000,
        step=0.01,
        value=90,
        label="Collateral USD scenario",
    )
    crvusd_usd_0 = mo.ui.number(
        start=0.5, stop=1.5, step=0.001, value=1, label="crvUSD USD now"
    )
    crvusd_usd_1 = mo.ui.number(
        start=0.5, stop=1.5, step=0.001, value=0.97, label="crvUSD USD scenario"
    )
    quote_controls = mo.hstack(
        [collateral_usd_0, collateral_usd_1, crvusd_usd_0, crvusd_usd_1],
        justify="start",
        wrap=True,
    )
    quote_controls
    return collateral_usd_0, collateral_usd_1, crvusd_usd_0, crvusd_usd_1


@app.cell
def _(
    collateral_usd_0,
    collateral_usd_1,
    crvusd_usd_0,
    crvusd_usd_1,
    current_ltv,
    go,
    liquidation_ltv,
    make_subplots,
    mo,
    np,
    pd,
):
    collateral_move = collateral_usd_1.value / collateral_usd_0.value
    crvusd_move = crvusd_usd_1.value / crvusd_usd_0.value
    usd_anchor_ltv = current_ltv.value / collateral_move
    relative_ltv = current_ltv.value * crvusd_move / collateral_move
    initial_relative_price = collateral_usd_0.value / crvusd_usd_0.value
    final_relative_price = collateral_usd_1.value / crvusd_usd_1.value

    quote_table = pd.DataFrame(
        [
            {
                "Oracle convention": "USD anchor: collateral / USD",
                "Scenario oracle price": collateral_usd_1.value,
                "Implied LTV, %": usd_anchor_ltv,
                "Peg arbitrage from crvUSD depeg": "Yes",
            },
            {
                "Oracle convention": "Market-relative: collateral / crvUSD",
                "Scenario oracle price": final_relative_price,
                "Implied LTV, %": relative_ltv,
                "Peg arbitrage from crvUSD depeg": "No",
            },
        ]
    ).round(4)
    ltv_error = usd_anchor_ltv - relative_ltv

    crvusd_prices = np.linspace(0.9, 1.1, 201)
    relative_ltv_space = (
        current_ltv.value
        * (crvusd_prices / crvusd_usd_0.value)
        / collateral_move
    )
    usd_anchor_ltv_space = np.full_like(crvusd_prices, usd_anchor_ltv)
    oracle_mispricing = 100 * (crvusd_prices - 1)

    quote_figure = make_subplots(specs=[[{"secondary_y": True}]])
    quote_figure.add_trace(
        go.Scatter(
            x=crvusd_prices,
            y=usd_anchor_ltv_space,
            name="LTV with USD anchor",
        ),
        secondary_y=False,
    )
    quote_figure.add_trace(
        go.Scatter(
            x=crvusd_prices,
            y=relative_ltv_space,
            name="LTV with relative oracle",
        ),
        secondary_y=False,
    )
    quote_figure.add_trace(
        go.Scatter(
            x=crvusd_prices,
            y=oracle_mispricing,
            name="USD-anchor mispricing",
            line={"dash": "dot"},
        ),
        secondary_y=True,
    )
    quote_figure.add_hline(
        y=liquidation_ltv.value,
        line_dash="dash",
        annotation_text="Liquidation LTV",
        secondary_y=False,
    )
    quote_figure.add_vline(x=1, line_dash="dash", annotation_text="crvUSD peg")
    quote_figure.update_xaxes(title_text="crvUSD / USD market price")
    quote_figure.update_yaxes(title_text="Oracle-implied LTV, %", secondary_y=False)
    quote_figure.update_yaxes(
        title_text="USD-anchor price error vs relative market, %", secondary_y=True
    )

    if crvusd_usd_1.value < 0.9995:
        arbitrage = (
            "crvUSD is below $1: an arbitrageur buys crvUSD externally, spends it "
            "in LLAMMA for underpriced collateral, and sells the collateral outside. "
            "This creates crvUSD demand and moves borrower bands toward soft-liquidation."
        )
    elif crvusd_usd_1.value > 1.0005:
        arbitrage = (
            "crvUSD is above $1: an arbitrageur buys collateral outside, sells it "
            "into LLAMMA for overpriced crvUSD, and sells the crvUSD externally. "
            "This creates sell pressure and moves borrower bands toward de-liquidation."
        )
    else:
        arbitrage = "crvUSD is at peg, so this quote choice creates no depeg-only arbitrage."

    mo.vstack(
        [
            mo.md(
                f"""
                ## USD anchor vs market-relative quote

                `Collateral/USD ÷ crvUSD/USD` and a direct `collateral/crvUSD`
                pool produce the same quote. They are one market-relative convention;
                only their source and manipulation risks differ.

                $$P_{{collateral/crvUSD}}=P_{{collateral/USD}}/P_{{crvUSD/USD}}.$$

                Initial relative price: **{initial_relative_price:.4f} crvUSD**

                Scenario relative price: **{final_relative_price:.4f} crvUSD**

                Using the USD anchor instead changes the scenario LTV by
                **{ltv_error:+.2f} percentage points**.

                **Arbitrage:** {arbitrage}

                The peg support is therefore real, but it is paid for with LLAMMA
                inventory and borrower state. Below peg, the USD anchor is more
                conservative than the relative quote; above peg it is less conservative.
                It affects both band conversion and hard-liquidation health. Capacity is
                limited by the collateral/crvUSD inventory available in active bands.
                """
            ),
            quote_table,
            quote_figure,
        ]
    )
    return


@app.cell
def _(mo):
    debt_ceiling = mo.ui.number(
        start=0,
        stop=1_000_000_000,
        step=100_000,
        value=10_000_000,
        label="Market debt ceiling, USD",
    )
    coverage = mo.ui.number(
        start=0, stop=500, step=5, value=20, label="Required TVL / debt ceiling, %"
    )
    current_tvl = mo.ui.number(
        start=0, stop=10_000_000_000, step=100_000, value=10_000_000, label="Pool TVL"
    )
    low_tvl = mo.ui.number(
        start=0,
        stop=10_000_000_000,
        step=100_000,
        value=7_000_000,
        label="Observed low TVL",
    )
    outflow = mo.ui.slider(
        start=0, stop=100, step=1, value=50, label="Stress outflow, %"
    )
    oracle_age = mo.ui.number(
        start=0, stop=10_000, step=1, value=15, label="Current source age, min"
    )
    max_age = mo.ui.number(
        start=1, stop=10_000, step=1, value=60, label="Maximum source age, min"
    )
    permissionless = mo.ui.checkbox(value=True, label="Permissionless price source")
    update_path = mo.ui.checkbox(value=True, label="Market has an oracle replacement path")
    source_active = mo.ui.checkbox(value=True, label="Pool/source is active")
    liquidity_controls = mo.vstack(
        [
            mo.hstack([debt_ceiling, coverage, current_tvl, low_tvl, outflow], wrap=True),
            mo.hstack(
                [oracle_age, max_age, permissionless, update_path, source_active],
                justify="start",
                wrap=True,
            ),
        ]
    )
    liquidity_controls
    return (
        coverage,
        current_tvl,
        debt_ceiling,
        low_tvl,
        max_age,
        oracle_age,
        outflow,
        permissionless,
        source_active,
        update_path,
    )


@app.cell
def _(
    coverage,
    current_tvl,
    debt_ceiling,
    low_tvl,
    max_age,
    mo,
    oracle_age,
    outflow,
    pd,
    permissionless,
    source_active,
    update_path,
):
    required_tvl = debt_ceiling.value * coverage.value / 100
    stressed_tvl = min(low_tvl.value, current_tvl.value * (1 - outflow.value / 100))
    source_checks = [
        ("Permissionless source", permissionless.value),
        ("Stressed TVL covers the configured exposure", stressed_tvl >= required_tvl),
        ("Source is fresh", oracle_age.value <= max_age.value),
        ("Pool/source is active", source_active.value),
        ("Oracle can be replaced if the source degrades", update_path.value),
    ]
    must_replace = not all(value for name, value in source_checks if name != "Permissionless source")
    source_table = pd.DataFrame(
        [{"Requirement": name, "Pass": value} for name, value in source_checks]
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ## Permissionless source and liquidity lifecycle

                Required source TVL: **${required_tvl:,.0f}**

                Stressed source TVL: **${stressed_tvl:,.0f}**

                Operational decision: **{'REPLACE / PAUSE ORACLE' if must_replace else 'KEEP AND MONITOR'}**

                Permissionless pool oracles are preferred because anyone can update
                them and their state is auditable. Permissionless does not mean
                manipulation-proof: size the source against market exposure, monitor
                both TVL and trading activity, and define an oracle replacement or
                market-pause procedure before launch.
                """
            ),
            source_table,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Launch checklist

    - Confirm quote direction and decimals end-to-end with the deployed controller.
    - Simulate upward and downward jumps, stale periods, donations/rate inflation,
      pool imbalance and TVL withdrawal.
    - Replay the worst historical moves and measure bands crossed, bad debt,
      arbitrage loss and time spent on the wrong side of the market.
    - Bound oracle update size or add EMA smoothing; document why the chosen $\tau$
      is safe for the selected LTV, liquidation discount, $A$ and keeper cadence.
    - Check the exact AMM version's internal oracle limiter and dynamic fee; model
      the effective price after both the wrapper EMA and the AMM limiter.
    - Use more than one independent path where possible, and avoid circular sources
      that derive their price from the market being protected.
    - Set monitoring thresholds and an executable governance/emergency process for
      replacing a deprecated or drained pool.

    Curve's production oracle contracts use the e-folding form shown above; see
    [`CryptoWithStablePrice.vy`](https://github.com/curvefi/curve-stablecoin/blob/master/curve_stablecoin/price_oracles/CryptoWithStablePrice.vy).
    The current [`AMM.vy`](https://github.com/curvefi/curve-stablecoin/blob/master/curve_stablecoin/AMM.vy)
    defines the band grid and also limits abrupt external-oracle changes.
    """
    )
    return


if __name__ == "__main__":
    app.run()
