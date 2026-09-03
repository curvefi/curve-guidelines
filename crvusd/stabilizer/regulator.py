import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np, pd


@app.cell
def _(mo):
    regulator_intro = mo.md(
        r"""
    # PegKeeperRegulator simulator

    This notebook reproduces the debt limit and the `provide_allowed` /
    `withdraw_allowed` gates of the deployed mainnet regulator
    [`0x36a0…855f`](https://etherscan.io/address/0x36a04CAffc681fa179558B2Aaba30395CDdd855f#readContract).

    Current on-chain defaults (read 2026-09-03) are $\alpha=0.5$,
    $\beta=0.25$, worst-price threshold $=0.03\%$, and price deviation
    $=100\%$. The last value effectively disables the spot/oracle spam guard
    for ordinary prices; the deployment default used to be $0.05\%$.
    """
    )
    return (regulator_intro,)


@app.cell
def _(debt_controls, debt_view, guard_controls, guard_view, mo, regulator_intro):
    mo.vstack(
        [
            regulator_intro,
            mo.md("## Debt parameters"),
            debt_controls,
            debt_view,
            mo.md("## Price gates"),
            guard_controls,
            guard_view,
        ]
    )
    return


@app.cell
def _(mo):
    alpha = mo.ui.number(start=0, stop=1, step=0.01, value=0.5, label="alpha")
    beta = mo.ui.number(start=0, stop=1, step=0.01, value=0.25, label="beta")
    peer_count = mo.ui.slider(
        start=1, stop=7, step=1, value=3, label="Other PegKeepers"
    )
    target_debt = mo.ui.number(
        start=0,
        stop=1_000_000_000,
        step=100_000,
        value=2_500_000,
        label="Target PegKeeper debt",
    )
    target_idle = mo.ui.number(
        start=0,
        stop=1_000_000_000,
        step=100_000,
        value=7_500_000,
        label="Target idle crvUSD balance",
    )
    return alpha, beta, peer_count, target_debt, target_idle


@app.cell
def _(alpha, beta, mo, peer_count, target_debt, target_idle):
    peer_ratios = mo.ui.array(
        [
            mo.ui.slider(
                start=0,
                stop=1,
                step=0.01,
                value=0.25,
                label=f"Other #{i + 1} debt ratio",
            )
            for i in range(int(peer_count.value))
        ]
    )
    debt_controls = mo.vstack(
        [
            mo.hstack(
                [alpha, beta, peer_count, target_debt, target_idle],
                justify="start",
                wrap=True,
            ),
            mo.hstack(list(peer_ratios), justify="start", wrap=True),
        ]
    )
    return debt_controls, peer_ratios


@app.cell
def _(
    alpha,
    beta,
    go,
    make_subplots,
    mo,
    np,
    peer_count,
    peer_ratios,
    target_debt,
    target_idle,
):
    def max_ratio(alpha_value, beta_value, ratios):
        return (alpha_value + beta_value * sum(np.sqrt(ratios))) ** 2

    other_ratios = np.array(peer_ratios.value, dtype=float)
    allowed_ratio = max_ratio(alpha.value, beta.value, other_ratios)
    equivalent_peer_ratio = float(np.mean(np.sqrt(other_ratios)) ** 2)
    target_total = target_debt.value + target_idle.value
    target_ratio = target_debt.value / (1 + target_total)
    raw_allowance = allowed_ratio * target_total - target_debt.value
    allowance_text = (
        "would revert on unsigned subtraction"
        if raw_allowance < 0
        else f"{raw_allowance:,.0f} crvUSD"
    )

    ratio_space = np.linspace(0, 1, 201)
    limit_space = [
        min(
            1.0,
            max_ratio(alpha.value, beta.value, np.full(int(peer_count.value), ratio)),
        )
        for ratio in ratio_space
    ]
    debt_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Target debt limit", "Other PegKeeper debt ratios"),
    )
    debt_figure.add_trace(
        go.Scatter(
            x=100 * ratio_space,
            y=100 * np.array(limit_space),
            mode="lines",
            name="Effective target maximum",
        ),
        row=1,
        col=1,
    )
    debt_figure.add_trace(
        go.Scatter(
            x=[100 * equivalent_peer_ratio],
            y=[100 * min(1.0, allowed_ratio)],
            mode="markers",
            marker={"size": 11},
            name="Selected state",
        ),
        row=1,
        col=1,
    )
    debt_figure.add_trace(
        go.Bar(
            x=[f"PK #{i + 1}" for i in range(len(other_ratios))],
            y=100 * other_ratios,
            text=[f"{100 * ratio:.1f}%" for ratio in other_ratios],
            textposition="auto",
            name="Other PegKeepers",
        ),
        row=1,
        col=2,
    )
    debt_figure.update_xaxes(
        title_text="Common/equivalent other-PK ratio, %", row=1, col=1
    )
    debt_figure.update_yaxes(
        title_text="Effective maximum target ratio, %", range=[0, 100], row=1, col=1
    )
    debt_figure.update_yaxes(
        title_text="Debt ratio, %", range=[0, 100], row=1, col=2
    )
    debt_figure.update_layout(legend_title="")

    debt_view = mo.vstack(
        [
            mo.md(
                rf"""
                ## Debt allowance

                For the other PegKeeper ratios $r_i$:

                $$r_{{max}}=\left(\alpha+\beta\sum_i\sqrt{{r_i}}\right)^2.$$

                Effective $r_{{max}}$: **{100 * min(1.0, allowed_ratio):.2f}%**

                Current target debt ratio: **{100 * target_ratio:.2f}%**

                Additional provide allowance from the debt rule:
                **{allowance_text}**

                """
            ),
            debt_figure,
        ]
    )
    return (debt_view,)


@app.cell
def _(mo):
    aggregate_price = mo.ui.number(
        start=0.8, stop=1.2, step=0.0001, value=1.0, label="Aggregate crvUSD price"
    )
    target_spot = mo.ui.number(
        start=0.8, stop=1.2, step=0.0001, value=1.0, label="Target pool spot"
    )
    target_oracle = mo.ui.number(
        start=0.8, stop=1.2, step=0.0001, value=1.0, label="Target pool oracle"
    )
    largest_other_oracle = mo.ui.number(
        start=0.8,
        stop=1.2,
        step=0.0001,
        value=1.0,
        label="Largest other pool oracle",
    )
    price_deviation = mo.ui.number(
        start=0,
        stop=100,
        step=0.01,
        value=100,
        label="Allowed spot/oracle deviation, %",
    )
    worst_threshold = mo.ui.number(
        start=0,
        stop=1,
        step=0.01,
        value=0.03,
        label="Worst-price threshold, %",
    )
    killed = mo.ui.dropdown(
        options={"Nothing": 0, "Provide": 1, "Withdraw": 2, "Both": 3},
        value="Nothing",
        label="Killed direction",
    )
    guard_controls = mo.vstack(
        [
            mo.hstack(
                [aggregate_price, target_spot, target_oracle, largest_other_oracle],
                justify="start",
                wrap=True,
            ),
            mo.hstack(
                [price_deviation, worst_threshold, killed],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return (
        aggregate_price,
        guard_controls,
        killed,
        largest_other_oracle,
        price_deviation,
        target_oracle,
        target_spot,
        worst_threshold,
    )


@app.cell
def _(
    aggregate_price,
    killed,
    largest_other_oracle,
    mo,
    pd,
    price_deviation,
    target_oracle,
    target_spot,
    worst_threshold,
):
    deviation = price_deviation.value / 100
    threshold = worst_threshold.value / 100
    spot_in_range = abs(target_spot.value - target_oracle.value) < deviation
    target_not_worst = largest_other_oracle.value >= target_oracle.value - threshold
    provide_open = killed.value not in (1, 3)
    withdraw_open = killed.value not in (2, 3)

    provide_checks = [
        ("Direction is not killed", provide_open),
        ("Aggregate crvUSD price ≥ 1", aggregate_price.value >= 1),
        ("Target spot is close to its oracle", spot_in_range),
        ("Another pool is not materially worse", target_not_worst),
    ]
    withdraw_checks = [
        ("Direction is not killed", withdraw_open),
        ("Aggregate crvUSD price ≤ 1", aggregate_price.value <= 1),
        ("Target spot is close to its oracle", spot_in_range),
    ]
    provide_allowed = all(value for _, value in provide_checks)
    withdraw_allowed = all(value for _, value in withdraw_checks)

    checks = pd.DataFrame(
        [
            {"Action": "Provide", "Check": name, "Pass": passed}
            for name, passed in provide_checks
        ]
        + [
            {"Action": "Withdraw", "Check": name, "Pass": passed}
            for name, passed in withdraw_checks
        ]
    )
    guard_view = mo.vstack(
        [
            mo.md(
                f"""
                ## Price and safety gates

                `provide_allowed`: **{'OPEN' if provide_allowed else 'BLOCKED'}**

                `withdraw_allowed`: **{'OPEN' if withdraw_allowed else 'BLOCKED'}**

                Provide also returns the debt allowance calculated above. Withdraw
                returns `max_value(uint256)` when all of its gates pass.
                """
            ),
            checks,
        ]
    )
    return (guard_view,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Contract correspondence

    The simulator follows
    [`PegKeeperRegulator.vy`](https://github.com/curvefi/curve-stablecoin/blob/master/curve_stablecoin/stabilizer/PegKeeperRegulator.vy):

    - $r_i=debt_i/(1+debt_i+idle\ crvUSD_i)$;
    - provide requires aggregate price at or above 1, a target spot close to
      its oracle, and at least one other oracle no lower than the target minus
      `worst_price_threshold`;
    - withdraw requires aggregate price at or below 1 and the same spot/oracle guard.
    """
    )
    return


if __name__ == "__main__":
    app.run()
