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
    mo.md(
        r"""
    # aggregate_stable_price simulator

    This notebook reproduces the core aggregation logic of `AggregatorStablePrice`
    in a simplified notation.

    Let:

    - `n` be the number of active price sources
    - `p_i` be the price from source `i`
    - `D_i(tvl)` be the TVL / EMA TVL of source `i`
    - `sigma` be the normalized tolerance to deviations from the weighted average
    - equivalently, in contract precision: `sigma_raw = sigma * 1e18`

    Then:

    $$
    p_{avg} = \frac{\sum_{i=1}^{n} D_i p_i}{\sum_{i=1}^{n} D_i}
    $$

    $$
    e_i = \frac{(p_i - p_{avg})^2}{\sigma^2}
    $$

    $$
    w_i = D_i \cdot \exp(-(e_i - e_{min})),
    \quad e_{min} = \min_i e_i
    $$

    $$
    p_{final} = \frac{\sum_{i=1}^{n} w_i p_i}{\sum_{i=1}^{n} w_i}
    $$

    Intuition:

    - `D_i` gives more influence to deeper sources
    - `sigma` controls how aggressively outliers are down-weighted
    - if a source is far from `p_avg`, its effective weight falls exponentially

    For simplicity, below we assume all selected sources are already above the
    contract's minimum liquidity threshold.
    """
    )
    return


@app.cell
def _():
    MAX_SOURCES = 8
    DEFAULT_PRICES = [1.0000, 0.9992, 1.0024, 0.9965, 1.0060, 0.9940, 1.0011, 0.9984]
    DEFAULT_TVLS = [100_000_000, 70_000_000, 30_000_000, 10_000_000, 10_000_000, 10_000_000, 10_000_000, 10_000_000]
    return DEFAULT_PRICES, DEFAULT_TVLS, MAX_SOURCES


@app.cell
def _(MAX_SOURCES, mo):
    n_sources = mo.ui.number(
        start=1,
        stop=MAX_SOURCES,
        step=1,
        value=4,
        label="n sources",
    )
    sigma = mo.ui.number(
        start=0.000001,
        stop=0.1,
        step=0.000001,
        value=0.001,
        label="sigma",
    )
    return n_sources, sigma


@app.cell
def _(DEFAULT_PRICES, mo, n_sources):
    prices = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0,
                stop=2.0,
                step=0.0001,
                value=DEFAULT_PRICES[i],
                label="",
                show_value=True,
                full_width=True,
            )
            for i in range(n_sources.value)
        ],
        label="",
    )
    return (prices,)


@app.cell
def _(DEFAULT_TVLS, mo, n_sources):
    tvls = mo.ui.array(
        [
            mo.ui.slider(
                start=0,
                stop=300_000_000,
                step=100_000,
                value=DEFAULT_TVLS[i],
                label="",
                show_value=True,
                include_input=True,
                full_width=True,
            )
            for i in range(n_sources.value)
        ],
        label="",
    )
    return (tvls,)


@app.cell
def _(mo, n_sources, prices, tvls):
    row_views = [
        mo.hstack(
            [
                mo.md(f"`source_{i + 1}`"),
                prices[i],
                tvls[i],
            ],
            justify="start",
            widths=[1, 3, 2],
            align="center",
        )
        for i in range(n_sources.value)
    ]

    source_inputs_view = mo.vstack(
        [
            mo.md("Rows below correspond to `source_1 ... source_n`."),
            mo.hstack(
                [
                    mo.md(""),
                    mo.md("**price p**"),
                    mo.md("**D (TVL)**"),
                ],
                justify="start",
                widths=[1, 3, 2],
            ),
            *row_views,
        ]
    )
    return (source_inputs_view,)


@app.cell
def _(n_sources, np, prices, sigma, tvls):
    source_ids = [f"source_{i + 1}" for i in range(n_sources.value)]
    p = np.array(prices.value, dtype=float)
    D = np.array(tvls.value, dtype=float)
    sigma_error = None
    sigma_value = float(sigma.value)
    sigma_raw = int(round(sigma_value * 10**18))
    if sigma_value <= 0:
        sigma_error = "sigma must be positive"

    valid = D > 0
    active_ids = [source_ids[i] for i, is_valid in enumerate(valid) if is_valid]
    active_prices = p[valid]
    active_tvls = D[valid]

    if sigma_error is not None or active_tvls.size == 0:
        p_avg = None
        e = np.array([])
        e_min = None
        exp_score = np.array([])
        weights = np.array([])
        p_final = None
    else:
        p_avg = float(np.average(active_prices, weights=active_tvls))
        e = ((active_prices - p_avg) ** 2) / (sigma_value ** 2)
        e_min = float(e.min())
        exp_score = np.exp(-(e - e_min))
        weights = active_tvls * exp_score
        p_final = float(np.average(active_prices, weights=weights))
    return (
        active_ids,
        active_prices,
        active_tvls,
        e,
        exp_score,
        p_avg,
        p_final,
        sigma_error,
        sigma_raw,
        sigma_value,
        weights,
    )


@app.cell
def _(
    active_ids,
    active_prices,
    active_tvls,
    e,
    exp_score,
    p_avg,
    pd,
    sigma_error,
    weights,
):
    total_tvl = None
    if sigma_error is not None:
        summary_df = pd.DataFrame(
            {
                "status": [sigma_error],
            }
        )
    elif len(active_ids) == 0:
        summary_df = pd.DataFrame(
            {
                "status": ["Add at least one source with positive TVL."],
            }
        )
    else:
        total_tvl = active_tvls.sum()
        total_weight = weights.sum()
        summary_df = pd.DataFrame(
            {
                "source": active_ids,
                "price": active_prices,
                "tvl": active_tvls,
                "tvl_share": active_tvls / total_tvl,
                "e_i": e,
                "exp(-(e_i-e_min))": exp_score,
                "effective_weight": weights,
                "effective_share": weights / total_weight,
                "distance_to_p_avg_bps": (active_prices - p_avg) * 10_000,
                "contribution_to_final": weights * active_prices / total_weight,
            }
        )
    return summary_df, total_tvl


@app.cell
def _(
    mo,
    p_avg,
    p_final,
    sigma_error,
    sigma_raw,
    sigma_value,
    summary_df,
    total_tvl,
):
    if sigma_error is not None:
        result_view = mo.callout(sigma_error, kind="warn")
    elif p_avg is None or p_final is None:
        result_view = mo.callout("At least one source must have positive TVL.", kind="warn")
    else:
        result_view = mo.md(
            f"""
        ## Current result

        - `sigma = {sigma_value:.6f} ({sigma_raw})`
        - `total TVL = {total_tvl:,.0f}`
        - `p_avg = {p_avg:.6f}`
        - `p_final = {p_final:.6f}`
        - `aggregate adjustment = {(p_final - p_avg) * 10_000:+.2f} bps`
        """
        )

    summary_view = mo.ui.table(summary_df)
    return result_view, summary_view


@app.cell
def _(
    active_ids,
    active_prices,
    active_tvls,
    go,
    make_subplots,
    mo,
    p_avg,
    p_final,
):
    if p_avg is None or p_final is None:
        chart_view = mo.callout("No chart yet: check sigma_raw and source TVLs.", kind="warn")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=active_ids,
                y=active_tvls,
                name="TVL",
                marker_color="#d8c3a5",
                opacity=0.60,
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=active_ids,
                y=active_prices,
                name="source price",
                mode="lines+markers+text",
                text=[f"{value:.4f}" for value in active_prices],
                textposition="top center",
                marker=dict(size=11, color="#264653"),
                line=dict(color="#264653", width=2),
            ),
            secondary_y=False,
        )

        fig.add_hline(
            y=p_avg,
            line_dash="dash",
            line_color="#2a9d8f",
            annotation_text=f"p_avg = {p_avg:.6f}",
            annotation_position="top left",
        )

        fig.add_hline(
            y=p_final,
            line_dash="dot",
            line_color="#e76f51",
            annotation_text=f"p_final = {p_final:.6f}",
            annotation_position="bottom left",
        )

        fig.update_layout(
            title="Source prices vs TVL, p_avg and final aggregate price",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=70, b=40),
        )
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="TVL", secondary_y=True)

        chart_view = mo.vstack(
            [
                mo.md("## Comparison chart"),
                mo.ui.plotly(fig),
            ]
        )
    return (chart_view,)


@app.cell
def _(
    chart_view,
    mo,
    n_sources,
    result_view,
    sigma,
    source_inputs_view,
    summary_view,
):
    dashboard = mo.vstack(
        [
            mo.md("## Parameters and result"),
            mo.hstack([n_sources, sigma], justify="start", wrap=True),
            source_inputs_view,
            result_view,
            summary_view,
            chart_view,
        ]
    )
    dashboard
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
