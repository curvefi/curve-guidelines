import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# Settings""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, mo, np, pd


@app.cell
def _():
    MAX_SOURCES = 8
    DEFAULT_PRICES = [1.0000, 0.9992, 1.0024, 0.9965, 1.0060, 0.9940, 1.0011, 0.9984]
    DEFAULT_TVLS = [
        100_000_000,
        70_000_000,
        30_000_000,
        10_000_000,
        10_000_000,
        10_000_000,
        10_000_000,
        10_000_000,
    ]
    return DEFAULT_PRICES, DEFAULT_TVLS, MAX_SOURCES


@app.cell
def _(np):
    def bounded_normalize(values, max_share):
        values = np.array(values, dtype=float)
        if values.size == 0:
            return values, None

        effective_max_share = max(float(max_share), 1.0 / values.size)
        remaining = np.arange(values.size)
        shares = np.zeros(values.size)
        remaining_mass = 1.0

        while remaining.size > 0:
            remaining_values = values[remaining]
            remaining_sum = remaining_values.sum()
            if remaining_sum <= 0:
                shares[remaining] = remaining_mass / remaining.size
                break

            candidate = remaining_mass * remaining_values / remaining_sum
            capped = candidate > effective_max_share
            if not capped.any():
                shares[remaining] = candidate
                break

            capped_indices = remaining[capped]
            shares[capped_indices] = effective_max_share
            remaining_mass -= effective_max_share * capped_indices.size
            remaining = remaining[~capped]

        return shares, effective_max_share
    return (bounded_normalize,)


@app.cell
def _(
    mo,
    original_chart_view,
    original_formula_view,
    original_n_sources,
    original_result_view,
    original_sigma,
    original_source_inputs_view,
    original_summary_view,
):
    dashboard = mo.vstack(
        [
            original_formula_view,
            mo.md("### Parameters"),
            mo.hstack([original_n_sources, original_sigma], justify="start", wrap=True),
            original_source_inputs_view,
            original_result_view,
            original_summary_view,
            original_chart_view,
        ]
    )
    dashboard
    return


@app.cell
def _(mo):
    original_formula_view = mo.md(
        r"""
    # aggregate_stable_price simulator

    ## Original version

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
    return (original_formula_view,)


@app.cell
def _(MAX_SOURCES, mo):
    original_n_sources = mo.ui.number(
        start=1,
        stop=MAX_SOURCES,
        step=1,
        value=4,
        label="n sources",
    )
    original_sigma = mo.ui.number(
        start=0.000001,
        stop=0.1,
        step=0.000001,
        value=0.001,
        label="sigma",
    )
    return original_n_sources, original_sigma


@app.cell
def _(DEFAULT_PRICES, mo, original_n_sources):
    original_prices = mo.ui.array(
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
            for i in range(original_n_sources.value)
        ],
        label="",
    )
    return (original_prices,)


@app.cell
def _(DEFAULT_TVLS, mo, original_n_sources):
    original_tvls = mo.ui.array(
        [
            mo.ui.slider(
                start=0,
                stop=1_000_000_000,
                step=100_000,
                value=DEFAULT_TVLS[i],
                label="",
                show_value=True,
                include_input=True,
                full_width=True,
            )
            for i in range(original_n_sources.value)
        ],
        label="",
    )
    return (original_tvls,)


@app.cell
def _(mo, original_n_sources, original_prices, original_tvls):
    original_row_views = [
        mo.hstack(
            [
                mo.md(f"`source_{i + 1}`"),
                original_prices[i],
                original_tvls[i],
            ],
            justify="start",
            widths=[1, 3, 2],
            align="center",
        )
        for i in range(original_n_sources.value)
    ]

    original_source_inputs_view = mo.vstack(
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
            *original_row_views,
        ]
    )
    return (original_source_inputs_view,)


@app.cell
def _(np, original_n_sources, original_prices, original_sigma, original_tvls):
    original_source_ids = [
        f"source_{i + 1}" for i in range(original_n_sources.value)
    ]
    original_p = np.array(original_prices.value, dtype=float)
    original_D = np.array(original_tvls.value, dtype=float)
    original_sigma_error = None
    original_sigma_value = float(original_sigma.value)
    original_sigma_raw = int(round(original_sigma_value * 10**18))
    if original_sigma_value <= 0:
        original_sigma_error = "sigma must be positive"

    original_valid = original_D > 0
    original_active_ids = [
        original_source_ids[i]
        for i, is_valid in enumerate(original_valid)
        if is_valid
    ]
    original_active_prices = original_p[original_valid]
    original_active_tvls = original_D[original_valid]

    if original_sigma_error is not None or original_active_tvls.size == 0:
        original_p_avg = None
        original_e = np.array([])
        original_exp_score = np.array([])
        original_weights = np.array([])
        original_p_final = None
    else:
        original_p_avg = float(
            np.average(original_active_prices, weights=original_active_tvls)
        )
        original_e = (
            (original_active_prices - original_p_avg) ** 2
        ) / (original_sigma_value ** 2)
        original_exp_score = np.exp(-(original_e - original_e.min()))
        original_weights = original_active_tvls * original_exp_score
        original_p_final = float(
            np.average(original_active_prices, weights=original_weights)
        )
    return (
        original_active_ids,
        original_active_prices,
        original_active_tvls,
        original_e,
        original_exp_score,
        original_p_avg,
        original_p_final,
        original_sigma_error,
        original_sigma_raw,
        original_sigma_value,
        original_weights,
    )


@app.cell
def _(
    original_active_ids,
    original_active_prices,
    original_active_tvls,
    original_e,
    original_exp_score,
    original_p_avg,
    original_sigma_error,
    original_weights,
    pd,
):
    original_total_tvl = None
    if original_sigma_error is not None:
        original_summary_df = pd.DataFrame({"status": [original_sigma_error]})
    elif len(original_active_ids) == 0:
        original_summary_df = pd.DataFrame(
            {"status": ["Add at least one source with positive TVL."]}
        )
    else:
        original_total_tvl = original_active_tvls.sum()
        original_total_weight = original_weights.sum()
        original_summary_df = pd.DataFrame(
            {
                "source": original_active_ids,
                "price": original_active_prices,
                "tvl": original_active_tvls,
                "tvl_share": original_active_tvls / original_total_tvl,
                "e_i": original_e,
                "exp(-(e_i-e_min))": original_exp_score,
                "effective_weight": original_weights,
                "effective_share": original_weights / original_total_weight,
                "distance_to_p_avg_bps": (original_active_prices - original_p_avg)
                * 10_000,
                "contribution_to_final": original_weights
                * original_active_prices
                / original_total_weight,
            }
        )
    return original_summary_df, original_total_tvl


@app.cell
def _(
    mo,
    original_p_avg,
    original_p_final,
    original_sigma_error,
    original_sigma_raw,
    original_sigma_value,
    original_summary_df,
    original_total_tvl,
):
    if original_sigma_error is not None:
        original_result_view = mo.callout(original_sigma_error, kind="warn")
    elif original_p_avg is None or original_p_final is None:
        original_result_view = mo.callout(
            "At least one source must have positive TVL.",
            kind="warn",
        )
    else:
        original_result_view = mo.md(
            f"""
        ## Current result

        - `sigma = {original_sigma_value:.6f} ({original_sigma_raw})`
        - `total TVL = {original_total_tvl:,.0f}`
        - `p_avg = {original_p_avg:.6f}`
        - `p_final = {original_p_final:.6f}`
        - `aggregate adjustment = {(original_p_final - original_p_avg) * 10_000:+.2f} bps`
        """
        )

    original_summary_view = mo.ui.table(original_summary_df)
    return original_result_view, original_summary_view


@app.cell
def _(
    go,
    make_subplots,
    mo,
    original_active_ids,
    original_active_prices,
    original_active_tvls,
    original_p_avg,
    original_p_final,
):
    if original_p_avg is None or original_p_final is None:
        original_chart_view = mo.callout(
            "No chart yet: check sigma and source TVLs.",
            kind="warn",
        )
    else:
        original_fig = make_subplots(specs=[[{"secondary_y": True}]])
        original_fig.add_trace(
            go.Bar(
                x=original_active_ids,
                y=original_active_tvls,
                name="TVL",
                marker_color="#d8c3a5",
                opacity=0.60,
            ),
            secondary_y=True,
        )
        original_fig.add_trace(
            go.Scatter(
                x=original_active_ids,
                y=original_active_prices,
                name="source price",
                mode="lines+markers+text",
                text=[f"{value:.4f}" for value in original_active_prices],
                textposition="top center",
                marker=dict(size=11, color="#264653"),
                line=dict(color="#264653", width=2),
            ),
            secondary_y=False,
        )
        original_fig.add_hline(
            y=original_p_avg,
            line_dash="dash",
            line_color="#2a9d8f",
            annotation_text=f"p_avg = {original_p_avg:.6f}",
            annotation_position="top left",
        )
        original_fig.add_hline(
            y=original_p_final,
            line_dash="dot",
            line_color="#e76f51",
            annotation_text=f"p_final = {original_p_final:.6f}",
            annotation_position="bottom left",
        )
        original_fig.update_layout(
            title="Source prices vs TVL, p_avg and final aggregate price",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=40, r=40, t=70, b=40),
        )
        original_fig.update_yaxes(title_text="Price", secondary_y=False)
        original_fig.update_yaxes(title_text="TVL", secondary_y=True)
        original_chart_view = mo.vstack(
            [mo.md("## Comparison chart"), mo.ui.plotly(original_fig)]
        )
    return (original_chart_view,)


@app.cell
def _(
    mo,
    rel_capped_chart_view,
    rel_capped_compare_view,
    rel_capped_formula_view,
    rel_capped_n_sources,
    rel_capped_result_view,
    rel_capped_share_cap,
    rel_capped_sigma,
    rel_capped_source_inputs_view,
    rel_capped_summary_view,
):
    rel_capped_dashboard = mo.vstack(
        [
            rel_capped_formula_view,
            mo.md("### Parameters"),
            mo.hstack(
                [rel_capped_n_sources, rel_capped_sigma, rel_capped_share_cap],
                justify="start",
                wrap=True,
            ),
            rel_capped_source_inputs_view,
            rel_capped_result_view,
            rel_capped_compare_view,
            rel_capped_summary_view,
            rel_capped_chart_view,
        ]
    )
    rel_capped_dashboard
    return


@app.cell
def _(mo):
    rel_capped_formula_view = mo.md(
        r"""
    # Capped relative share

    In this variant, the pool weight is first converted into a relative share

    $$
    s_i = \frac{D_i}{\sum_j D_j}
    $$

    and then capped:

    $$
    s^{cap} = boundedNormalize(s, c), \quad s_i^{cap} \le c
    $$

    The capped shares replace raw `D_i / sum(D)` when computing `p_avg`:

    $$
    p_{avg}^{cap} = \sum_{i=1}^{n} s_i^{cap} p_i
    $$

    $$
    e_i^{cap} = \frac{(p_i - p_{avg}^{cap})^2}{\sigma^2}
    $$

    $$
    w_i^{cap} = s_i^{cap} \cdot \exp(-(e_i^{cap} - e_{min}^{cap})),
    \quad e_{min}^{cap} = \min_i e_i^{cap}
    $$

    $$
    p_{final}^{cap} = \frac{\sum_{i=1}^{n} w_i^{cap} p_i}{\sum_{i=1}^{n} w_i^{cap}}
    $$

    This keeps relative weighting across pools, but prevents one source from
    dominating the aggregate just because its TVL is much larger.

    Suggested cap policy:

    - `n = 2`: single-source attack is possible by construction; use a cap above
      `50%`, for example `70%`, if the deeper source should be trusted more.
    - `n = 3..5`: use `45%` to defend against one broken source.
    - `n = 6..8`: use `24%` to defend against two broken sources.

    The threshold interpretation is:

    - protection from one source requires `cap < 50%`;
    - protection from two sources requires `cap < 25%`;
    - intermediate values model partial compromise rather than a clean count of
      broken sources.
    """
    )
    return (rel_capped_formula_view,)


@app.cell
def _(MAX_SOURCES, mo):
    rel_capped_n_sources = mo.ui.number(
        start=1,
        stop=MAX_SOURCES,
        step=1,
        value=4,
        label="n sources",
    )
    rel_capped_sigma = mo.ui.number(
        start=0.000001,
        stop=0.1,
        step=0.000001,
        value=0.001,
        label="sigma",
    )
    rel_capped_share_cap = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.01,
        value=0.50,
        label="relative share cap",
        show_value=True,
    )
    return rel_capped_n_sources, rel_capped_share_cap, rel_capped_sigma


@app.cell
def _(DEFAULT_PRICES, mo, rel_capped_n_sources):
    rel_capped_prices = mo.ui.array(
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
            for i in range(rel_capped_n_sources.value)
        ],
        label="",
    )
    return (rel_capped_prices,)


@app.cell
def _(DEFAULT_TVLS, mo, rel_capped_n_sources):
    rel_capped_tvls = mo.ui.array(
        [
            mo.ui.slider(
                start=0,
                stop=1_000_000_000,
                step=100_000,
                value=DEFAULT_TVLS[i],
                label="",
                show_value=True,
                include_input=True,
                full_width=True,
            )
            for i in range(rel_capped_n_sources.value)
        ],
        label="",
    )
    return (rel_capped_tvls,)


@app.cell
def _(mo, rel_capped_n_sources, rel_capped_prices, rel_capped_tvls):
    rel_capped_row_views = [
        mo.hstack(
            [
                mo.md(f"`source_{i + 1}`"),
                rel_capped_prices[i],
                rel_capped_tvls[i],
            ],
            justify="start",
            widths=[1, 3, 2],
            align="center",
        )
        for i in range(rel_capped_n_sources.value)
    ]

    rel_capped_source_inputs_view = mo.vstack(
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
            *rel_capped_row_views,
        ]
    )
    return (rel_capped_source_inputs_view,)


@app.cell
def _(
    np,
    rel_capped_n_sources,
    rel_capped_prices,
    rel_capped_sigma,
    rel_capped_tvls,
):
    rel_capped_source_ids = [
        f"source_{i + 1}" for i in range(rel_capped_n_sources.value)
    ]
    rel_capped_p = np.array(rel_capped_prices.value, dtype=float)
    rel_capped_D = np.array(rel_capped_tvls.value, dtype=float)
    rel_capped_sigma_error = None
    rel_capped_sigma_value = float(rel_capped_sigma.value)
    if rel_capped_sigma_value <= 0:
        rel_capped_sigma_error = "sigma must be positive"

    rel_capped_valid = rel_capped_D > 0
    rel_capped_active_ids = [
        rel_capped_source_ids[i]
        for i, is_valid in enumerate(rel_capped_valid)
        if is_valid
    ]
    rel_capped_active_prices = rel_capped_p[rel_capped_valid]
    rel_capped_active_tvls = rel_capped_D[rel_capped_valid]

    if rel_capped_sigma_error is not None or rel_capped_active_tvls.size == 0:
        rel_capped_original_p_avg = None
        rel_capped_original_p_final = None
    else:
        rel_capped_original_p_avg = float(
            np.average(rel_capped_active_prices, weights=rel_capped_active_tvls)
        )
        rel_capped_original_e = (
            (rel_capped_active_prices - rel_capped_original_p_avg) ** 2
        ) / (rel_capped_sigma_value ** 2)
        rel_capped_original_exp_score = np.exp(
            -(rel_capped_original_e - rel_capped_original_e.min())
        )
        rel_capped_original_weights = (
            rel_capped_active_tvls * rel_capped_original_exp_score
        )
        rel_capped_original_p_final = float(
            np.average(
                rel_capped_active_prices,
                weights=rel_capped_original_weights,
            )
        )
    return (
        rel_capped_active_ids,
        rel_capped_active_prices,
        rel_capped_active_tvls,
        rel_capped_original_p_avg,
        rel_capped_original_p_final,
        rel_capped_sigma_error,
        rel_capped_sigma_value,
    )


@app.cell
def _(
    bounded_normalize,
    np,
    rel_capped_active_prices,
    rel_capped_active_tvls,
    rel_capped_share_cap,
    rel_capped_sigma_error,
    rel_capped_sigma_value,
):
    if rel_capped_sigma_error is not None or len(rel_capped_active_tvls) == 0:
        rel_capped_pre_shares = np.array([])
        rel_capped_shares = np.array([])
        rel_capped_e = np.array([])
        rel_capped_exp_score = np.array([])
        rel_capped_effective_share = np.array([])
        rel_capped_p_avg = None
        rel_capped_p_final = None
        rel_capped_raw_shares = np.array([])
    else:
        rel_capped_raw_shares = (
            rel_capped_active_tvls / rel_capped_active_tvls.sum()
        )
        rel_capped_shares, rel_capped_effective_share_cap = bounded_normalize(
            rel_capped_raw_shares,
            rel_capped_share_cap.value,
        )
        rel_capped_pre_shares = np.minimum(
            rel_capped_raw_shares,
            rel_capped_effective_share_cap,
        )
        rel_capped_p_avg = float(
            np.sum(rel_capped_shares * rel_capped_active_prices)
        )
        rel_capped_e = (
            (rel_capped_active_prices - rel_capped_p_avg) ** 2
        ) / (rel_capped_sigma_value ** 2)
        rel_capped_exp_score = np.exp(-(rel_capped_e - rel_capped_e.min()))
        rel_capped_effective_weight = rel_capped_shares * rel_capped_exp_score
        rel_capped_effective_share = (
            rel_capped_effective_weight / rel_capped_effective_weight.sum()
        )
        rel_capped_p_final = float(
            np.sum(rel_capped_effective_share * rel_capped_active_prices)
        )
    return (
        rel_capped_e,
        rel_capped_effective_share,
        rel_capped_exp_score,
        rel_capped_p_avg,
        rel_capped_p_final,
        rel_capped_pre_shares,
        rel_capped_raw_shares,
        rel_capped_shares,
    )


@app.cell
def _(
    pd,
    rel_capped_active_ids,
    rel_capped_active_prices,
    rel_capped_active_tvls,
    rel_capped_e,
    rel_capped_effective_share,
    rel_capped_exp_score,
    rel_capped_original_p_avg,
    rel_capped_original_p_final,
    rel_capped_p_avg,
    rel_capped_p_final,
    rel_capped_pre_shares,
    rel_capped_raw_shares,
    rel_capped_share_cap,
    rel_capped_shares,
    rel_capped_sigma_error,
):
    if rel_capped_sigma_error is not None:
        rel_capped_summary_df = pd.DataFrame(
            {"status": [rel_capped_sigma_error]}
        )
    elif len(rel_capped_active_ids) == 0:
        rel_capped_summary_df = pd.DataFrame(
            {"status": ["Add at least one source with positive TVL."]}
        )
    else:
        rel_capped_summary_df = pd.DataFrame(
            {
                "source": rel_capped_active_ids,
                "price": rel_capped_active_prices,
                "tvl": rel_capped_active_tvls,
                "raw_share": rel_capped_raw_shares,
                "capped_share_pre": rel_capped_pre_shares,
                "capped_share": rel_capped_shares,
                "e_i_capped": rel_capped_e,
                "exp_capped": rel_capped_exp_score,
                "effective_share_capped": rel_capped_effective_share,
            }
        )

    if rel_capped_sigma_error is not None:
        rel_capped_compare_df = pd.DataFrame(
            {"status": [rel_capped_sigma_error]}
        )
    elif (
        rel_capped_original_p_avg is None
        or rel_capped_original_p_final is None
        or rel_capped_p_avg is None
        or rel_capped_p_final is None
    ):
        rel_capped_compare_df = pd.DataFrame(
            {"status": ["Add at least one source with positive TVL."]}
        )
    else:
        rel_capped_compare_df = pd.DataFrame(
            {
                "version": ["original", "rel_capped"],
                "p_avg": [rel_capped_original_p_avg, rel_capped_p_avg],
                "p_final": [
                    rel_capped_original_p_final,
                    rel_capped_p_final,
                ],
                "delta_bps": [
                    0.0,
                    (rel_capped_p_final - rel_capped_original_p_final)
                    * 10_000,
                ],
                "share_cap": [None, rel_capped_share_cap.value],
            }
        )
    return rel_capped_compare_df, rel_capped_summary_df


@app.cell
def _(
    mo,
    rel_capped_compare_df,
    rel_capped_original_p_final,
    rel_capped_p_avg,
    rel_capped_p_final,
    rel_capped_share_cap,
    rel_capped_sigma_error,
    rel_capped_summary_df,
):
    if rel_capped_sigma_error is not None:
        rel_capped_result_view = mo.callout(
            rel_capped_sigma_error,
            kind="warn",
        )
    elif rel_capped_p_avg is None or rel_capped_p_final is None:
        rel_capped_result_view = mo.callout(
            "At least one source must have positive TVL.",
            kind="warn",
        )
    else:
        rel_capped_result_view = mo.md(
            f"""
        ### Rel-capped result

        - `share cap = {rel_capped_share_cap.value:.0%}`
        - `p_avg_rel_capped = {rel_capped_p_avg:.6f}`
        - `p_final_rel_capped = {rel_capped_p_final:.6f}`
        - `vs original p_final = {(rel_capped_p_final - rel_capped_original_p_final) * 10_000:+.2f} bps`
        """
        )

    rel_capped_compare_view = mo.ui.table(rel_capped_compare_df)
    rel_capped_summary_view = mo.ui.table(rel_capped_summary_df)
    return (
        rel_capped_compare_view,
        rel_capped_result_view,
        rel_capped_summary_view,
    )


@app.cell
def _(
    go,
    make_subplots,
    mo,
    rel_capped_active_ids,
    rel_capped_active_prices,
    rel_capped_original_p_avg,
    rel_capped_original_p_final,
    rel_capped_p_avg,
    rel_capped_p_final,
    rel_capped_raw_shares,
    rel_capped_shares,
    rel_capped_sigma_error,
):
    if (
        rel_capped_sigma_error is not None
        or rel_capped_original_p_avg is None
        or rel_capped_p_avg is None
        or rel_capped_original_p_final is None
        or rel_capped_p_final is None
    ):
        rel_capped_chart_view = mo.callout(
            "No chart yet: check sigma and source TVLs.",
            kind="warn",
        )
    else:
        rel_capped_fig = make_subplots(specs=[[{"secondary_y": True}]])
        rel_capped_fig.add_trace(
            go.Bar(
                x=rel_capped_active_ids,
                y=rel_capped_raw_shares,
                name="raw share",
                marker_color="#cdb4db",
                opacity=0.65,
            ),
            secondary_y=True,
        )
        rel_capped_fig.add_trace(
            go.Bar(
                x=rel_capped_active_ids,
                y=rel_capped_shares,
                name="capped share",
                marker_color="#84a59d",
                opacity=0.65,
            ),
            secondary_y=True,
        )
        rel_capped_fig.add_trace(
            go.Scatter(
                x=rel_capped_active_ids,
                y=rel_capped_active_prices,
                name="source price",
                mode="lines+markers+text",
                text=[f"{value:.4f}" for value in rel_capped_active_prices],
                textposition="top center",
                marker=dict(size=11, color="#264653"),
                line=dict(color="#264653", width=2),
            ),
            secondary_y=False,
        )
        rel_capped_fig.add_hline(
            y=rel_capped_original_p_avg,
            line_dash="dash",
            line_color="#2a9d8f",
            annotation_text=f"orig p_avg = {rel_capped_original_p_avg:.6f}",
            annotation_position="top left",
        )
        rel_capped_fig.add_hline(
            y=rel_capped_original_p_final,
            line_dash="dot",
            line_color="#e76f51",
            annotation_text=f"orig p_final = {rel_capped_original_p_final:.6f}",
            annotation_position="bottom left",
        )
        rel_capped_fig.add_hline(
            y=rel_capped_p_avg,
            line_dash="dash",
            line_color="#3d5a80",
            annotation_text=f"rel_capped p_avg = {rel_capped_p_avg:.6f}",
            annotation_position="top right",
        )
        rel_capped_fig.add_hline(
            y=rel_capped_p_final,
            line_dash="dot",
            line_color="#bc6c25",
            annotation_text=f"rel_capped p_final = {rel_capped_p_final:.6f}",
            annotation_position="bottom right",
        )
        rel_capped_fig.update_layout(
            title="Original vs rel-capped aggregate",
            barmode="group",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=40, r=40, t=70, b=40),
        )
        rel_capped_fig.update_yaxes(title_text="Price", secondary_y=False)
        rel_capped_fig.update_yaxes(
            title_text="Relative share",
            secondary_y=True,
            tickformat=".0%",
        )
        rel_capped_chart_view = mo.vstack(
            [
                mo.md("### Rel-capped comparison chart"),
                mo.ui.plotly(rel_capped_fig),
            ]
        )
    return (rel_capped_chart_view,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
