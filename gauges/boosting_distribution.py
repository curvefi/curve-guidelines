import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    return mo, np, pd, px


@app.cell
def _(mo):
    boost_intro = mo.md(
        r"""
    # Curve gauge boost allocator

    The calculator maximizes the user's CRV share after fees. Direct staking uses
    the user's own veCRV and has no fee. Convex, Stake DAO and Yearn use their
    aggregate veCRV, but take a fee from the user's CRV rewards.

    Stake DAO OnlyBoost is not a fifth independent locker: it splits one deposit
    between Stake DAO and Convex. When OnlyBoost is selected, those two result rows
    show its internal allocation, avoiding double-counting their veCRV and gauge balances.
    """
    )
    return (boost_intro,)


@app.cell
def _(boost_intro, controls, mo, result_view):
    mo.vstack([boost_intro, mo.md("## Parameters"), controls, result_view])
    return


@app.cell
def _(mo):
    onlyboost = mo.ui.checkbox(value=True, label="Stake DAO OnlyBoost route")
    my_gauge_share = mo.ui.number(
        start=0, stop=100, step=0.01, value=1, label="My gauge share, %"
    )
    my_vecrv_share = mo.ui.number(
        start=0, stop=100, step=0.01, value=0, label="My veCRV share, %"
    )

    provider_defaults = [
        ("Convex", 54.14, 17.0),
        ("Stake DAO", 23.79, 15.5),
        ("Yearn", 10.67, 10.0),
    ]
    provider_inputs = [
        (
            name,
            mo.ui.number(
                start=0, stop=100, step=0.01, value=ve, label="veCRV share, %"
            ),
            mo.ui.number(
                start=0, stop=100, step=0.1, value=fee, label="CRV fee, %"
            ),
            mo.ui.number(
                start=0,
                stop=100,
                step=0.01,
                value=0,
                label="Existing gauge share, %",
            ),
        )
        for name, ve, fee in provider_defaults
    ]

    provider_rows = [
        mo.hstack(
            [mo.md(f"**{name}**"), ve, fee, gauge],
            widths=[1, 2, 2, 2],
            align="center",
        )
        for name, ve, fee, gauge in provider_inputs
    ]
    controls = mo.vstack(
        [
            mo.hstack([my_gauge_share, my_vecrv_share, onlyboost], justify="start"),
            mo.hstack(
                [
                    mo.md("**Route**"),
                    mo.md("**veCRV share**"),
                    mo.md("**Fee**"),
                    mo.md("**Existing gauge share**"),
                ],
                widths=[1, 2, 2, 2],
            ),
            *provider_rows,
        ]
    )
    return controls, my_gauge_share, my_vecrv_share, onlyboost, provider_inputs


@app.cell
def _(np):
    def working_balance(lp_share, ve_share):
        return min(lp_share, 0.4 * lp_share + 0.6 * ve_share)

    def allocation_result(routes, allocation):
        user_working = []
        route_working = []
        for (_, ve_share, fee, existing_share), added_share in zip(
            routes, allocation
        ):
            combined_share = existing_share + added_share
            working = working_balance(combined_share, ve_share)
            mine = added_share / combined_share * working if combined_share else 0.0
            route_working.append(working)
            user_working.append(mine)

        reported_share = sum(
            route[3] + added for route, added in zip(routes, allocation)
        )
        working_supply = sum(route_working) + 0.4 * max(0.0, 1.0 - reported_share)
        net_weight = sum(
            mine * (1.0 - route[2]) for route, mine in zip(routes, user_working)
        )
        reward_share = net_weight / working_supply if working_supply else 0.0
        return reward_share, user_working, working_supply

    def optimize_allocation(routes, amount):
        n_routes = len(routes)
        if amount <= 0:
            empty = np.zeros(n_routes)
            return empty, allocation_result(routes, empty)[0]

        starts = [np.full(n_routes, amount / n_routes)]
        starts.extend(np.eye(n_routes) * amount)
        best_allocation = starts[0]
        best_value = -1.0

        # Pairwise coordinate search is deterministic and handles the min() kink in
        # Curve's working-balance formula without an external nonlinear solver.
        for start in starts:
            allocation = start.copy()
            value = allocation_result(routes, allocation)[0]
            for _ in range(20):
                changed = False
                for i in range(n_routes):
                    for j in range(i + 1, n_routes):
                        pair_total = allocation[i] + allocation[j]
                        if pair_total == 0:
                            continue
                        candidates = np.linspace(0.0, pair_total, 101)
                        candidate_values = []
                        for candidate in candidates:
                            trial = allocation.copy()
                            trial[i] = candidate
                            trial[j] = pair_total - candidate
                            candidate_values.append(allocation_result(routes, trial)[0])
                        best_i = int(np.argmax(candidate_values))
                        fine_candidates = np.linspace(
                            candidates[max(0, best_i - 1)],
                            candidates[min(len(candidates) - 1, best_i + 1)],
                            101,
                        )
                        fine_values = []
                        for candidate in fine_candidates:
                            trial = allocation.copy()
                            trial[i] = candidate
                            trial[j] = pair_total - candidate
                            fine_values.append(allocation_result(routes, trial)[0])
                        fine_i = int(np.argmax(fine_values))
                        if fine_values[fine_i] > value + 1e-12:
                            allocation[i] = fine_candidates[fine_i]
                            allocation[j] = pair_total - fine_candidates[fine_i]
                            value = fine_values[fine_i]
                            changed = True
                if not changed:
                    break
            if value > best_value:
                best_allocation = allocation
                best_value = value

        return best_allocation, best_value
    return allocation_result, optimize_allocation


@app.cell
def _(
    allocation_result,
    mo,
    my_gauge_share,
    my_vecrv_share,
    np,
    onlyboost,
    optimize_allocation,
    pd,
    provider_inputs,
    px,
):
    providers = [
        (name, ve.value / 100, fee.value / 100, gauge.value / 100)
        for name, ve, fee, gauge in provider_inputs
    ]
    routes = [*providers, ("Direct", my_vecrv_share.value / 100, 0.0, 0.0)]
    amount = my_gauge_share.value / 100
    occupied = amount + sum(route[3] for route in routes)
    total_vecrv = sum(route[1] for route in routes)
    mo.stop(
        occupied > 1.0000001 or total_vecrv > 1.0000001,
        mo.callout(
            "Gauge shares and independent veCRV shares must each fit within 100%.",
            kind="danger",
        ),
    )

    allocation, reward_share = optimize_allocation(routes, amount)
    _, user_working, working_supply = allocation_result(routes, allocation)

    labels = [route[0] for route in routes]
    if onlyboost.value:
        labels[0] = "OnlyBoost → Convex"
        labels[1] = "OnlyBoost → Stake DAO"

    rows = []
    for label, route, added, mine in zip(labels, routes, allocation, user_working):
        boost = mine / (0.4 * added) if added else 0.0
        rows.append(
            {
                "Route": label,
                "Gauge allocation, %": 100 * added,
                "Share of my deposit, %": 100 * added / amount if amount else 0.0,
                "Gross boost": boost,
                "Fee, %": 100 * route[2],
                "Net working weight, %": 100 * mine * (1 - route[2]),
            }
        )
    result_table = pd.DataFrame(rows).round(4)

    baseline_rows = []
    for i, label in enumerate(labels):
        baseline = np.zeros(len(routes))
        baseline[i] = amount
        baseline_rows.append(
            {
                "Strategy": f"100% {label}",
                "Net CRV share, %": 100 * allocation_result(routes, baseline)[0],
            }
        )
    baseline_rows.append(
        {"Strategy": "Optimized", "Net CRV share, %": 100 * reward_share}
    )
    comparison = pd.DataFrame(baseline_rows)
    comparison_chart = px.bar(
        comparison,
        x="Strategy",
        y="Net CRV share, %",
        color="Strategy",
        text_auto=".4f",
    )
    comparison_chart.update_layout(showlegend=False)

    onlyboost_share = 100 * (allocation[0] + allocation[1]) / amount if amount else 0
    onlyboost_text = (
        f"OnlyBoost receives **{onlyboost_share:.2f}%** of the deposit and internally "
        f"splits it {100 * allocation[0] / amount if amount else 0:.2f}% to Convex / "
        f"{100 * allocation[1] / amount if amount else 0:.2f}% to Stake DAO."
        if onlyboost.value
        else "Convex and Stake DAO are treated as separate routes."
    )
    result_view = mo.vstack(
        [
            mo.md(
                f"""
                ## Result

                Net user share of all gauge CRV: **{100 * reward_share:.6f}%**

                Total working supply: **{100 * working_supply:.4f}%** of gauge supply

                {onlyboost_text}
                """
            ),
            result_table,
            comparison_chart,
            mo.callout(
                "Fees apply only to the service routes. The Direct row uses your own "
                "veCRV at zero fee, so the optimizer can keep the self-boosted part "
                "outside Convex, Stake DAO, Yearn and OnlyBoost.",
                kind="info",
            ),
        ]
    )
    return (result_view,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model

    For route $i$ with gauge share $l_i$, veCRV share $v_i$ and user addition
    $d_i$, Curve gives the route

    $$w_i = \min(l_i+d_i,\;0.4(l_i+d_i)+0.6v_i).$$

    The user's gross working balance inside that route is
    $d_i w_i/(l_i+d_i)$; its net contribution is multiplied by $(1-f_i)$.
    Unreported gauge balances are conservatively treated as unboosted. Provider
    veCRV shares and fees change over time, so the 2026-09-03 defaults remain editable.
    """
    )
    return


if __name__ == "__main__":
    app.run()
