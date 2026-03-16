import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    # StableSwap curve vs A
    For a 2-coin StableSwap pool, the invariant is

    $$
    4 A (x + y) + D = 4 A D + \frac{D^3}{4 x y},
    $$

    where `A` is shown here in clean form.

    Spot price along the invariant can be written as

    $$
    p = \frac{4A + \frac{D^3}{4x^2y}}{4A + \frac{D^3}{4xy^2}}.
    $$

    We use the stable interval

    $$
    |\ln p| \le \frac{2}{2A + 1}.
    $$

    Near `p = 1`, one has

    $$
    \ln p \approx p - 1,
    $$

    so this behaves like a rule of the form `|1 - p| = O(1 / A)`.
    """
    )
    return


@app.cell
def _():
    import math
    import marimo as mo
    import plotly.graph_objects as go
    from stableswap.simulation import StableSwap
    return StableSwap, go, math, mo


@app.cell
def _(mo):
    A_control = mo.ui.number(
        start=1,
        stop=10_000,
        step=0.1,
        value=200,
        label="A (clean)",
    )
    scale_control = mo.ui.slider(
        start=1,
        stop=4,
        step=0.1,
        value=1,
        label="scale",
    )
    points_control = mo.ui.number(
        start=10,
        stop=20_000,
        step=10,
        value=100,
        label="points",
    )

    controls_panel = mo.hstack(
        [A_control, scale_control, points_control],
        justify="start",
        gap=2,
    )
    return A_control, controls_panel, points_control, scale_control


@app.cell
def _(mo):
    calc_A_control = mo.ui.number(
        start=1,
        stop=10_000,
        step=0.1,
        value=200,
        label="A -> p",
    )
    calc_edge_price_control = mo.ui.number(
        start=0.5,
        stop=10.0,
        step=0.0001,
        value=1.005,
        label="p* -> A",
    )

    calculator_panel = mo.hstack(
        [calc_A_control, calc_edge_price_control],
        justify="start",
        gap=2,
    )
    return calc_A_control, calc_edge_price_control, calculator_panel


@app.cell
def _(
    A_control,
    StableSwap,
    controls_panel,
    go,
    math,
    mo,
    points_control,
    scale_control,
):
    A_value = max(1e-12, float(A_control.value))
    A_clean = max(1, int(round(A_value)))
    D = 1_000_000 * 10**18
    midpoint = D // 2
    scale = float(scale_control.value)
    sample_points = max(2, int(points_control.value))

    sim = StableSwap(A_clean, [midpoint, midpoint], 2, fee=0)
    initial_balances = sim.x[:]
    max_scale = 1.0 + scale
    scale_grid = [
        max_scale ** (i / (sample_points - 1))
        for i in range(sample_points)
    ]

    curve = []
    for branch_scale in scale_grid[::-1]:
        sim.x = initial_balances[:]
        y_int = int(midpoint * branch_scale)
        x_int = sim.y(1, 0, y_int)
        sim.x = [x_int, y_int]
        p_raw = sim.get_p()
        p = p_raw[0] / 10**18
        curve.append(
            {
                "x": sim.x[0] / 10**18,
                "y": sim.x[1] / 10**18,
                "p": p,
            }
        )

    curve += [{"x": p["y"], "y": p["x"], "p": 1 / p["p"]} for p in curve[::-1]][1:]
    curve.sort(key=lambda point: point["x"])
    for point in curve:
        point["stable_interval"] = abs(math.log(point["p"])) <= 2.0 / (2 * A_value + 1)

    stable_curve = [point for point in curve if point["stable_interval"]]

    xy_fig = go.Figure()
    xy_fig.add_trace(
        go.Scatter(
            x=[point["x"] for point in curve],
            y=[point["y"] for point in curve],
            mode="lines",
            name="y(x)",
            line=dict(color="rgba(120,120,120,0.75)", width=2),
        )
    )
    xy_fig.add_trace(
        go.Scatter(
            x=[
                point["x"] if point["stable_interval"] else None
                for point in curve
            ],
            y=[
                point["y"] if point["stable_interval"] else None
                for point in curve
            ],
            mode="lines",
            name="|ln p| <= 2/(2A+1)",
            line=dict(color="#059669", width=4),
        )
    )
    if stable_curve:
        xy_fig.add_trace(
            go.Scatter(
                x=[stable_curve[0]["x"], stable_curve[-1]["x"]],
                y=[stable_curve[0]["y"], stable_curve[-1]["y"]],
                mode="markers+text",
                name="stable interval",
                marker=dict(color="#059669", size=9),
                text=["stable start", "stable end"],
                textposition="top center",
            )
        )
    xy_fig.update_layout(
        width=700,
        height=700,
        margin=dict(l=70, r=50, t=70, b=70),
        title=f"y(x), A={A_value:g}",
        xaxis_title="x",
        yaxis_title="y",
    )

    depth_rows = []
    for l, r in zip(curve, curve[1:]):
        p_mid = (l["p"] + r["p"]) / 2
        dp = abs(r["p"] - l["p"]) / p_mid
        dx = abs(r["x"] - l["x"])
        depth_rows.append({"p": p_mid, "depth": dx / dp})
    depth_rows.sort(key=lambda row: row["p"])
    for row in depth_rows:
        row["stable_interval"] = abs(math.log(row["p"])) <= 2.0 / (2 * A_value + 1)

    stable_depth_rows = [row for row in depth_rows if row["stable_interval"]]

    depth_fig = go.Figure()
    depth_fig.add_trace(
        go.Scatter(
            x=[row["p"] for row in depth_rows],
            y=[row["depth"] for row in depth_rows],
            mode="lines",
            name="depth",
            line=dict(color="rgba(120,120,120,0.75)", width=2),
        )
    )
    depth_fig.add_trace(
        go.Scatter(
            x=[
                row["p"] if row["stable_interval"] else None
                for row in depth_rows
            ],
            y=[
                row["depth"] if row["stable_interval"] else None
                for row in depth_rows
            ],
            mode="lines",
            name="|ln p| <= 2/(2A+1)",
            line=dict(color="#059669", width=4),
        )
    )
    depth_fig.update_layout(
        width=700,
        height=700,
        margin=dict(l=70, r=50, t=70, b=70),
        title=f"Depth vs price, A={A_value:g}",
        xaxis_title="price",
        yaxis_title="depth",
    )
    depth_fig.update_xaxes(type="log")
    if stable_depth_rows:
        stable_left = min(row["p"] for row in stable_depth_rows)
        stable_right = max(row["p"] for row in stable_depth_rows)
        depth_fig.add_vrect(
            x0=stable_left,
            x1=stable_right,
            fillcolor="#059669",
            opacity=0.10,
            layer="below",
            line_width=0,
            annotation_text="stable interval",
            annotation_position="top left",
        )
    depth_fig.add_vline(
        x=1.0,
        line_dash="dash",
        line_color="rgba(120,120,120,0.9)",
        annotation_text="p = 1",
        annotation_position="top",
    )
    if depth_rows:
        peak_row = max(depth_rows, key=lambda row: row["depth"])
        depth_fig.add_annotation(
            x=peak_row["p"],
            y=peak_row["depth"],
            text=f"max depth: {peak_row['depth']:.2f}",
            showarrow=True,
            arrowhead=2,
            ay=-40,
        )

    if stable_curve:
        highlighted_range = (
            f"[{min(point['p'] for point in stable_curve):.6f}, "
            f"{max(point['p'] for point in stable_curve):.6f}]"
        )
    else:
        highlighted_range = "no sampled points"

    summary = mo.md(
        f"""
Stable interval: **`|ln p| <= 2 / (2A + 1)`**  
Highlighted sampled price range: **{highlighted_range}**
"""
    )

    charts = mo.hstack(
        [mo.ui.plotly(xy_fig), mo.ui.plotly(depth_fig)],
        justify="start",
        gap=1,
        wrap=False,
    )

    charts_block = mo.vstack([controls_panel, summary, charts], gap=1)
    charts_block
    return


@app.cell
def _(calc_A_control, calc_edge_price_control, calculator_panel, math, mo):
    calc_A_value = max(1e-12, float(calc_A_control.value))
    calc_edge_price = max(1e-12, float(calc_edge_price_control.value))
    calc_upper_price = math.exp(2.0 / (2 * calc_A_value + 1))
    calc_lower_price = 1.0 / calc_upper_price
    calc_log_edge = abs(math.log(calc_edge_price))
    calc_implied_A = (2.0 / calc_log_edge - 1.0) / 2.0 if calc_log_edge > 0 else float("inf")

    calculator_title = mo.md(
        r"""
## Calculator
For the selected rule

$$
|\ln p| \le \frac{2}{2A + 1},
$$

convert between `A` and the boundary price `p*`.
"""
    )
    calculator_result = mo.md(
        f"""
From `A = {calc_A_value:g}`: **p in [{calc_lower_price:.6f}, {calc_upper_price:.6f}]**  
From `p* = {calc_edge_price:.6f}`: **A = {calc_implied_A:.6f}**
"""
    )

    calculator_block = mo.vstack([calculator_title, calculator_panel, calculator_result], gap=1)
    calculator_block
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
