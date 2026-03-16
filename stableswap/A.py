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

    The stable interval is defined by

    $$
    |1 - p| \le \frac{1}{A},
    $$

    where `p` is the spot price along the invariant curve.
    """
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
    A_control = mo.ui.slider(
        start=1,
        stop=10_000,
        step=1,
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
def _(
    A_control,
    StableSwap,
    controls_panel,
    go,
    mo,
    points_control,
    scale_control,
):
    A_clean = int(A_control.value)
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
                "stable_interval": abs(1.0 - p) <= 1.0 / A_clean,
            }
        )

    curve += [{"x": p["y"], "y": p["x"], "p": 1/p["p"], "stable_interval": p["stable_interval"]} for p in curve[::-1]][1:]
    curve.sort(key=lambda point: point["x"])
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
            x=[point["x"] for point in curve if point["stable_interval"]],
            y=[point["y"] for point in curve if point["stable_interval"]],
            mode="lines",
            name="A <= 1/|1-p|",
            line=dict(color="#d97706", width=4),
        )
    )
    if stable_curve:
        xy_fig.add_trace(
            go.Scatter(
                x=[stable_curve[0]["x"], stable_curve[-1]["x"]],
                y=[stable_curve[0]["y"], stable_curve[-1]["y"]],
                mode="markers+text",
                name="stable interval",
                marker=dict(color="#d97706", size=9),
                text=["stable start", "stable end"],
                textposition="top center",
            )
        )
    xy_fig.update_layout(
        width=700,
        height=700,
        margin=dict(l=70, r=50, t=70, b=70),
        title=f"y(x), A={A_clean}",
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
    stable_depth_rows = [row for row in depth_rows if abs(1.0 - row["p"]) <= 1.0 / A_clean]

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
            x=[row["p"] if abs(1.0 - row["p"]) <= 1.0 / A_clean else None for row in depth_rows],
            y=[row["depth"] if abs(1.0 - row["p"]) <= 1.0 / A_clean else None for row in depth_rows],
            mode="lines",
            name="A <= 1/|1-p|",
            line=dict(color="#d97706", width=4),
        )
    )
    depth_fig.update_layout(
        width=700,
        height=700,
        margin=dict(l=70, r=50, t=70, b=70),
        title=f"Depth vs price, A={A_clean}",
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
            fillcolor="#d97706",
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

    charts = mo.hstack(
        [mo.ui.plotly(xy_fig), mo.ui.plotly(depth_fig)],
        justify="start",
        gap=1,
        wrap=False,
    )

    result = mo.vstack([controls_panel, charts], gap=1)
    result
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
