import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    from copy import copy
    from stableswap.simulation import StableSwap
    return StableSwap, copy, mo, px


@app.cell
def _(mo):
    n = mo.ui.number(start=2, stop=8, step=1, value=2, label="n")
    A = mo.ui.number(start=10, stop=100_000, step=1, value=200, label="A")
    skew = mo.ui.number(start=0, stop=1, step=0.0001, value=0.5, label="skew")

    D = 1_000_000 * 10 ** 18
    return A, D, n, skew


@app.cell
def _(A, mo):
    mo.md(f"""{A}""")
    return


@app.cell
def _(mo, skew):
    mo.md(f"""{skew}""")
    return


@app.cell
def _(A, D, StableSwap, copy, mo, n, px, skew):
    # Assuming p = price_oracle
    MAX_P_FACTOR = 1.75  # Factor to limit price changes
    dx = D // 10_000
    dy = 0

    skewed_prices = [10 ** 18, int((1 + skew.value) * 10 ** 18)]
    pool = StableSwap(A.value, D, n.value, p=skewed_prices, fee=0)
    p = pool.get_p()
    while p[0] / 10 ** 18 < MAX_P_FACTOR and 10 ** 18 / p[0] < MAX_P_FACTOR:
        dy = pool.exchange(0, 1, dx)
        p = pool.get_p()
    pool.exchange(1, 0, dy)  # go 1 step back
    p = pool.get_p()

    points = []

    while p[0] / 10 ** 18 < MAX_P_FACTOR and 10 ** 18 / p[0] < MAX_P_FACTOR:
        pool.exchange(1, 0, dx)
        p = pool.get_p()
        points.append({
            "balances": copy(pool.x),
            "prices": [10 ** 18] + p,
            "effective_prices": [10 ** 18, pool.dy(0, 1, 10 ** 18)],
        })

    data = [
        {
            "price": point["prices"][1] / 10**18,
            "proportion of skewed token": point["balances"][1] / sum(point["balances"]),
        }
        for point in points
    ]

    # Line chart
    ratio_line = px.line(
        data,
        x="price",
        y="proportion of skewed token",
        title="Skewed proportion",
    )

    x_points = [round(0.8 + i * 0.05, 2) for i in range(int(round((1.3 - 0.8) / 0.05)) + 1)]

    # Gather the nearest price to each target
    tickvals, ticktext = [], []
    for x in x_points:
        closest = min(data, key=lambda d: abs(d["price"] - x))
        yval = closest["proportion of skewed token"]
        if yval > 0:
            tickvals.append(yval)
            ticktext.append(f"{yval:.4f}")

    # Apply log scale + custom ticks
    ratio_line.update_yaxes(
        # type="log",
        tickvals=tickvals,
        ticktext=ticktext,
    )
    ratio_plot = mo.ui.plotly(ratio_line)

    ratio_plot
    return (points,)


@app.cell
def _(mo, points, px):
    balances_plot = mo.ui.plotly(
        px.line(
            [{"price": point["prices"][1] / 10 ** 18, "token balance": point["balances"][0] / 10 ** 18, "skewed balance": point["balances"][1] / 10 ** 18} for point in points],
            x="price",
            y=["token balance", "skewed balance"],
            title="Balances",
        )
    )

    balances_plot
    return


@app.cell
def _(mo, points, px):
    tokens_plot = mo.ui.plotly(
        px.line(
            [{"price": point["prices"][1] / 10 ** 18, "Token": point["balances"][0] / 10 ** 18, "Skewed token": point["balances"][1] / 10 ** 18} for point in points],
            x="Skewed token",
            y="Token",
            title="Balances",
            hover_data=["price"],
        ).update_layout(
            yaxis_scaleanchor="x",  # <-- ensures 1:1 aspect ratio
            yaxis_scaleratio=1,
            xaxis=dict(range=[0, 1_000_000]),  # set both to start from 0
            yaxis=dict(range=[0, 1_000_000]),
            width=600,               # square figure
            height=600
        )
    )

    tokens_plot
    return


@app.cell
def _():
    # density_plot = mo.ui.plotly(
    #     px.line(
    #         [],
    #         x="price",
    #         y=["normal density", "density"],
    #         title="Skewed density",
    #     )
    # )

    # density_plot
    return


if __name__ == "__main__":
    app.run()
