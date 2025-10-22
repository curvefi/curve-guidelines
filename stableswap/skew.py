import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Skewed oracle
    ## About
    There was an idea of setting a skewed oracle in a pool to get imbalanced proportion of tokens in the pools.
    This will make one side more dense than the other. This notebook contains simulation and graphs of such setup.

    ## Results
    It makes no sense since the pool becomes imbalanced w.r.t. price and not the balances(see density graph).
    In order to achieve liquidity density ratio one should alter StableSwap invariant: 

    $$
    A n^n \sum a_i x_i + D = A D n^n + \frac{D^{n+1}}{n^n \prod x_i^{a_i}},
    where \sum a_i = 1
    $$
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import math
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from copy import copy
    from stableswap.simulation import StableSwap
    return StableSwap, copy, go, mo, np, px


@app.cell
def _(mo):
    n = mo.ui.number(start=2, stop=8, step=1, value=2, label="n")
    A = mo.ui.number(start=10, stop=100_000, step=1, value=200, label="A")
    skew = mo.ui.number(start=0, stop=4, step=0.0001, value=0.5, label="skew")

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
def _(A, D, StableSwap, copy, n, skew):
    # Assuming p = price_oracle
    MAX_P_FACTOR = 2.15  # Factor to limit price changes
    dx = D // 20_000
    dy = 0

    skewed_prices = [10 ** 36 // int((1 + skew.value) * 10 ** 18), int((1 + skew.value) * 10 ** 18)]
    pool = StableSwap(A.value, D, n.value, p=skewed_prices, fee=0)
    _p = pool.get_p()
    while _p[0] / 10 ** 18 < MAX_P_FACTOR and 10 ** 18 / _p[0] < MAX_P_FACTOR:
        dy = pool.exchange(0, 1, dx)
        _p = pool.get_p()
    pool.exchange(1, 0, dy)  # go 1 step back
    _p = pool.get_p()

    points = []

    while _p[0] / 10 ** 18 < MAX_P_FACTOR and 10 ** 18 / _p[0] < MAX_P_FACTOR:
        pool.exchange(1, 0, dx)
        _p = pool.get_p()
        points.append({
            "balances": copy(pool.x),
            "raw_prices": [10 ** 18] + _p,
            "adjusted_prices": [p * _p // 10 ** 18 for p, _p in zip([10 ** 18] + _p, pool.p)],
            "effective_prices": [10 ** 18, pool.dy(1, 0, 10 ** 18)],
        })
    price_used = "effective_prices"
    return points, price_used


@app.cell(hide_code=True)
def _(go, mo, np, points, price_used):
    token_color = "blue"
    skewed_color = "orange"

    mid_price = []
    density_token, density_skewed = [], []
    price_from, price_to = [], []
    bps_span = []
    for i in range(len(points) - 1):
        p0 = points[i][price_used][1] / 1e18
        p1 = points[i+1][price_used][1] / 1e18

        dP = abs(p1 - p0)
        Pmid = 0.5 * (p0 + p1)
        dP_bp = (dP / Pmid) * 1e4 if Pmid > 0 else None

        pt_bal_0 = points[i]["balances"]
        pt_bal_1 = points[i + 1]["balances"]

        dQ_token = abs(pt_bal_1[0] - pt_bal_0[0]) / 10**18
        dQ_skewed = abs(pt_bal_1[1] - pt_bal_0[1]) / 10**18

        mid_price.append(Pmid)
        density_token.append(dQ_token / dP_bp if dP_bp and dP_bp > 0 else 0.0)
        density_skewed.append(dQ_skewed / dP_bp if dP_bp and dP_bp > 0 else 0.0)
        price_from.append(min(p0, p1))
        price_to.append(max(p0, p1))
        bps_span.append(dP_bp)

    center = (0, 0)
    for p, d in zip(mid_price, density_skewed):
        if center[1] < d:
            center = (p, d)

    def smoothen(mid_price, density_token, density_skewed, center_p):
        # --- build a smoothed series on an even grid (no pandas/scipy) ---
        def moving_avg(arr, k):
            # centered moving average with edge handling
            n = len(arr)
            half = k // 2
            out = np.empty(n)
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                out[i] = np.mean(arr[lo:hi])
            return out

        # ensure x is sorted (just in case)
        order = np.argsort(mid_price)
        x = np.array([mid_price[i] for i in order], dtype=float)
        y0 = np.array([density_token[i] for i in order], dtype=float)
        y1 = np.array([density_skewed[i] for i in order], dtype=float)

        # grid within your visible range
        xmin, xmax = center_p * 0.9, center_p * 1.1
        grid = np.linspace(xmin, xmax, 401)

        # piecewise-linear interpolation; clamp outside range
        eps = 1e-12  # avoid log(0)
        y0g = np.interp(grid, x, y0, left=y0[0], right=y0[-1])
        y1g = np.interp(grid, x, y1, left=y1[0], right=y1[-1])

        # smooth in log space (works nicely with yaxis_type="log")
        win =  nine = 9  # odd window size; tweak for more/less smoothing
        y0_smooth = np.exp(moving_avg(np.log(y0g + eps), nine))
        y1_smooth = np.exp(moving_avg(np.log(y1g + eps), nine))
        return grid, y0_smooth, y1_smooth

    grid, y0_smooth, y1_smooth = smoothen(mid_price, density_token, density_skewed, center[0])

    customdata = list(zip(price_from, price_to, bps_span))

    density = go.Figure()

    density.add_bar(
        x=mid_price,
        y=density_token,
        name="Token",
        marker_color=token_color,
        opacity=0.5,
        customdata=customdata,
        hovertemplate=(
            "<b>Token</b><br>"
            "density=%{y:.6f} token/bp<br>"
            "interval=[%{customdata[0]:.6f}, %{customdata[1]:.6f}]<br>"
            "bps span=%{customdata[2]:.3f}<extra></extra>"
        ),
    )
    density.add_scatter(
        x=grid, y=y0_smooth, name="Token (smoothed)", mode="lines",
        line=dict(color=token_color, width=2),
        legendgroup="Token",
        showlegend=False,
        hovertemplate="density=%{y:.6f} token/bp<extra></extra>"
    )

    density.add_bar(
        x=mid_price,
        y=density_skewed,
        name="Skewed token",
        marker_color=skewed_color,
        opacity=0.5,
        customdata=customdata,
        hovertemplate=(
            "<b>Skewed token</b><br>"
            "density=%{y:.6f} token/bp<br>"
            "interval=[%{customdata[0]:.6f}, %{customdata[1]:.6f}]<br>"
            "bps span=%{customdata[2]:.3f}<extra></extra>"
        ),
    )
    density.add_scatter(
        x=grid, y=y1_smooth, name="Skewed token (smoothed)", mode="lines",
        line=dict(color=skewed_color, width=2),
        legendgroup="Skewed token",
        showlegend=False,
        hovertemplate="density=%{y:.6f} token/bp<extra></extra>"
    )

    density.update_layout(
        barmode="overlay",  # overlay with transparency
        bargap=0,
        title="Liquidity Density Profile (Both Tokens)",
        xaxis_title="Price",
        yaxis_title="Liquidity density (token per 1 bp)",
        hovermode="x unified",
        xaxis=dict(range=[center[0] * 0.9, center[0] * 1.1]),
        # xaxis_type="log",
    )
    mo.ui.plotly(density)
    return


@app.cell(hide_code=True)
def _(mo, points, price_used, px):
    data = [
        {
            "price": point[price_used][1] / 10**18,
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
    return


@app.cell(hide_code=True)
def _(mo, points, price_used, px):
    balances_plot = mo.ui.plotly(
        px.line(
            [{"price": point[price_used][1] / 10 ** 18, "token balance": point["balances"][0] / 10 ** 18, "skewed balance": point["balances"][1] / 10 ** 18} for point in points],
            x="price",
            y=["token balance", "skewed balance"],
            title="Balances",
        )
    )

    balances_plot
    return


@app.cell(hide_code=True)
def _(mo, points, price_used, px):
    tokens_plot = mo.ui.plotly(
        px.line(
            [{"price": point[price_used][1] / 10 ** 18, "Token": point["balances"][0] / 10 ** 18, "Skewed token": point["balances"][1] / 10 ** 18} for point in points],
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
    return


if __name__ == "__main__":
    app.run()
