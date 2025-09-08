import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import pandas as pd
    from copy import copy
    from stableswap.simulation import StableSwap
    return StableSwap, copy, mo, px


@app.cell
def _(mo):
    n = mo.ui.number(start=2, stop=8, step=1, value=2, label="n")
    A = mo.ui.number(start=10, stop=100_000, step=1, value=200, label="A")

    D = 1_000_000 * 10 ** 18
    return A, D, n


@app.cell
def _(A, mo):
    mo.md(
        f"""
        {A}  \n
        """
    )
    return


@app.cell
def _(A, D, StableSwap, copy, mo, n, px):
    # Assuming p = price_oracle
    MAX_P_FACTOR = 1.25  # Factor to limit price changes
    dx = D // 10_000
    dy = 0

    pool = StableSwap(A.value, D, n.value, fee=0)
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
            "prices": [10 ** 18] + copy(p),
            "vp": pool.get_virtual_price(),
            "total_supply": copy(pool.tokens),
        })

    def real_lp_price(point):
        return sum([b * p // 10 ** 18 for p, b in zip(point["prices"], point["balances"])]) / point["total_supply"]

    def simplified_lp_price(point):
        min_p = min(point["prices"])
        return min_p * point["vp"] // 10 ** 18 / 10 ** 18

    plot = mo.ui.plotly(
        px.line(
            [{"price": point["prices"][1] / 10 ** 18, "Real": real_lp_price(point), "Simplified": simplified_lp_price(point)} for point in points],
            x="price",
            y=["Real", "Simplified"],
            title="LP Price",
        )
    )

    plot
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
