import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    import marimo as mo

    import pandas as pd
    import math
    return math, mo, np, pd, px


@app.cell
def _(mo):
    mo.md(
        """
    # Rate vs Debt Fraction — Function Comparison (Plotly)

    This interactive **marimo** notebook uses **Plotly** to visualize interest rate
    functions as a function of the debt ratio.

    It plots the resulting **`rate` as a function of
    `debt_fraction = debt / debt_cap`** for the following models:

    - **Original**:  
      `rate = r0 · exp(-x / t)`

    - **Squashed (power + weight)**:  
        The squashed model first transforms the debt fraction toward the target and then applies the same exponential decay.

        1. **Power squash transformation**  
            Given a target `t` in (0,1) and a power parameter `p >= 1`:

            ```
            if x <= t:
              s(x) = t - t * ((t - x) / t)^p
            else:
              s(x) = t + (1 - t) * ((x - t) / (1 - t))^p
            ```
        
            This keeps `x = 0`, `x = t`, and `x = 1` fixed while pulling intermediate values closer to the target.
            Higher `p` causes stronger concentration near the target.

        2. **Weighted blend between identity and squash**  
            Let `weight` be in `[0,1]`:
            ```
            x_squash = (1 - weight) * x + weight * s(x)
            ```
            - `weight = 0` → no squashing (`x_squash = x`)
            - `weight = 1` → full squashing (`x_squash = s(x)`)

        3. **Final squashed rate**  
            ```
            rate_squashed(x) = r0 * exp(-(x_squash / t))
            ```
    ---

    ## Parameters

    The models are controlled by the following interactive parameters:

    - **`r0`** — base rate (value at zero debt)
    - **`target (t)`** — target debt fraction
    - **`squash_power`** — exponent `p` controlling shape  
    (`1 = identity`, `2 = quadratic`, higher = stronger contraction near target)
    - **`squash_weight`** — blending weight  
    (`0 = original`, `1 = full squash`, values in between interpolate smoothly)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Implementation""")
    return


@app.cell
def _(mo):
    r0 = mo.ui.number(value=1.0, step=0.01, label="r0 (base rate)")
    target = mo.ui.slider(0.01, 0.99, value=0.40, step=0.01, label="target debt fraction (t)")
    squash_power = mo.ui.slider(
        0.0, 10.0,
        value=2.0,
        step=0.1,
        label="squash power (1 = original, 2 = square)"
    )
    squash_weight = mo.ui.slider(
        0.0, 1.0,
        value=1.0,
        step=0.01,
        label="squash weight (0 = original, 1 = full squash)"
    )
    return r0, squash_power, squash_weight, target


@app.cell
def _(math, r0, squash_power, squash_weight, target):
    # Original: r0 * exp(-x/t)
    def rate_original(x):
        r0v = float(r0.value)
        tv  = float(target.value)
        return r0v * math.exp(-(x / tv))

    def rate_squashed(x):
        tv  = float(target.value)
        w = float(squash_weight.value)
        p = float(squash_power.value)

        # power-based squash around target
        def squash(x):
            if x <= tv:
                # left side: normalize to [0,1], p-power, map back
                y  = (tv - x) / tv  # in [0,1]
                yp = y ** p
                return tv - yp * tv
            else:
                # right side: normalize to [0,1], p-power, map back
                z  = (x - tv) / (1.0 - tv)  # in [0,1]
                zp = z ** p
                return tv + zp * (1.0 - tv)
        xs = squash(x)
        # convex combination: x' = (1-w)*x + w*xs
        xs = (1.0 - w) * x + w * xs
        return rate_original(xs)
    return rate_original, rate_squashed


@app.cell
def _(np, pd, rate_original, rate_squashed):
    xs = np.linspace(0.0, 1.0, 2001)

    rows = []

    for x in xs:
        rows.append({
            "debt_fraction": x,
            "original":      rate_original(x),
            "rate_squashed": rate_squashed(x),
        })

    df = pd.DataFrame(rows)
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## Graphs""")
    return


@app.cell
def _(df, mo, px, r0, squash_power, squash_weight, target):
    df_long = df.melt(
        id_vars="debt_fraction",
        var_name="model",
        value_name="rate",
    )


    fig = px.line(
        df_long,
        x="debt_fraction",
        y="rate",
        color="model",
    )
    controls = mo.vstack([
        mo.hstack([
            mo.vstack([r0, target]),
            mo.vstack([squash_power, squash_weight]),
        ]),
    ])

    mo.hstack([
        mo.callout(mo.vstack([r0, target, squash_power, squash_weight]), kind="neutral"),
        fig,
    ])
    return (df_long,)


@app.cell
def _(df_long, mo, px, r0, squash_power, squash_weight, target):
    fig_log = px.line(
        df_long,
        x="debt_fraction",
        y="rate",
        color="model",
        log_y=True,
    )
    mo.hstack([
        mo.callout(mo.vstack([r0, target, squash_power, squash_weight]), kind="neutral"),
        fig_log,
    ])
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Historical rate graph
    TODO: Data
    """
    )
    return


@app.cell
def _(pd, px):
    # load
    pk_debt_df = pd.read_csv("crvusd/monetary_policy/data/pk_debts.csv")

    # ensure datetime parses correctly
    pk_debt_df["dt"] = pd.to_datetime(
        pk_debt_df["dt"],
        format="%B %d, %Y, %I:%M %p",
    )

    # convert keeper_debt to float (strip commas if needed)
    pk_debt_df["keeper_debt"] = (
        pk_debt_df["keeper_debt"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # # load controller debt
    # controller_df = pd.read_csv("crvusd/monetary_policy/data/controller_debts.csv")
    # controller_df["dt"] = pd.to_datetime(
    #     controller_df["dt"],
    #     format="%B %d, %Y, %I:%M %p",
    # )
    # controller_df["controller_debt"] = (
    #     controller_df["controller_debt"]
    #     .astype(str)
    #     .str.replace(",", "", regex=False)
    #     .astype(float)
    # )

    # # merge and compute ratio
    # pk_debt_df = pk_debt_df.merge(
    #     controller_df[["dt", "controller_debt"]],
    #     on="dt",
    #     how="left",
    # )
    # pk_debt_df["controller_debt"] = pk_debt_df["controller_debt"].ffill()
    pk_debt_df["keeper_ratio"] = pk_debt_df["keeper_debt"] / 100_000_000

    pk_debt_fig = px.line(
        pk_debt_df,
        x="dt",
        y="keeper_ratio",
        title="PK Debt Ratio Over Time",
        markers=True,
    )

    pk_debt_fig.show()
    return (pk_debt_df,)


@app.cell
def _(
    mo,
    pk_debt_df,
    px,
    r0,
    rate_original,
    rate_squashed,
    squash_power,
    squash_weight,
    target,
):
    pk_debt_df["rate_original"] = pk_debt_df["keeper_ratio"].apply(rate_original)
    pk_debt_df["rate_squashed"] = pk_debt_df["keeper_ratio"].apply(rate_squashed)

    comparison_rates = px.line(
        pk_debt_df,
        x="dt",
        y=["rate_original", "rate_squashed"],
        title="PK -> Rate (Original vs Squashed)",
        markers=False,
    )
    mo.hstack([
        mo.callout(mo.vstack([r0, target, squash_power, squash_weight]), kind="neutral"),
        comparison_rates,
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
