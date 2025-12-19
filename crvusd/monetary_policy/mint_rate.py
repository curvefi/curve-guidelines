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

    - **Plateau Original (divisive form)**:  
      A locally flattened version of the original curve around the target,
      implemented by dividing the exponent by `1 + k · g(x)`.

    - **Plateau Original (multiplicative form)**:  
      A locally flattened version of the original curve around the target,
      implemented by multiplying the exponent by `1 − k · g(x)`.

    Here, `g(x)` is a smooth Gaussian-shaped bump centered at the target
    debt fraction, controlling how strongly the curve is flattened locally.

    ---

    ## Parameters

    The models are controlled by the following interactive parameters:

    - **`r0`** — base rate (value at zero debt)
    - **`target (t)`** — target debt fraction
    - **`k`** — strength of the local flattening effect
    - **`eps`** — smoothing parameter (reserved for soft-distance variants)
    - **`sigma`** — width of the region around the target where flattening applies
    """
    )
    return


@app.cell
def _(mo):
    r0 = mo.ui.number(value=1.0, step=0.01, label="r0 (base rate)")
    target = mo.ui.slider(0.01, 0.99, value=0.40, step=0.01, label="target debt fraction (t)")
    k = mo.ui.slider(0.005, 0.50, value=0.05, step=0.005, label="k (tail width)")

    eps = mo.ui.slider(1e-12, 1e-2, value=1e-6, step=1e-6, label="eps (smoothing)")
    sigma = mo.ui.slider(0.01, 1.0, value=0.25, step=0.01, label="sigma (gaussian width)")

    mo.vstack([
        mo.md("## Parameters"),
        mo.callout(
            mo.vstack([
                mo.hstack([r0, target, k]),
                mo.hstack([eps, sigma]),
            ]),
            kind="neutral",
        ),
    ])
    return k, r0, sigma, target


@app.cell
def _(k, math, r0, sigma, target):
    # Original: r0 * exp(-x/t)
    def rate_original(x):
        r0v = float(r0.value)
        tv  = float(target.value)
        return r0v * math.exp(-(x / tv))


    # # Soft original: r0 * exp(-sqrt((x/t)^2 + eps))
    # # (soft around 0, but close to original)
    # def rate_soft_original(x):
    #     r0v  = float(r0.value)
    #     tv   = float(target.value)
    #     epsv = float(eps.value)
    #     return r0v * math.exp(-math.sqrt((x / tv) ** 2 + epsv))

    # 3) Plateau original (divided by 1 + k*g), g — a "bump" around the target
    def rate_plateau_original(x):
        r0v = float(r0.value)
        tv  = float(target.value)
        kv  = float(k.value)
        sig = float(sigma.value)

        g = math.exp(-((x - tv) / sig) ** 2)
        return r0v * math.exp(-(x / tv) / (1.0 + kv * g))


    # 4) Plateau original (multiplicative slope): exp(-(x/t)*(1 - k*g))
    def rate_plateau_original2(x):
        r0v = float(r0.value)
        tv  = float(target.value)
        kv  = float(k.value)
        sig = float(sigma.value)

        g = math.exp(-((x - tv) / sig) ** 2)
        return r0v * math.exp(-(x / tv) * (1.0 - kv * g))
    return rate_original, rate_plateau_original, rate_plateau_original2


@app.cell
def _(np, pd, rate_original, rate_plateau_original, rate_plateau_original2):
    xs = np.linspace(0.0, 1.0, 2001)

    rows = []

    for x in xs:
        rows.append({
            "debt_fraction": x,
            "original":            rate_original(x),
            # "soft_original":       rate_soft_original(x),
            "plateau_original":    rate_plateau_original(x),
            "plateau_original_2":  rate_plateau_original2(x),
        })

    df = pd.DataFrame(rows)
    return (df,)


@app.cell
def _(df, px):
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
    fig.show()
    return (df_long,)


@app.cell
def _(df_long, px):
    fig_log = px.line(
        df_long,
        x="debt_fraction",
        y="rate",
        color="model",
        log_y=True,
    )
    fig_log.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
