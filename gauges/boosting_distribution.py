import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    n_lockers = 2  # Change if necessary
    lockers = [
        (
            mo.ui.number(start=0, stop=100, step=0.01, value=52.86 if i == 0 else 0, label="veCRV power (%)"),
            mo.ui.number(start=0, stop=100, step=0.01, value=17 if i == 0 else 100, label="fee (%)"),
            mo.ui.number(start=0, stop=100, step=0.01, value=0, label="gauge share (%)"),
        )
        for i in range(n_lockers)
    ]
    my_vecrv = mo.ui.number(start=0, stop=100, step=0.01, value=0, label="My veCRV share")
    my_share = mo.ui.number(start=0, stop=100, step=0.01, value=0, label="My gauge share (%)")
    mo.md(
        '  \n'.join([
            "veCRV balances be checked in [explorer](https://etherscan.io/token/0x5f3b5DfEb7B28CDbD7FAba78963EE202a494e2A2#balances) or dedicated [L2 balances page](https://curvefi.github.io/storage-proofs/). Update values wrt your case, increase *n* in the code slot if needed.  ",
            "",
            "| Locker   | Common Fees | veCRV Balance |  ",
            "|:---------|-------------|---------------|  ",
            "| Convex   | 17%         | 52.86%        |  ",
            "| StakeDAO | 15%         | 14.84%        |  ",
            "| Yearn    | 10%         | 10.48%        |  ",
            "\n",
            *[f"Locker {i}: {locker[0]} {locker[1]} {locker[2]}" for i, locker in enumerate(lockers)],
            f"{my_vecrv} {my_share}",
        ])
    )
    return lockers, my_share, my_vecrv


@app.cell
def _(lockers, mo, my_share, my_vecrv, np):
    from typing import List, Tuple
    import math
    from scipy.optimize import minimize


    def working_balance(lp_share: float, ve_share: float):
        return min(lp_share, 0.4 * lp_share + 0.6 * ve_share)

    def my_profit(lockers: List[Tuple[float, float, float]], distribution: List[float], log: bool=False):
        """
        lockers: list of tuples (lp_i, ve_i, fee_i)
            lp_i   = amount of lp_share already put into locker
            ve_i  = veCRV share of this destination
            fee_i = fee fraction in [0,1]
        distribution: list of floats d_i to add to each locker

        Returns:
            total profit (fraction of gauge rewards after fees)
        """
        assert len(lockers) == len(distribution)

        total_lp = 0
        total_ve = 0
        working_total_supply = 0
        total_w_b = 0
        working_balances = []
        for (ve, fee, lp), d in zip(lockers, distribution):
            total_lp += lp + d
            total_ve += ve
            w_b = working_balance(lp + d, ve)
            working_total_supply += w_b
            my_w_b = d / (lp + d) * w_b if lp + d > 0 else 0
            working_balances.append(my_w_b)
            total_w_b += my_w_b * (1 - fee)
        assert total_lp < 1.001, "Bad sum of gauge shares"
        assert total_ve < 1.001, "Bad sum of ve"

        # Add not reported users as if they don't have boost
        if total_lp < 1:
            working_total_supply += 0.4 * (1 - total_lp)

        if log:
            for i, wb in enumerate(working_balances):
                print(f"Working balance: {wb}, profit: {wb / working_total_supply}")

        return total_w_b / working_total_supply

    def calc_distribution(lockers: List[Tuple[float, float, float]], lp_share_to_distribute: float):
        n = len(lockers)

        def objective(d):
            return -my_profit(lockers, d)  # negative because minimize() wants minimum

        # constraint: sum(d) = lp_share_to_distribute
        cons = {'type': 'eq', 'fun': lambda d: np.sum(d) - lp_share_to_distribute}

        # bounds: d_i >= 0
        bnds = [(0, lp_share_to_distribute) for _ in range(n)]

        # start with equal split
        x0 = np.array([lp_share_to_distribute / n] * n)
        res = minimize(objective, x0, bounds=bnds, constraints=cons, method="SLSQP", options={"maxiter": 5000})
        # Solver might stall around nondifferentiable points and find not optimal solution
        return res.x, -res.fun

    _lockers = [(l[0].value / 100, l[1].value / 100, l[2].value / 100) for l in lockers]
    _lockers.append((0, my_vecrv.value / 100, 0))  # self locker

    lp_share_to_distribute = my_share.value / 100

    distribution, profit = calc_distribution(_lockers, lp_share_to_distribute)
    my_profit(_lockers, distribution, log=True)
    distribution, self_alloc = distribution[:-1], distribution[-1]
    mo.md(
        '  \n'.join([
            "Your optimal allocation is (TODO: change solver, might stall in not optimal solution):",
            *[f"Locker {i}: {100 * a:.2f}%" for i, a in enumerate(distribution)],
            f"Not locked: {100 * self_alloc:.2f}%",
            f"Share of total gauge profit: {100 * profit:.2f}%",
        ])
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
