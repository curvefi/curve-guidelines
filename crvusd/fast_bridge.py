import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import os
    import requests
    from web3 import Web3
    return Web3, mo, np, os, requests


@app.cell
def _():
    CHAINS = {
        "arbitrum": {
            "chainid": 42161,
            "rpc": "https://arb1.arbitrum.io/rpc",
            "defillama": "arbitrum",
            "crvusd": "0x498Bf2B1e120FeD3ad3D42EA2165E9b73f99C1e5",
            "pools": [
                "0xec090cf6DD891D2d014beA6edAda6e05E025D93d",  # crvUSD/USDC
                "0x73aF1150F265419Ef8a5DB41908B700C32D49135",  # crvUSD/USDT0
                "0x3aDf984c937FA6846E5a24E0A68521Bdaf767cE1",  # crvUSD/USDC.e
            ]
        },
        "optimism": {
            "chainid": 10,
            "rpc": "https://mainnet.optimism.io",
            "defillama": "op-mainnet",
            "crvusd": "0xC52D7F23a2e460248Db6eE192Cb23dD12bDDCbf6",
            "pools": [
                "0x03771e24b7C9172d163Bf447490B142a15be3485",  # crvUSD/USDC
                "0x05FA06D4Fb883F67f1cfEA0889edBff9e8358101",  # crvUSD/USDC.e
                "0xD1b30BA128573fcd7D141C8A987961b40e047BB6",  # crvUSD/USDT
            ]
        },
        "fraxtal": {
            "chainid": 252,
            "rpc": "https://rpc.frax.com",
            "defillama": "fraxtal",
            "crvusd": "0xB102f7Efa0d5dE071A8D37B3548e1C7CB148Caf3",
            "pools": [
                "0x63Eb7846642630456707C3efBb50A03c79B89D81",  # crvUSD/frxUSD
            ]
        }
    }
    return (CHAINS,)


@app.cell
def _(CHAINS, Web3, mo, requests):
    def get_total_supply(chain_name: str):
        ERC20_ABI = [
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function",
            },
        ]
        chain = CHAINS[chain_name]
        w3 = Web3(Web3.HTTPProvider(chain["rpc"]))
        if not w3.is_connected():
            raise ConnectionError(f"Could not connect to {chain_name} RPC.")

        contract = w3.eth.contract(address=Web3.to_checksum_address(chain["crvusd"]), abi=ERC20_ABI)

        supply_raw = contract.functions.totalSupply().call()
        decimals = contract.functions.decimals().call()

        return supply_raw / (10 ** decimals)

    # def calculate_std(ohlc_data: list, key: str) -> float:
    #     values = [float(item.get(key, 0)) for item in ohlc_data if key in item]
    #     return float(np.std(values, ddof=0))  # population std. dev.

    # def get_max_deviation(chain: str, pool: str, days: int):
    #     end_ts = datetime.utcnow().timestamp()
    #     start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    #     url = (
    #         f"https://prices.curve.finance/v1/lp_ohlc/"
    #         f"{chain}/{pool_address}"
    #         f"?agg_number=1&agg_units=day&start={start_ts}&end={end_ts}&price_units=usd"
    #     )
    #     resp = requests.get(url)
    #     resp.raise_for_status()
    #     data = resp.json().get("data", [])
    #     if len(data) < days:
    #         print(f"Warning: fetched only {len(data)} records for {days}-day period.")

    #     std_high = calculate_std(data, 'high')
    #     std_low = calculate_std(data, 'low')
    #     return std_high, std_low

    def get_defillama_metrics(chain_name: str):
        def _get(req: str):
            result = requests.get("https://api.llama.fi" + req)
            try:
                return float(result.text)
            except Exception:
                pass
            return 0.

        chain_data = CHAINS[chain_name]
        return {
            "tvl": _get(f"/tvl/{chain_data['defillama']}"),
            "bridge_tvl": _get(f"/tvl/{chain_data['defillama']}-bridge"),
        }


    def get_default_chain_metrics(chain_name: str):
        return {
            "arbitrum": {
                "tvl": 3_249_000_000.
            }
        }.get(chain_name, {})


    def get_metrics(chain_name: str):
        chain_metrics = get_defillama_metrics(chain_name)
        default_chain_metrics = get_default_chain_metrics(chain_name)
        format = lambda value, default_value: f"{value:_.0f}" if value else f"*{default_value:_.0f}*"
        return {
            "crvusd_tvl": format(get_total_supply(chain_name), None),
            "chain_tvl": format(chain_metrics["tvl"], default_chain_metrics.get('tvl', 0.)),
            "bridge_tvl": format(chain_metrics["bridge_tvl"], default_chain_metrics.get('bridge_tvl', 0.)), 
        }

    header = (
            "| Chain | crvUSD TVL | Chain TVL | Bridge TVL |\n"
            "|-------|------------|-----------|------------|\n"
        )

    rows = []
    for chain_name in CHAINS.keys():
        metrics = get_metrics(chain_name)
        row = (
            f"| {chain_name} "
            f"| {metrics['crvusd_tvl']} "
            f"| {metrics['chain_tvl']} "
            f"| {metrics['bridge_tvl']} |"
        )
        rows.append(row)

    mo.md(
        header + "\n".join(rows)
    )
    return (
        chain_name,
        get_default_chain_metrics,
        get_defillama_metrics,
        get_metrics,
        get_total_supply,
        header,
        metrics,
        row,
        rows,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ## Per-chain statistics\n
        Data is retrived from backend services, so would be updated on demand
        """
    )
    return


@app.cell
def _(CHAINS, mo):
    chain = mo.ui.dropdown(list(CHAINS.keys()), value="arbitrum", label="Chain")
    mo.md(f"""{chain}  \n""")
    return (chain,)


@app.cell
def _(__file__, chain, mo):
    from pathlib import Path

    NOTEBOOK_DIR = Path(__file__).parent

    price = mo.image(src=NOTEBOOK_DIR / "imgs" / chain.value / "price.png")
    volume = mo.image(src=NOTEBOOK_DIR / "imgs" / chain.value / "vol.png")
    tvl = mo.image(src=NOTEBOOK_DIR / "imgs" / chain.value / "tvl.png")

    mo.md(f"{price} {volume} {tvl}")
    return NOTEBOOK_DIR, Path, price, tvl, volume


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Parameters""")
    return


@app.cell
def _(mo):
    debt_ceiling = mo.ui.number(start=0, stop=10**8, step=1000, value=100_000, label="Total debt ceiling (mainnet)")
    mo.md(f"{debt_ceiling}")
    return (debt_ceiling,)


@app.cell
def _(debt_ceiling, mo):
    per_day = debt_ceiling.value // 5
    mo.md(
        f"Limit per day (L2): **{per_day:_}**  \n"
        f"overflow of {per_day * 7 - debt_ceiling.value:_} per week  \n"
    )
    return (per_day,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
