# /// script
# dependencies = ["curve-voting-lib @ git+https://github.com/curvefi/curve-voting-lib.git"]
# ///

import os

import boa
from voting import BrowserEnv, OWNERSHIP, vote


ETHERSCAN_API_KEY = os.environ["ETHERSCAN_API_KEY"]
RPC_URL = os.environ["RPC_URL"]
LIVE = False

POOL = "0x..."
PEG_KEEPER = "0x..."
DEBT_CEILING = 3_000_000 * 10**18
CALLER_SHARE = 50_000

FACTORY = "0xC9332fdCB1C491Dcc683bAe86Fe3cb70360738BC"
REGULATOR = "0x36a04CAffc681fa179558B2Aaba30395CDdd855f"
MONETARY_POLICIES = [
    # "0x8c5A7F011f733fBb0A6c969c058716d5CE9bc933",  # wBTC, tBTC, cbBTC, ETH
    # "0x8D76F31E7C3b8f637131dF15D9b4a3F8ba93bd75",  # LBTC, wstETH, sfrxETH v2, weETH
    # "0xc684432FD6322c6D58b6bC5d28B18569aA0AD0A1",  # sfrxETH v1
]
AGGREGATE_ORACLES = ["0x18672b1b0c623a30089A280Ed9256379fb0E4E62"]
DESCRIPTION = "[crvUSD] Onboard <coin> Peg Keeper. <governance post>"


def main():
    boa.fork(RPC_URL)
    pk = boa.from_etherscan(PEG_KEEPER, api_key=ETHERSCAN_API_KEY)
    factory = boa.from_etherscan(FACTORY, api_key=ETHERSCAN_API_KEY)
    regulator = boa.from_etherscan(REGULATOR, api_key=ETHERSCAN_API_KEY)
    pool = boa.from_etherscan(POOL, api_key=ETHERSCAN_API_KEY)
    aggregate_oracles = [
        boa.from_etherscan(address, api_key=ETHERSCAN_API_KEY)
        for address in AGGREGATE_ORACLES
    ]

    with vote(
        OWNERSHIP,
        DESCRIPTION,
        live_env=BrowserEnv() if LIVE else None,
    ):
        pk.set_new_caller_share(CALLER_SHARE)
        factory.set_debt_ceiling(pk, DEBT_CEILING)
        regulator.add_peg_keepers([pk])
        for policy in MONETARY_POLICIES:
            boa.from_etherscan(policy, api_key=ETHERSCAN_API_KEY).add_peg_keeper(pk)
        for oracle in aggregate_oracles:
            oracle.add_price_pair(pool)


if __name__ == "__main__":
    main()
