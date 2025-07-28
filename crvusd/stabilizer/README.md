# Peg Keeper

## Addresses
- **PegKeeperRegulator**: [0x36a04CAffc681fa179558B2Aaba30395CDdd855f](https://etherscan.io/address/0x36a04CAffc681fa179558B2Aaba30395CDdd855f)
- **PegKeeperOffboarding**: [0x81813e506CdB58Cc2f2eD1619bE6383fc3699cA8](https://etherscan.io/address/0x81813e506CdB58Cc2f2eD1619bE6383fc3699cA8)
- **Price Aggregators**:
	- [0x18672b1b0c623a30089A280Ed9256379fb0E4E62](https://etherscan.io/address/0x18672b1b0c623a30089A280Ed9256379fb0E4E62)
	- (TBD)
- **Monetary policies**: fetch manually from curve.finance or via API/script(TBD).

## Prerequisites
1. Deploy a crvUSD StableSwap pool (see [Pool](#Pool) parameters).
2. Ensure pool TVL ≥ $1M (issuer allocation or incentivized). Must have multiple holders to reduce donation attack risk.
3. Obtain a risk review (e.g. from [Llama Risk](https://www.llamarisk.com)) and set a debt ceiling.
4. Update PKRegulator parameters if needed (see [Regulator](#Regulator)).

## Deployment
1. Deploy [PegKeeperV2](https://github.com/curvefi/curve-stablecoin/tree/master/contracts/stabilizer) or latest version.
2. Create a governance vote to:
	1. Add Peg Keeper to regulator via `PegKeeperRegulator.add_peg_keepers([new_pk])`
	2. Mint `debt_ceiling` of crvUSD to PK through `ControllerFactory.set_debt_ceiling(pk, amount)`.
	3. Update Regulator parameters (if required).
	4. Add to monetary policies.
	5. \[Optional] Add to price aggregator.

## Offboarding
Create a governance vote to:
	1. Remove PK from regulator via `PegKeeperRegulator.remove_peg_keepers([pk]).
	2. Add PK to `OffboardingRegulator.add_peg_keepers([pk]).
	3. Update regulator `PegKeeper.set_new_regulator(offboarding)`.
	4. Remove from monetary policies.
	5. Remove from price aggregators (if required).

## Parameters
Peg Keepers pair against 100% fiat-backed stablecoins.
### Pool
- crvUSD typically set as second coin.
- Conventional parameters:

| A    | fee   | off peg mul |
| :--- | :---- | :---------- |
| 2000 | 0.01% | 5           |

### Peg Keeper
- action_delay: **1 block (12s)**.
- caller_share: **20%** (simulation link TBD).
- debt_ceiling: set per risk vs. other stablecoins in Regulator.

### Regulator
(TBD) Formulas and parameters reasoning can be found [here](https://github.com/curvefi/curve-stablecoin-researches/tree/main/peg_keeper).
