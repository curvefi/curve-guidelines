## Oraclized
Pools with an oracle-priced asset (stETH/ETH-style). Result: single oracle update should be <= 2 * fee.

## LP oracle
LP price oracle: real LP price vs simplified `min(p) * virtual_price`. Result: simplified is slightly lower than real.
