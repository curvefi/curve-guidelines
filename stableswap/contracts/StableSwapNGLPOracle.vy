# pragma version 0.4.3
"""
@title StableSwapNGLPOracle
@author Curve.Fi
@license MIT
@notice LP oracle for StableSwap-NG (n=2) reusing lp_oracle_bisection math.
"""

import lp_oracle_bisection


interface IStableSwapNG:
    def A_precise() -> uint256: view
    def get_virtual_price() -> uint256: view
    def price_oracle(i: uint256) -> uint256: view


PRECISION: constant(uint256) = 10**18
N_COINS: constant(uint256) = 2
POOL_A_PRECISION: constant(uint256) = 100

POOL: public(immutable(IStableSwapNG))


@deploy
def __init__(_pool: IStableSwapNG):
    assert _pool.address != empty(address)

    # Basic sanity checks for required pool methods.
    assert staticcall _pool.get_virtual_price() > 0
    assert staticcall _pool.price_oracle(0) > 0
    assert staticcall _pool.A_precise() > POOL_A_PRECISION, "Bad A value"
    success: bool = False
    response: Bytes[32] = b""
    success, response = raw_call(
        _pool.address,
        abi_encode(convert(2, uint256), method_id=method_id("coins(uint256)")),
        max_outsize=32,
        revert_on_failure=False
        )
    assert not success, "Supports only 2-coin pool"

    POOL = _pool


@internal
@view
def _scaled_A_raw() -> uint256:
    # Pool stores A as: A_true * N_COINS**(N_COINS-1) * 100.
    # Secant solver expects: A_true * lp_oracle_secant.A_PRECISION.
    A_pool: uint256 = staticcall POOL.A_precise()
    return unsafe_div(
        A_pool * lp_oracle_secant.A_PRECISION,
        N_COINS**(N_COINS-1) * POOL_A_PRECISION
    )


@internal
@view
def _portfolio_value() -> uint256:
    return lp_oracle_bisection._portfolio_value_secant(self._scaled_A_raw(), staticcall POOL.price_oracle(0))


@internal
@view
def _lp_price_in_coin0() -> uint256:
    return unsafe_div(self._portfolio_value() * staticcall POOL.get_virtual_price(), PRECISION)


@external
@view
def lp_price() -> uint256:
    return self._lp_price_in_coin0()
