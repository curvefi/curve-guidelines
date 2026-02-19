# pragma version 0.4.3
"""
@title FXSwapLPOracle
@notice LP oracle for Twocrypto(FXSwap)-style pools.
@dev Reuses stable bisection solver and adjusts for pool internal price scaling.
"""

import lp_oracle_bisection


interface IFXSwap:
    def A() -> uint256: view
    def virtual_price() -> uint256: view
    def price_scale() -> uint256: view
    def price_oracle() -> uint256: view
    def D() -> uint256: view
    def totalSupply() -> uint256: view


interface PriceOracle:
    def price() -> uint256: view
    def price_w() -> uint256: nonpayable


PRECISION: constant(uint256) = 10**18
N_COINS: constant(uint256) = 2
POOL_A_PRECISION: constant(uint256) = 10_000

POOL: public(immutable(IFXSwap))
AGG: public(immutable(PriceOracle))


@deploy
def __init__(_pool: IFXSwap, _agg: PriceOracle):
    assert _pool.address != empty(address)
    assert staticcall _pool.A() >= POOL_A_PRECISION, "Bad A value"
    assert staticcall _pool.virtual_price() > 0
    assert staticcall _pool.price_scale() > 0
    assert staticcall _pool.price_oracle() > 0
    POOL = _pool

    assert _agg.address != empty(address)
    assert staticcall _agg.price() > 0
    assert extcall _agg.price_w() > 0
    AGG = _agg


@internal
@view
def _scaled_A_raw() -> uint256:
    # Pool stores A as: A_true * N_COINS**(N_COINS-1) * 10_000.
    # Secant solver expects: A_true * lp_oracle_secant.A_PRECISION.
    A_pool: uint256 = staticcall POOL.A()
    return unsafe_div(
        A_pool * lp_oracle_secant.A_PRECISION,
        N_COINS**(N_COINS-1) * POOL_A_PRECISION
    )


@internal
@view
def _scaled_price() -> uint256:
    # Pool invariant is computed on balances scaled by price_scale.
    # Convert oracle price into that scaled coordinate system.
    p_oracle: uint256 = staticcall POOL.price_oracle()
    p_scale: uint256 = staticcall POOL.price_scale()
    return unsafe_div(p_oracle * PRECISION, p_scale)


@view
@external
def portfolio_value() -> uint256:
    return self._portfolio_value()


@internal
@view
def _portfolio_value() -> uint256:
    return lp_oracle_bisection._portfolio_value_secant(self._scaled_A_raw(), self._scaled_price())


@internal
@view
def _lp_price_in_coin0() -> uint256:
    D: uint256 = staticcall POOL.D()
    total_supply: uint256 = staticcall POOL.totalSupply()
    return self._portfolio_value() * D // total_supply


@view
@external
def lp_price() -> uint256:
    return self._lp_price_in_coin0()


@view
@external
def price() -> uint256:
    return self._lp_price_in_coin0() * staticcall AGG.price() // PRECISION


@external
def price_w() -> uint256:
    return self._lp_price_in_coin0() * extcall AGG.price_w() // PRECISION
