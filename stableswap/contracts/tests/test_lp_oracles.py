import boa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


WAD = 10**18
A_PRECISION = 10**2


@pytest.fixture(
    scope="module",
    params=[
        "stableswap/contracts/lp_oracle_bisection.vy",
        "stableswap/contracts/lp_oracle_secant.vy",
        "stableswap/contracts/lp_oracle_newton.vy",
        "stableswap/contracts/lp_oracle_brent.vy",
    ],
)
def oracle(request):
    return boa.load(request.param)


@settings(
    max_examples=1000,
)
@given(
    a_eff=st.integers(min_value=1, max_value=100_000 * A_PRECISION),
    p_int=st.integers(min_value=10**16, max_value=10**20),  # [0.01, 100.0] * WAD
)
def test_oracle_against_invariant(oracle, a_eff, p_int):
    x, y = oracle.internal._get_x_y(a_eff, p_int)

    # Invariant at D=1:
    # ((4A(x+y-WAD) + WAD) * 4xy) ?= WAD^3
    assert (4 * a_eff * (x + y - WAD) // A_PRECISION + WAD) * 4 * x * y == pytest.approx(WAD ** 3, rel=1e-10)

    # Marginal price from (x, y):
    # p_hat = (4A + 1/(4xy^2)) / (4A + 1/(4x^2y))
    term4a_wad = 4 * a_eff * WAD
    inv1 = A_PRECISION * (WAD ** 4) // (4 * x * y * y)
    inv2 = A_PRECISION * (WAD ** 4) // (4 * x * x * y)
    p_hat = ((term4a_wad + inv1) * WAD) // (term4a_wad + inv2)
    assert p_hat == pytest.approx(p_int, rel=1e-10)

# Keep this test purely functional against the closed-form checks.
