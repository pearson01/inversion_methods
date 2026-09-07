import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic Generator for tests that need reproducible sampling."""
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def seed_global_numpy_random():
    """
    Several bristau samplers (kappa_x.sample_kappa, sigma_rep.sample_sigma2_rep)
    draw from the *global* numpy random state rather than an injected
    Generator. Reseed it before every test so results are reproducible and
    tests can't leak randomness into one another.
    """
    np.random.seed(0)
    yield
