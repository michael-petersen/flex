import pytest
import numpy as np

import flex

def test_flex_initialization():
    # Test valid initialization
    rscl = 1.0
    mmax = 3
    nmax = 5
    R = np.array([0.1, 0.2, 0.3])
    phi = np.array([0.0, np.pi/4, np.pi/2])
    mass = np.array([1.0, 2.0, 3.0])
    velocity = np.array([10.0, 20.0, 30.0])

    flex_instance = flex.FLEX(rscl, mmax, nmax, R, phi, mass, velocity)
    assert flex_instance.rscl == rscl
    assert flex_instance.mmax == mmax
    assert flex_instance.nmax == nmax
    np.testing.assert_array_equal(flex_instance.R, R)
    np.testing.assert_array_equal(flex_instance.phi, phi)
    np.testing.assert_array_equal(flex_instance.mass, mass)
    np.testing.assert_array_equal(flex_instance.velocity, velocity)

    # Test invalid rscl type
    with pytest.raises(ValueError):
        flex.FLEX("invalid_rscl", mmax, nmax, R, phi, mass, velocity)

    # Test negative mmax
    with pytest.raises(ValueError):
        flex.FLEX(rscl, -1, nmax, R, phi, mass, velocity)