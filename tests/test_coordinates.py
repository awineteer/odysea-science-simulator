import pytest
import numpy as np

from odysim.coordinates import (
    llh_array_to_sch_array, sch_array_to_llh_array, WGS84
)


@pytest.fixture
def input_peg():
    peg_lat = np.array([45])
    peg_lon = np.array([45])
    peg_hdg = np.array([0])
    return peg_lat, peg_lon, peg_hdg

@pytest.fixture
def expected_llh():
    with np.load('./tests/data/expected_llh.npz', 'r') as data:
        expected_lat = data['lat']
        expected_lon = data['lon']
        expected_h = data['h']
    return expected_lat, expected_lon, expected_h


@pytest.fixture
def input_llh():
    lat = np.concat((np.linspace(44.6, 45, 50), np.linspace(45, 44.6, 50)))
    lon = np.linspace(54.5, 35.5, 100)
    h = np.concat((np.linspace(-150, 0, 50), np.linspace(0, -150, 50)))
    return lat, lon, h


@pytest.fixture
def expected_sch():
    with np.load('./tests/data/expected_sch.npz', 'r') as data:
        expected_s = data['s']
        expected_c = data['c']
        expected_h = data['h']
    return expected_s, expected_c, expected_h


@pytest.fixture
def input_sch():
    s = np.zeros(100)
    c = np.linspace(-1500000/2, 1500000/2, 100)
    h = np.zeros(100)
    return s, c, h


def test_llh_array_to_sch_array(input_llh, input_peg, expected_sch):
    lat, lon, h = input_llh
    peg_lat, peg_lon, peg_hdg = input_peg
    peg_local_radius = WGS84.local_radius(peg_hdg, peg_lat)
    s, c, h = llh_array_to_sch_array(lat, lon, h, peg_lat, peg_lon, peg_hdg, peg_local_radius)
    expected_s, expected_c, expected_h = expected_sch
    assert np.allclose(s, expected_s, atol=1e-6)
    assert np.allclose(c, expected_c, atol=1e-6)
    assert np.allclose(h, expected_h, atol=1e-6)


def test_sch_array_to_llh_array(input_sch, input_peg, expected_llh):
    s, c, h = input_sch
    peg_lat, peg_lon, peg_hdg = input_peg
    peg_local_radius = WGS84.local_radius(peg_hdg, peg_lat)
    lat, lon, h = sch_array_to_llh_array(s, c, h, peg_lat, peg_lon, peg_hdg, peg_local_radius)
    expected_lat, expected_lon, expected_h = expected_llh
    assert np.allclose(lat, expected_lat, atol=1e-6)
    assert np.allclose(lon, expected_lon, atol=1e-6)
    assert np.allclose(h, expected_h, atol=1e-6)
