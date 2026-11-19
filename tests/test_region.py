"""src.region: fixed-parameter region helper (exp 024)."""
import json

import numpy as np
import pytest

from src.region import Region, profile_over_region

H = 0.01


def _grid():
    re, im = np.meshgrid(np.arange(-0.3, -0.05 + 1e-9, H), np.arange(1.0, 1.2 + 1e-9, H), indexing="ij")
    tau = (re + 1j * im).ravel()
    chi2 = 5.0 + ((tau.real + 0.18) / 0.02) ** 2 + ((tau.imag - 1.1) / 0.03) ** 2
    return tau, chi2


def test_from_grid_level_and_complex_points():
    tau, chi2 = _grid()
    r = Region.from_grid(tau, chi2, level=6.18, step=H)
    assert r.names == ("re_tau", "im_tau") and r.points.shape[1] == 2
    assert r.chi2_min == pytest.approx(chi2.min())
    assert np.all(r.dchi2 <= 6.18) and len(r) == int(np.sum(chi2 <= chi2.min() + 6.18))
    assert r.best == pytest.approx([-0.18, 1.1], abs=1e-9)
    b = r.bounds()
    assert b["re_tau"][0] < -0.18 < b["re_tau"][1] and b["im_tau"][0] < 1.1 < b["im_tau"][1]
    assert r.restrict(2.30).level == 2.30 and len(r.restrict(2.30)) < len(r)


def test_contains_and_dchi2_at():
    tau, chi2 = _grid()
    r = Region.from_grid(tau, chi2, level=6.18, step=H)
    assert r.contains([-0.18, 1.1]) is True
    assert r.contains(-0.18 + 1.1j) is True
    assert r.contains([-0.18 + 0.004, 1.1 - 0.004]) is True
    assert r.contains([-0.30, 1.0]) is False
    assert r.dchi2_at([-0.18, 1.1]) == pytest.approx(0.0, abs=1e-9)
    assert r.dchi2_at([-0.30, 1.0]) == np.inf
    batch = r.contains(np.array([[-0.18, 1.1], [-0.3, 1.0]]))
    assert batch.tolist() == [True, False]
    assert r.dchi2_at(np.array([-0.18 + 1.1j, -0.3 + 1.0j]), outside=99.0)[1] == 99.0


def test_save_load_roundtrip(tmp_path):
    tau, chi2 = _grid()
    r = Region.from_grid(tau, chi2, level=2.30, step=H, meta={"source": "test"})
    p = tmp_path / "reg.json"
    r.save(p)
    d = json.loads(p.read_text())
    assert set(d) >= {"names", "points", "dchi2", "level", "chi2_min", "step"}
    r2 = Region.load(p)
    assert np.array_equal(r2.points, r.points) and np.array_equal(r2.dchi2, r.dchi2)
    assert r2.meta == {"source": "test"} and r2.contains([-0.18, 1.1])


def test_representatives_and_profile():
    tau, chi2 = _grid()
    r = Region.from_grid(tau, chi2, level=6.18, step=H)
    reps = r.representatives(5)
    assert reps.shape == (5, 2) and reps[0] == pytest.approx(r.best)
    # new-data chi2 prefers the region's lower-left edge
    f = lambda p: 100 * ((p[0] + 0.21) ** 2 + (p[1] - 1.05) ** 2) / 1e-3
    a = profile_over_region(f, r, add_dchi2=False)
    b = profile_over_region(f, r, add_dchi2=True)
    assert a["best_total"] == pytest.approx(a["best_chi2"])
    assert b["best_total"] == pytest.approx(b["best_chi2"] + b["best_dchi2"])
    assert b["best_total"] >= a["best_total"] - 1e-12
    assert r.contains(a["best_point"])
    with pytest.raises(ValueError):
        profile_over_region(f, r, points=np.array([[-0.3, 1.0]]))


def test_validation():
    with pytest.raises(ValueError):
        Region(np.zeros((0, 2)), np.zeros(0), ("a", "b"), 1.0)
    with pytest.raises(ValueError):
        Region(np.zeros((3, 2)), np.zeros(2), ("a", "b"), 1.0)
    r = Region(np.array([[0.0], [1.0]]), [0.0, 1.0], ("x",), 2.0)   # no step: nearest point
    assert r.dchi2_at([0.9]) == 1.0 and r.contains([5.0]) is True
