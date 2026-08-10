"""
Tests for library_nmr.fitting — pseudo-Voigt lineshape fitting.

Strategy: build a synthetic spectrum from KNOWN pseudo-Voigt parameters,
then check that fit_group recovers those parameters. This is the most
useful kind of test for a fitting routine — it doesn't just check that
the code runs, it checks that the optimization actually converges to the
right physical answer.
"""

import numpy as np
import pytest

from library_nmr.fitting import (
    pseudo_voigt,
    sum_pseudo_voigt,
    fit_group,
    evaluate_fit_quality,
)


# --- pseudo_voigt: shape sanity checks -------------------------------------

def test_pseudo_voigt_pure_lorentzian_matches_formula():
    """eta=1 should give a pure Lorentzian."""
    x = np.linspace(-10, 10, 1000)
    A, nu0, FWHM = 5.0, 0.0, 2.0
    y = pseudo_voigt(x, A, nu0, FWHM, eta=1.0)
    lorentzian = (A / np.pi) * ((FWHM / 2) / (x**2 + (FWHM / 2) ** 2))
    assert np.allclose(y, lorentzian)


def test_pseudo_voigt_pure_gaussian_matches_formula():
    """eta=0 should give a pure Gaussian."""
    x = np.linspace(-10, 10, 1000)
    A, nu0, FWHM = 5.0, 0.0, 2.0
    y = pseudo_voigt(x, A, nu0, FWHM, eta=0.0)
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    gaussian = A * np.exp(-(x**2) / (2 * sigma**2))
    assert np.allclose(y, gaussian)


def test_pseudo_voigt_peak_is_at_nu0():
    x = np.linspace(-10, 10, 2001)
    y = pseudo_voigt(x, A=1.0, nu0=3.0, FWHM=2.0, eta=0.5)
    assert x[np.argmax(y)] == pytest.approx(3.0, abs=0.02)


# --- sum_pseudo_voigt -------------------------------------------------

def test_sum_pseudo_voigt_equals_manual_sum_of_two_peaks():
    x = np.linspace(-20, 20, 1000)
    params = [1e6, -3.0, 4.0, 0.8, 5e5, 3.0, 6.0, 0.2]
    expected = (pseudo_voigt(x, 1e6, -3.0, 4.0, 0.8)
                + pseudo_voigt(x, 5e5, 3.0, 6.0, 0.2))
    assert np.allclose(sum_pseudo_voigt(x, *params), expected)


# --- fit_group: recovers known ground truth -------------------------------

def test_fit_group_single_peak_recovers_parameters():
    """One clean, noise-free peak — the fit should recover the exact
    injected position/width/eta/amplitude to high precision."""
    x = np.linspace(-20, 20, 3000)
    true_A, true_nu0, true_FWHM, true_eta = 1e6, 1.2, 4.0, 0.7
    y = pseudo_voigt(x, true_A, true_nu0, true_FWHM, true_eta)

    results = fit_group(
        x, y, ppm_min=-20, ppm_max=20,
        p0_list=[[8e5, 1.0, 3.5, 0.5]],
    )
    r = results[0]
    assert r["position"] == pytest.approx(true_nu0, abs=1e-3)
    assert r["width"] == pytest.approx(true_FWHM, abs=1e-3)
    assert r["eta"] == pytest.approx(true_eta, abs=1e-3)
    assert r["amplitude"] == pytest.approx(true_A, rel=1e-3)


def test_fit_group_two_components_with_fixed_eta():
    """Two overlapping components, one with eta fixed (as used in the
    two-component T1/T2 relaxation scripts) — checks that the fixed eta
    is respected exactly and the free parameters still converge.

    NOTE: this also illustrates why `amplitude` (the fit parameter A) is
    NOT the same as `integral` (the real area under the curve) — see the
    warning in fit_group's docstring. Here the broad component has a
    smaller A (1e7 vs 3.5e7) but a LARGER integral, because its area
    scales with both A and width, and it is much wider (FWHM=15 vs 5.8).
    A naive comparison of amplitudes to estimate relative populations
    would get the ranking backwards — this is exactly the trap the
    `integral` field exists to avoid.
    """
    x = np.linspace(-30, 30, 4000)
    # narrow, mostly-Lorentzian component + broad, pure-Gaussian component
    A_narrow, FWHM_narrow, eta_narrow = 3.5e7, 5.8, 0.99
    A_broad, FWHM_broad, eta_broad = 1.0e7, 15.0, 0.0
    y = (pseudo_voigt(x, A_narrow, 0.6, FWHM_narrow, eta_narrow)
         + pseudo_voigt(x, A_broad, 0.6, FWHM_broad, eta_broad))

    results = fit_group(
        x, y, ppm_min=-23, ppm_max=27,
        p0_list=[[3e7, 0.5, 6, 0.9], [9e6, 0.5, 14, 0.5]],
        eta_fixed_list=[None, 0.0],
        width_bounds_list=[(0, 8), (8, 100)],
        position_bounds_list=[(-4, 5), (-4, 5)],
    )
    narrow, broad = results
    assert narrow["width"] == pytest.approx(FWHM_narrow, abs=0.05)
    assert broad["width"] == pytest.approx(FWHM_broad, abs=0.05)
    assert broad["eta"] == 0.0
    assert broad["eta_fixed"] is True
    assert narrow["eta_fixed"] is False

    # Ground truth for the integrals: since fit_group integrates only over
    # the fit window (-23, 27), not to infinity, we compare against a
    # numerical trapezoidal integral over that SAME window — the pure
    # Lorentzian component (eta=0.99) has slowly-decaying tails (~1/x²),
    # so a naive infinite-axis analytical formula would overestimate it
    # noticeably even for a "wide" window like this one.
    window_mask = (x >= -23) & (x <= 27)
    x_window = x[window_mask]
    expected_narrow = np.trapezoid(
        pseudo_voigt(x_window, A_narrow, 0.6, FWHM_narrow, eta_narrow), x_window)
    expected_broad = np.trapezoid(
        pseudo_voigt(x_window, A_broad, 0.6, FWHM_broad, eta_broad), x_window)
    assert narrow["integral"] == pytest.approx(expected_narrow, rel=0.02)
    assert broad["integral"] == pytest.approx(expected_broad, rel=0.02)
    # and confirm the "surprising" direction explicitly: broader peak wins
    # on integral despite smaller amplitude
    assert broad["integral"] > narrow["integral"]


def test_fit_group_width_bounds_prevent_role_swap():
    """Without disjoint width_bounds, a narrow/broad pair can swap roles
    during optimization. This test checks that providing bounds keeps
    each component in its intended range."""
    x = np.linspace(-30, 30, 4000)
    y = (pseudo_voigt(x, 3.5e7, 0.6, 5.8, 0.99)
         + pseudo_voigt(x, 1.0e7, 0.6, 15.0, 0.0))

    results = fit_group(
        x, y, ppm_min=-23, ppm_max=27,
        p0_list=[[3e7, 0.5, 6, 0.9], [9e6, 0.5, 14, 0.5]],
        width_bounds_list=[(0, 8), (8, 100)],
        position_bounds_list=[(-4, 5), (-4, 5)],
    )
    assert results[0]["width"] < 8
    assert results[1]["width"] >= 8


def test_fit_group_integral_sign_is_positive_on_reversed_axis():
    """Regression test: the ppm axis in the real pipeline is often
    decreasing (NMR convention), which previously caused a sign bug in
    the integral if delta_peak wasn't sorted before trapezoidal
    integration. Checks the fix holds regardless of axis direction."""
    x_increasing = np.linspace(-20, 20, 2000)
    x_decreasing = x_increasing[::-1].copy()
    y_increasing = pseudo_voigt(x_increasing, 1e6, 0.0, 4.0, 0.5)
    y_decreasing = pseudo_voigt(x_decreasing, 1e6, 0.0, 4.0, 0.5)

    r_inc = fit_group(x_increasing, y_increasing, -20, 20, [[8e5, 0, 3.5, 0.5]])[0]
    r_dec = fit_group(x_decreasing, y_decreasing, -20, 20, [[8e5, 0, 3.5, 0.5]])[0]

    assert r_inc["integral"] > 0
    assert r_dec["integral"] > 0
    assert r_inc["integral"] == pytest.approx(r_dec["integral"], rel=1e-3)


# --- evaluate_fit_quality -------------------------------------------------

def test_evaluate_fit_quality_zero_for_perfect_fit():
    x = np.linspace(-20, 20, 1000)
    y = pseudo_voigt(x, 1e6, 0.0, 4.0, 0.5)
    results = fit_group(x, y, -20, 20, [[9e5, 0.1, 3.8, 0.4]])
    rss, rms = evaluate_fit_quality(results)
    assert rss == pytest.approx(0.0, abs=1.0)  # essentially perfect (noise-free)
    assert rms == pytest.approx(0.0, abs=1.0)


def test_evaluate_fit_quality_nonzero_for_bad_fit():
    """A model that clearly doesn't match the data should have a much
    larger residual than a good fit — sanity check that the metric is
    actually discriminating, used e.g. in the eta-hypothesis comparison
    in pipeline_1d.py."""
    x = np.linspace(-20, 20, 1000)
    y = pseudo_voigt(x, 1e6, 0.0, 4.0, eta=1.0)  # true: pure Lorentzian

    good_fit = fit_group(x, y, -20, 20, [[9e5, 0.1, 3.8, 0.9]])
    bad_fit = fit_group(x, y, -20, 20, [[9e5, 0.1, 3.8, 0.0]], eta_fixed_list=[0.0])  # forced Gaussian

    _, rms_good = evaluate_fit_quality(good_fit)
    _, rms_bad = evaluate_fit_quality(bad_fit)
    assert rms_bad > rms_good
