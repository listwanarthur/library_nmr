"""
Tests for the NNLS-based fixed-shape amplitude fitting in
relaxation_T2_components.py.

Strategy (same idea as test_fitting.py): build a synthetic spectrum from
a KNOWN sum of pseudo-Voigt components with known amplitudes, then check
that fit_amplitudes_fixed_shape recovers exactly those amplitudes. This
is the strongest kind of test for a fitting function — it checks the
actual numerical answer, not just that the code runs.
"""

import numpy as np
import pytest

from library_nmr.fitting import pseudo_voigt
from library_nmr.relaxation_T2_components import (
    fit_amplitudes_fixed_shape,
    monoexp_decay,
)


# Two fixed lineshapes representative of the narrow/broad ⁷Li components
# used throughout the library (same values as pipeline_1d.py's PEAKS).
NARROW = {"position": 0.6, "width": 5.8, "eta": 0.99}
BROAD = {"position": 0.6, "width": 15.0, "eta": 0.0}


def _synthetic_spectrum(amps, components, x=None):
    """Builds delta, signal for a spectrum that is EXACTLY the sum of the
    given components at the given amplitudes — the known ground truth
    fit_amplitudes_fixed_shape should recover."""
    if x is None:
        x = np.linspace(-30, 30, 2000)
    y = sum(pseudo_voigt(x, a, c["position"], c["width"], c["eta"])
            for a, c in zip(amps, components))
    return x, y


# --- fit_amplitudes_fixed_shape -------------------------------------------------

def test_recovers_exact_amplitudes_two_components():
    true_amps = [2.5e7, 8.0e6]
    x, y = _synthetic_spectrum(true_amps, [NARROW, BROAD])

    amps = fit_amplitudes_fixed_shape(x, y, -23, 27, [NARROW, BROAD])

    assert amps[0] == pytest.approx(true_amps[0], rel=1e-6)
    assert amps[1] == pytest.approx(true_amps[1], rel=1e-6)


def test_missing_component_gives_exact_zero_not_negative():
    """When a component is genuinely absent (amplitude 0 in the synthetic
    spectrum), NNLS should recover exactly 0 — never a negative value,
    which is precisely the failure mode this approach was chosen to avoid
    (see the docstring in relaxation_T2_components.py)."""
    true_amps = [2.5e7, 0.0]
    x, y = _synthetic_spectrum(true_amps, [NARROW, BROAD])

    amps = fit_amplitudes_fixed_shape(x, y, -23, 27, [NARROW, BROAD])

    assert amps[0] == pytest.approx(true_amps[0], rel=1e-6)
    assert amps[1] == pytest.approx(0.0, abs=1e-6)
    assert amps[1] >= 0  # NNLS guarantee: never negative


def test_recovers_amplitudes_with_noise():
    """With realistic noise, amplitudes should still be recovered within
    a few percent — not exact, but a meaningful sanity bound."""
    true_amps = [3.5e7, 1.0e7]
    x, y_clean = _synthetic_spectrum(true_amps, [NARROW, BROAD])
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01 * y_clean.max(), size=y_clean.shape)
    y_noisy = y_clean + noise

    amps = fit_amplitudes_fixed_shape(x, y_noisy, -23, 27, [NARROW, BROAD])

    assert amps[0] == pytest.approx(true_amps[0], rel=0.05)
    assert amps[1] == pytest.approx(true_amps[1], rel=0.05)


def test_single_component():
    """Sanity check with just one component, to isolate the NNLS
    machinery from the two-component overlap case."""
    x = np.linspace(-20, 20, 1000)
    true_amp = 1.2e7
    y = true_amp * pseudo_voigt(x, 1.0, NARROW["position"], NARROW["width"], NARROW["eta"])

    amps = fit_amplitudes_fixed_shape(x, y, -20, 20, [NARROW])

    assert len(amps) == 1
    assert amps[0] == pytest.approx(true_amp, rel=1e-6)


# --- monoexp_decay -------------------------------------------------

def test_monoexp_decay_formula():
    t = np.array([0.0, 100.0, 200.0])
    A0, T2 = 1e7, 150.0
    expected = A0 * np.exp(-t / T2)
    assert np.allclose(monoexp_decay(t, A0, T2), expected)


def test_monoexp_decay_at_t_zero_equals_A0():
    assert monoexp_decay(0.0, A0=5e6, T2=80.0) == pytest.approx(5e6)


def test_monoexp_decay_recovered_by_curve_fit():
    """Round-trip check: generate noisy decay data from known A0/T2,
    fit monoexp_decay to it, and recover those parameters — the same
    kind of fit performed per-component in the real script."""
    from scipy.optimize import curve_fit

    true_A0, true_T2 = 2.0e7, 220.0
    tau = np.array([80, 160, 320, 640, 1280, 2560, 5120, 10240], dtype=float)
    y_true = monoexp_decay(tau, true_A0, true_T2)
    rng = np.random.default_rng(1)
    y_noisy = y_true * (1 + rng.normal(0, 0.02, size=tau.shape))

    popt, _ = curve_fit(monoexp_decay, tau, y_noisy, p0=[y_noisy.max(), 500],
                         sigma=y_noisy, bounds=([0, 1], [1e8, 1e6]))

    assert popt[0] == pytest.approx(true_A0, rel=0.1)
    assert popt[1] == pytest.approx(true_T2, rel=0.1)
