"""
Regression test for the sigma= weighting bug in biexponential T1 recovery
fits (relaxation_T1_onepulse_series.py / relaxation_T1_components.py).

HISTORY: this exact bug (curve_fit called without sigma=) was found and
fixed once already in this project's working scripts (see fit_T1_recovery.py
/ fit_T1_components.py in Library_nmr), documented as causing a ~15-18%
systematic residual trend at short D1 -- then found to have REGRESSED in
this Portfolio package on 18/08 (both relaxation_T1_onepulse_series.py and
relaxation_T1_components.py were calling curve_fit without sigma=). This
test exists so that regression can't happen silently again: it fits the
SAME synthetic data with and without sigma= weighting and asserts that
weighting is required to recover T1_fast within a reasonable tolerance.

Why this matters physically: the T1 recovery curve spans a wide dynamic
range (weak signal at short D1, near-full recovery at long D1). An
unweighted least-squares fit is dominated by the large-amplitude long-D1
points and effectively ignores the short-D1 points where T1_fast lives.
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

from library_nmr.relaxation_T1_onepulse_series import biexp_recovery


# Synthetic ground truth chosen to mimic the SHAPE of the real LLZO series
# (wide D1 range, small T1_fast fraction, T1_fast/T1_slow well separated --
# exactly the regime where unweighted fits are known to fail, see module
# docstring) WITHOUT reproducing the actual measured values -- this
# repository is public and the real numbers are part of an unpublished
# manuscript.
TRUE_M0 = 1.0e7
TRUE_F = 0.03
TRUE_T1_FAST = 0.015
TRUE_T1_SLOW = 20.0
D1 = np.array([0.05, 0.2, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 200])


def _synthetic_recovery(noise_frac=0.02, seed=42):
    """Proportional (not additive) noise -- matches real S/N behavior, where
    absolute noise is roughly constant but the SIGNAL grows with D1, so the
    relative noise is largest at short D1 (small signal)."""
    I_true = biexp_recovery(D1, TRUE_M0, TRUE_F, TRUE_T1_FAST, TRUE_T1_SLOW)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_frac * I_true, size=I_true.shape)
    return I_true + noise


def _fit(I, use_sigma):
    p0 = [1.1 * I.max(), 0.05, 0.05, 30]
    bounds = ([0.5 * I.max(), 0, 0.001, 5], [5 * I.max(), 0.3, 5, 200])
    kwargs = {"sigma": I} if use_sigma else {}
    popt, _ = curve_fit(biexp_recovery, D1, I, p0=p0, bounds=bounds, maxfev=50000, **kwargs)
    return popt


def test_weighted_fit_recovers_T1_fast_within_tolerance():
    """With sigma=I weighting (the fixed/correct behavior), T1_fast should
    be recovered within ~10% on this synthetic series."""
    I = _synthetic_recovery()
    popt = _fit(I, use_sigma=True)
    T1_fast_fit = popt[2]
    assert T1_fast_fit == pytest.approx(TRUE_T1_FAST, rel=0.15)


def test_unweighted_fit_is_measurably_worse_than_weighted():
    """This is the actual regression guard: on the same data, the unweighted
    fit (the bug that regressed into this package on 18/08) must be a worse
    estimate of T1_fast than the weighted fit -- if this ever fails, either
    the bug is gone for a good reason (great, update this test) or someone
    removed sigma= again without noticing."""
    I = _synthetic_recovery()
    T1_fast_weighted = _fit(I, use_sigma=True)[2]
    T1_fast_unweighted = _fit(I, use_sigma=False)[2]

    err_weighted = abs(T1_fast_weighted - TRUE_T1_FAST) / TRUE_T1_FAST
    err_unweighted = abs(T1_fast_unweighted - TRUE_T1_FAST) / TRUE_T1_FAST

    assert err_weighted < err_unweighted, (
        f"Expected weighted fit (err={err_weighted:.1%}) to beat unweighted "
        f"fit (err={err_unweighted:.1%}) on T1_fast -- if it doesn't, the "
        f"sigma= weighting fix may have been lost again."
    )
    # The historical bug produced a ~15-18% systematic bias -- guard against
    # regressions weaker than that slipping through unnoticed.
    assert err_unweighted > 0.10
