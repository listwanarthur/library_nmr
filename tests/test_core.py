"""
Tests for library_nmr.core — the shared low-level functions (GRPDLY
handling, FID processing, automatic phasing, Bruker delay parsing).

These use small, synthetic signals with a known ground truth (e.g. a
single decaying complex exponential of known frequency), rather than
real Bruker data — the goal is to check that each function does what
its docstring claims, not to re-validate real experiments.
"""

import numpy as np
import pytest

from library_nmr.core import (
    find_grpdly_shift,
    process_row,
    find_best_ph0,
    parse_bruker_delay,
)


# --- find_grpdly_shift -------------------------------------------------

def test_find_grpdly_shift_normal_value():
    dic = {"acqus": {"GRPDLY": 67.75}}
    assert find_grpdly_shift(dic) == 68  # rounds to nearest int


def test_find_grpdly_shift_missing_key():
    # No "GRPDLY" key at all — should not raise, should return 0.
    dic = {"acqus": {}}
    assert find_grpdly_shift(dic) == 0


def test_find_grpdly_shift_zero_or_negative():
    assert find_grpdly_shift({"acqus": {"GRPDLY": 0}}) == 0
    assert find_grpdly_shift({"acqus": {"GRPDLY": -1}}) == 0


# --- process_row ---------------------------------------------------------

def _synthetic_fid(freq_hz=500.0, n=1024, dt=1e-5, decay_hz=200.0):
    """A single decaying complex exponential — the FFT of this is a
    single Lorentzian peak centered at `freq_hz`."""
    t = np.arange(n) * dt
    return np.exp(1j * 2 * np.pi * freq_hz * t) * np.exp(-t * decay_hz)


def test_process_row_peak_position():
    """The FFT of a pure exponential should peak at the injected frequency."""
    dt = 1e-5
    n = 1024
    freq_hz = 500.0
    fid = _synthetic_fid(freq_hz=freq_hz, n=n, dt=dt)

    spectrum = process_row(fid, dt, LB=10, ph0_rad=0.0, ph1_rad=0.0, grpdly_shift=0)

    freqs = np.fft.fftshift(np.fft.fftfreq(n, dt))
    peak_freq = freqs[np.argmax(np.abs(spectrum))]
    assert peak_freq == pytest.approx(freq_hz, abs=2 * (1 / (n * dt)))
    # tolerance = ~2 frequency bins, since freq_hz may fall between two bins


def test_process_row_grpdly_shift_changes_result():
    """A nonzero grpdly_shift should actually roll the FID (result differs
    from grpdly_shift=0), confirming the correction is applied when requested."""
    fid = _synthetic_fid()
    dt = 1e-5
    spec_no_shift = process_row(fid, dt, LB=10, ph0_rad=0.0, ph1_rad=0.0, grpdly_shift=0)
    spec_shifted = process_row(fid, dt, LB=10, ph0_rad=0.0, ph1_rad=0.0, grpdly_shift=5)
    assert not np.allclose(spec_no_shift, spec_shifted)


def test_process_row_output_same_length_as_input():
    fid = _synthetic_fid(n=512)
    spectrum = process_row(fid, dt=1e-5, LB=10, ph0_rad=0.0, ph1_rad=0.0, grpdly_shift=0)
    assert len(spectrum) == len(fid)


# --- find_best_ph0 ---------------------------------------------------------

def test_find_best_ph0_recovers_known_phase_error():
    """If we deliberately dephase a spectrum by a known angle, find_best_ph0
    should find (approximately) the opposite angle needed to correct it."""
    dt = 1e-5
    fid = _synthetic_fid()
    # Reference spectrum, correctly phased (PH0=0 is "correct" for this synthetic FID)
    spectrum_correct = process_row(fid, dt, LB=10, ph0_rad=0.0, ph1_rad=0.0, grpdly_shift=0)

    # Artificially dephase it by +30 degrees
    injected_error_deg = 30.0
    spectrum_dephased = spectrum_correct * np.exp(1j * np.deg2rad(injected_error_deg))

    # find_best_ph0 should recover the correction needed to undo that error,
    # i.e. approximately -30 degrees (mod sign convention / 360 wraparound)
    found_ph0 = find_best_ph0(spectrum_dephased, ph1_rad=0.0)
    recovered = (found_ph0 + injected_error_deg) % 360
    recovered = recovered if recovered <= 180 else recovered - 360
    assert recovered == pytest.approx(0.0, abs=1.0)  # within 1 degree


# --- parse_bruker_delay ---------------------------------------------------------

@pytest.mark.parametrize("token,expected", [
    ("80u", 80e-6),
    ("2.5m", 2.5e-3),
    ("150", 150.0),
    ("1s", 1.0),
    ("3n", 3e-9),
    ("  80u  ", 80e-6),  # surrounding whitespace should be stripped
])
def test_parse_bruker_delay_units(token, expected):
    assert parse_bruker_delay(token) == pytest.approx(expected, rel=1e-9)


def test_parse_bruker_delay_no_suffix_assumed_seconds():
    assert parse_bruker_delay("42") == 42.0
