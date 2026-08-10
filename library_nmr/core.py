"""
core.py — Shared low-level processing functions for library_nmr.

These functions are used by nearly every script in the library: reading
Bruker acquisition parameters, processing a single FID row (GRPDLY
correction, apodization, FFT, phasing), automatic phase search, and
parsing Bruker delay lists (vdlist).

Previously these were copy-pasted identically across 5-6 scripts
(pipeline_1d.py, pseudo2D_T1ir.py, pseudo2D_T1sr.py, pseudo2D_T2.py,
their two-component variants, Check_drift.py, MQMAS_2D_processing.py).
They now live here once, so a bug fix or improvement only needs to be
made in one place.
"""

import numpy as np


def find_grpdly_shift(dic):
    """Returns the number of points to roll out for GRPDLY correction, or 0 if none.

    GRPDLY (group delay) is a Bruker digital-filter artifact: the first
    points of the FID are shifted in time relative to the true start of
    the signal. This must be corrected (via a circular roll) before any
    further processing, or the resulting spectrum will show a
    frequency-dependent phase error.
    """
    grpdly = dic["acqus"].get("GRPDLY", 0)
    return int(round(grpdly)) if grpdly and grpdly > 0 else 0


def process_row(fid, dt, LB, ph0_rad, ph1_rad, grpdly_shift):
    """Processes a single FID row: GRPDLY correction, apodization, FFT, phasing.

    Parameters
    ----------
    fid : ndarray (complex)
        Raw time-domain signal (one row / one FID).
    dt : float
        Dwell time in seconds (1 / SW_h).
    LB : float
        Exponential line-broadening in Hz.
    ph0_rad, ph1_rad : float
        Zero- and first-order phase correction, in radians.
    grpdly_shift : int
        Number of points to roll out (from find_grpdly_shift). Use 0 if
        the correction has already been applied or is not needed.

    Returns
    -------
    ndarray (complex)
        The phased, frequency-domain spectrum (same length as `fid`).
    """
    if grpdly_shift > 0:
        fid = np.roll(fid, -grpdly_shift)
    N = len(fid)
    t = np.arange(N) * dt
    w = np.exp(-t * LB)
    fid_apodized = w * fid
    signal = np.fft.fftshift(np.fft.fft(fid_apodized))
    n = np.arange(N)
    phase = ph0_rad + ph1_rad * (n / N)
    return signal * np.exp(1j * phase)


def find_best_ph0(signal, ph1_rad):
    """Automatic PH0 search on a single spectrum, PH1 held fixed.

    Scans PH0 from -180 to 180 degrees (0.1° resolution) and returns the
    value that maximizes the sum of the real part of the phased spectrum
    — i.e. the value that makes the spectrum look most like a pure
    absorption lineshape (positive, symmetric peaks) rather than a
    dispersive one.
    """
    n = np.arange(len(signal))
    angles = np.linspace(-180, 180, 3601)
    scores = [
        np.sum(np.real(signal * np.exp(1j * (np.deg2rad(a) + ph1_rad * n / len(signal)))))
        for a in angles
    ]
    return angles[int(np.argmax(scores))]


def parse_bruker_delay(token):
    """Parses a single Bruker delay-list value (e.g. from vdlist) into seconds.

    Bruker delay lists use a numeric value optionally followed by a unit
    suffix: 'p' = picoseconds, 'n' = nanoseconds, 'u' = microseconds,
    'm' = milliseconds, 's' = seconds. No suffix is assumed to already be
    in seconds.

    Examples
    --------
    >>> parse_bruker_delay('80u')
    8e-05
    >>> parse_bruker_delay('2.5m')
    0.0025
    >>> parse_bruker_delay('150')
    150.0
    """
    token = token.strip()
    units = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "s": 1.0}
    if token and token[-1] in units:
        return float(token[:-1]) * units[token[-1]]
    return float(token)
