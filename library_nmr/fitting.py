"""
fitting.py — Shared pseudo-Voigt lineshape fitting for library_nmr.

Used by pipeline_1d.py (single-spectrum multi-component deconvolution)
and by the two-component T1 scripts (pseudo2D_T1ir_twocomponents.py,
pseudo2D_T1sr_twocomponents.py), which fit the lineshape once on the
best-conditioned row and reuse it.

Previously duplicated identically in three scripts.
"""

import numpy as np
from scipy.optimize import curve_fit


def pseudo_voigt(x, A, nu0, FWHM, eta):
    """Pseudo-Voigt profile: Lorentzian (eta) + Gaussian (1-eta) combination."""
    lorentzian = (A / np.pi) * ((FWHM / 2) / ((x - nu0) ** 2 + (FWHM / 2) ** 2))
    gaussian = A * np.exp(-((x - nu0) ** 2) / (2 * (FWHM / (2 * np.sqrt(2 * np.log(2)))) ** 2))
    return eta * lorentzian + (1 - eta) * gaussian


def sum_pseudo_voigt(x, *params):
    """Sum of N pseudo-Voigt profiles. params = [A1,nu1,FWHM1,eta1, A2,nu2,FWHM2,eta2, ...]."""
    total = np.zeros_like(x, dtype=float)
    n_peaks = len(params) // 4
    for i in range(n_peaks):
        A, nu0, FWHM, eta = params[i * 4:(i + 1) * 4]
        total += pseudo_voigt(x, A, nu0, FWHM, eta)
    return total


def fit_group(delta, spectrum, ppm_min, ppm_max, p0_list, eta_fixed_list=None,
              width_bounds_list=None, position_bounds_list=None):
    """Fits SIMULTANEOUSLY several overlapping pseudo-Voigt profiles within [ppm_min, ppm_max].

    Parameters
    ----------
    delta : ndarray
        Full ppm axis.
    spectrum : ndarray
        Full (real) spectrum, same length as `delta`.
    ppm_min, ppm_max : float
        Fit window.
    p0_list : list of [A, nu0, FWHM, eta]
        One entry per component to deconvolve.
    eta_fixed_list : list, optional
        Same length as p0_list. For each component:
            - None              -> eta stays a free fit parameter (default)
            - a value (0.0-1.0) -> eta is FIXED to this value.
    width_bounds_list : list of (FWHM_min, FWHM_max), optional
        Prevents the fit from swapping roles between components (a
        "narrow" one becoming broad and vice versa) depending on the
        starting point.
    position_bounds_list : list of (nu0_min, nu0_max), optional
        By default (None), each component can be positioned ANYWHERE
        within the whole [ppm_min, ppm_max] window — which lets a
        component that is supposed to overlap the center instead "latch
        onto" a shoulder or bump elsewhere in the window (a locally
        poorly-explained residual, an unphysical local minimum). Giving
        tight bounds around the expected position (e.g. (-5, 5) if the
        peak is around 0-1 ppm) forces each component to stay on the REAL
        peak of interest. Must be re-adjusted whenever a reference-shift
        correction changes, just like ppm_min/ppm_max.

    Returns
    -------
    list of dict
        One dict per component, with keys: position, err_position, width,
        err_width, amplitude, err_amplitude, eta, err_eta, eta_fixed,
        height, popt, delta_peak, signal_peak, integral, total_curve.
    """
    mask = (delta >= ppm_min) & (delta <= ppm_max)
    delta_peak = delta[mask]
    signal_peak = spectrum[mask]

    n_peaks = len(p0_list)
    if eta_fixed_list is None:
        eta_fixed_list = [None] * n_peaks
    if width_bounds_list is None:
        width_bounds_list = [(0, 100)] * n_peaks
    if position_bounds_list is None:
        position_bounds_list = [(ppm_min, ppm_max)] * n_peaks

    # --- Build the REDUCED parameter vector (without the fixed etas) ---
    p0_reduced, lower_bounds_reduced, upper_bounds_reduced = [], [], []
    for p0_i, eta_fixed, (fwhm_min, fwhm_max), (nu0_min, nu0_max) in zip(
        p0_list, eta_fixed_list, width_bounds_list, position_bounds_list
    ):
        A0, nu0_0, FWHM0, eta0 = p0_i
        p0_reduced += [A0, nu0_0, FWHM0]
        lower_bounds_reduced += [0, nu0_min, fwhm_min]
        upper_bounds_reduced += [np.inf, nu0_max, fwhm_max]
        if eta_fixed is None:
            p0_reduced.append(eta0)
            lower_bounds_reduced.append(0)
            upper_bounds_reduced.append(1)
        # if eta_fixed is not None, it is not added to the fitted parameters

    def reduced_model(x, *reduced_params):
        """Reconstructs the full [A,nu0,FWHM,eta]*n_peaks vector by re-injecting
        the fixed etas, then calls the pseudo-Voigt sum."""
        full_params = []
        idx = 0
        for eta_fixed in eta_fixed_list:
            A, nu0, FWHM = reduced_params[idx:idx + 3]
            idx += 3
            if eta_fixed is None:
                eta = reduced_params[idx]
                idx += 1
            else:
                eta = eta_fixed
            full_params += [A, nu0, FWHM, eta]
        return sum_pseudo_voigt(x, *full_params)

    popt_reduced, pcov_reduced = curve_fit(
        reduced_model, delta_peak, signal_peak, p0=p0_reduced,
        bounds=(lower_bounds_reduced, upper_bounds_reduced), maxfev=20000
    )
    uncertainties_reduced = np.sqrt(np.diag(pcov_reduced))

    # --- Reconstruct the full vector (with fixed etas, uncertainty = 0 for them) ---
    popt, uncertainties = [], []
    idx = 0
    for eta_fixed in eta_fixed_list:
        A, nu0, FWHM = popt_reduced[idx:idx + 3]
        errA, err_nu0, err_FWHM = uncertainties_reduced[idx:idx + 3]
        idx += 3
        if eta_fixed is None:
            eta, err_eta = popt_reduced[idx], uncertainties_reduced[idx]
            idx += 1
        else:
            eta, err_eta = eta_fixed, 0.0
        popt += [A, nu0, FWHM, eta]
        uncertainties += [errA, err_nu0, err_FWHM, err_eta]
    popt = np.array(popt)
    uncertainties = np.array(uncertainties)
    total_curve = sum_pseudo_voigt(delta_peak, *popt)

    results = []
    for i in range(n_peaks):
        A, nu0, FWHM, eta = popt[i * 4:(i + 1) * 4]
        err_A, err_nu0, err_FWHM, err_eta = uncertainties[i * 4:(i + 1) * 4]
        popt_i = popt[i * 4:(i + 1) * 4]
        curve_i = pseudo_voigt(delta_peak, *popt_i)
        # Sort before integrating: delta_peak may be decreasing (ppm axis
        # reversed relative to increasing frequencies) — np.trapezoid would
        # then give a negative-signed area, physically absurd for a positive peak.
        sort_idx = np.argsort(delta_peak)
        integral = np.trapezoid(curve_i[sort_idx], delta_peak[sort_idx])
        height = pseudo_voigt(nu0, A, nu0, FWHM, eta)
        results.append({
            "position": nu0, "err_position": err_nu0,
            "width": FWHM, "err_width": err_FWHM,
            "amplitude": A, "err_amplitude": err_A,  # fit parameter A — NOT the integrated area
            "eta": eta, "err_eta": err_eta, "eta_fixed": eta_fixed_list[i] is not None,
            "height": height,
            "popt": popt_i, "delta_peak": delta_peak, "signal_peak": signal_peak,
            "integral": integral, "total_curve": total_curve  # real area under the curve
        })
    return results


def evaluate_fit_quality(r_group):
    """Computes the quality of a group fit: sum of squared residuals (RSS)
    and RMS of the residual, over the group's window. Used to objectively
    compare several hypotheses (e.g. different fixed eta values) rather
    than judging by eye.
    """
    signal_peak = r_group[0]["signal_peak"]
    total_curve = r_group[0]["total_curve"]
    residual = signal_peak - total_curve
    rss = float(np.sum(residual ** 2))
    rms = float(np.sqrt(rss / len(residual)))
    return rss, rms
