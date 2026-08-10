import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.fitting import pseudo_voigt, fit_group, evaluate_fit_quality

# ============================================================
# 1D NMR PROCESSING PIPELINE — library_nmr (v3, English)
# Usage: edit the CONFIGURATION block below, then run
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\202"  # path to the Bruker experiment folder
LB = 10  # line broadening in Hz (10-200 Hz typical for solids)
PH0_MANUAL = -103.394  # PHC0 in degrees, used only if AUTO_PH0 = False
PH1 = -49.524  # PHC1 in degrees (first-order phase correction)
AUTO_PH0 = True  # True: automatic PH0 search by maximizing the real part
READ_PHASE_FROM_PROCS = False  # set True to read ph0/ph1 from TopSpin (procs) — takes priority over AUTO_PH0
REFERENCE_SHIFT_PPM = 2  # additive shift applied to the ppm axis (referencing).
                            #   REFERENCE_SHIFT_PPM = expected_literature_position - current_measured_position
                            # -> ppm_min/ppm_max in PEAKS and ZOOM must be adjusted accordingly.
ZF_FACTOR = 1  # zero-filling: 1=none, 2=double, 4=quadruple
PEAKS = [
    # A "group" = a ppm window in which ONE OR MORE pseudo-Voigt profiles are
    # fitted SIMULTANEOUSLY (deconvolution of overlapping peaks).
    # "p0" is a LIST of [A, nu0, FWHM, eta], one sub-p0 per component.
    # "eta_fixed" (optional, same length as p0): None = eta stays free,
    #   otherwise a value between 0 (Gaussian) and 1 (Lorentzian), fixed.
    # "width_bounds" (optional, same length as p0): (FWHM_min, FWHM_max)
    #   per component — PREVENTS the fit from swapping roles between
    #   components (e.g. the "narrow" one converging to a broad solution
    #   and vice versa), which would break the link between eta_fixed and
    #   the intended physical component.
    # "position_bounds" (optional, same length as p0): (nu0_min, nu0_max)
    #   per component — PREVENTS a component from latching onto a shoulder
    #   or bump elsewhere in the window instead of staying on the real peak.
    #   Must be re-adjusted whenever REFERENCE_SHIFT_PPM changes (the peak
    #   moves along with it).
    {
        "ppm_min": -23, "ppm_max": 27,  # +1.3 ppm shift applied
        "p0": [
            [3.5e7, 0.6, 5.8,  0.99],  # narrow component — eta well-determined, left free
            [1.0e7, 0.6, 15,   0.5 ],  # broad component — eta poorly constrained -> fixed below
        ],
        "eta_fixed": [None, 0.0],        # broad component fixed to pure Gaussian (static disorder)
        "width_bounds": [(0, 8), (8, 100)],  # narrow < 8 ppm ; broad > 8 ppm — disjoint bounds
        "position_bounds": [(-4, 5), (-4, 5)],  # both components stay near the real peak (~0.6 ppm)
    },
]
BASELINE_CORRECTION = False  # set True if the baseline is poor
BASELINE_REGIONS = [(-500, -100), (200, 500)]  # signal-free regions in ppm
BASELINE_METHOD = "polynomial"  # "polynomial" or "spline"
POLYNOMIAL_DEGREE = 3
SPLINE_SMOOTHING = 1e18
DISPLAY_ONLY = False  # set True to just display the spectrum without fitting
ZOOM = (30, -30)  # zoom on the central band to visualize the two-component fit
OUTPUT_NAME = "results"

# Comparison of eta hypotheses for one component (robustness test):
# runs the same fit multiple times, varying ONLY the eta_fixed value of the
# targeted component, and compares the residual (RSS/RMS) for each hypothesis.
# Set to None to disable.
ETA_COMPARISON = {
    "group_idx": 0,             # index in PEAKS
    "component_idx": 1,         # index of the tested component in this group (0=narrow, 1=broad)
    "eta_values": [0.0, 0.25, 0.5, 0.75, 1.0],  # 0=pure Gaussian, 1=pure Lorentzian
}
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, read_phase_from_procs=False, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.

    Built on top of the shared core.process_row (which does GRPDLY + apodization
    + FFT + phasing in one call) and core.find_best_ph0 (automatic PH0 search).

    reference_shift_ppm: additive shift applied to the ppm AXIS after computation
        (referencing). Useful when O1 was set to an arbitrary value (e.g. 0)
        because the true chemical shift was not known at acquisition time. Once
        the correct literature reference is found, compute:
            reference_shift_ppm = expected_literature_position - current_measured_position
        and apply it here. Changes NOTHING about the spectrum shape, only its
        ppm labeling — so ppm_min/ppm_max in PEAKS must be adjusted accordingly.

    Returns the ppm axis, the phased complex spectrum, and the Bruker dictionary
    `dic` (useful afterwards to reuse SFO1, O1, etc. without re-reading files).
    PH0 is chosen with this priority:
        1) read_phase_from_procs=True -> read from procs (TopSpin)
        2) auto_ph0=True              -> automatic search (maximizes Re(spectrum))
        3) otherwise                  -> ph0_manual given in configuration
    """
    dic, data = ng.bruker.read(path, read_procs=read_phase_from_procs)

    grpdly_shift = find_grpdly_shift(dic)
    if grpdly_shift > 0:
        print(f"GRPDLY corrected: shifted by {grpdly_shift} points")
    else:
        print("GRPDLY not found or zero — no correction applied")

    N = data.shape[0]
    dt = 1 / dic["acqus"]["SW_h"]

    # Zero-filling is applied here (before calling process_row) since process_row
    # operates on a fixed-length FID; we pad first, then let process_row handle
    # GRPDLY/apodization/FFT/phase on the padded array.
    data_zf = np.concatenate([data, np.zeros(zf_factor * N, dtype=complex)])

    # --- PH0 determination ---
    ph0_deg = ph0_manual
    ph1_deg = ph1
    if read_phase_from_procs:
        try:
            ph0_deg = dic["procs"]["PHC0"]
            ph1_deg = dic["procs"]["PHC1"]
            print(f"Phase read from procs: PH0={ph0_deg:.3f}°, PH1={ph1_deg:.3f}°")
        except Exception:
            print("Could not read phase from procs — using manual/auto value instead")
    elif auto_ph0:
        # First pass with PH0=0 to get the unphased (but PH1-corrected) spectrum,
        # then search PH0 on it. IMPORTANT: the score must include PH1 (already
        # fixed/known), otherwise the search optimizes PH0 as if PH1=0, which
        # gives the correct angle for the wrong problem as soon as PH1 is not
        # negligible.
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1_deg))
        print(f"Optimal PH0 found (auto, PH1={ph1_deg:.3f}° accounted for): {ph0_deg:.3f}°")

    # Final pass: full GRPDLY + apodization + FFT + phase, with the chosen PH0.
    # (grpdly_shift is applied here since the search pass above used grpdly_shift=0
    # to avoid double-rolling; np.roll on the raw data would already be needed
    # before zero-filling in principle, but since GRPDLY only shifts within the
    # original N points and the padding is zeros, applying it inside process_row
    # on data_zf is equivalent and keeps a single code path.)
    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1_deg), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm  # manual referencing (see docstring)

    return delta, spectrum, dic


def export(results, base_name, delta, signal, percentages=None, zoom=None):
    """Exports the results to CSV and generates the PDF figure."""
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ["popt", "delta_peak", "signal_peak"]} for r in results])
    df.to_csv(f"{base_name}.csv", index=False)

    colors = ["red", "green", "orange", "purple"]
    fig, axes = plt.subplots(figsize=(10, 5))
    axes.plot(delta, signal, color="blue", linewidth=1, label="spectrum")
    for i, r in enumerate(results):
        axes.plot(r["delta_peak"], pseudo_voigt(r["delta_peak"], *r["popt"]),
                        color=colors[i % len(colors)], linewidth=1.5,
                        label=f"fit {i+1} ({percentages[i]:.1f}%)" if percentages else f"fit {i+1}")

    residual = signal.copy()
    for r in results:
        mask = (delta >= r["delta_peak"][0]) & (delta <= r["delta_peak"][-1])
        residual[mask] -= pseudo_voigt(delta[mask], *r["popt"])
    axes2 = axes.twinx()
    axes2.plot(delta, residual, color="gray", linewidth=0.5, alpha=0.5, label="residual")
    axes2.set_ylabel("Residual", color="gray")
    axes2.tick_params(axis="y", labelcolor="gray")
    axes2.spines["top"].set_visible(False)

    axes.invert_xaxis()
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.set_xlabel("Chemical shift (ppm)")
    axes.set_ylabel("Intensity (a.u.)")
    axes.legend()
    if zoom is not None:
        axes.set_xlim(zoom[0], zoom[1])
    plt.savefig(f"{base_name}.pdf")
    plt.show()


# === PROCESSING ===
delta, spectrum, bruker_dic = process_1d_spectrum(
    PATH, LB, PH0_MANUAL, PH1, ZF_FACTOR,
    auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS, reference_shift_ppm=REFERENCE_SHIFT_PPM
)
SFO1 = bruker_dic["acqus"]["SFO1"]  # MHz — used to convert widths from ppm to Hz
signal = spectrum.real  # real part: phased absorptive spectrum

# Baseline correction
if BASELINE_CORRECTION:
    baseline_mask = np.zeros(len(delta), dtype=bool)
    for ppm_min, ppm_max in BASELINE_REGIONS:
        baseline_mask |= (delta >= ppm_min) & (delta <= ppm_max)
    if BASELINE_METHOD == "polynomial":
        coeffs = np.polyfit(delta[baseline_mask], signal[baseline_mask], POLYNOMIAL_DEGREE)
        baseline = np.polyval(coeffs, delta)
        print(f"Polynomial baseline correction (degree {POLYNOMIAL_DEGREE}) applied")
    elif BASELINE_METHOD == "spline":
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(delta[baseline_mask], signal[baseline_mask], s=SPLINE_SMOOTHING)
        baseline = spline(delta)
        print("Spline baseline correction applied")
    signal = signal - baseline

# Display-only mode
if DISPLAY_ONLY:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(delta, signal, color="blue", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.invert_xaxis()
    ax.set_xlabel("Chemical shift (ppm)")
    ax.set_ylabel("Intensity (a.u.)")
    if ZOOM is not None:
        ax.set_xlim(ZOOM[0], ZOOM[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()
    exit()

# Fit — each entry in PEAKS is a GROUP, potentially multi-component
results = []
for i, group in enumerate(PEAKS):
    try:
        r_group = fit_group(
            delta, signal, group["ppm_min"], group["ppm_max"], group["p0"],
            eta_fixed_list=group.get("eta_fixed", None),
            width_bounds_list=group.get("width_bounds", None),
            position_bounds_list=group.get("position_bounds", None)
        )
        results.extend(r_group)
        for j, r in enumerate(r_group):
            print(f"Group {i+1}, component {j+1}:")
            print(f"   position   = {r['position']:.3f} +/- {r['err_position']:.3f} ppm")
            print(f"   width      = {r['width']:.3f} +/- {r['err_width']:.3f} ppm  "
                  f"({r['width']*SFO1:.1f} +/- {r['err_width']*SFO1:.1f} Hz)")
            if r["eta_fixed"]:
                print(f"   eta        = {r['eta']:.3f}  (FIXED, not adjusted)")
            else:
                print(f"   eta        = {r['eta']:.3f} +/- {r['err_eta']:.3f}  "
                      f"({'!! close to bound 0 or 1 !!' if r['eta'] < 0.03 or r['eta'] > 0.97 else 'ok'})")
            print(f"   amplitude  = {r['amplitude']:.3e} +/- {r['err_amplitude']:.3e}  (fit parameter A, NOT the area)")
            print(f"   integral   = {r['integral']:.3e}  (real area under the curve)")
    except RuntimeError:
        print(f"Fit failed for group {i+1} — adjust ppm_min/ppm_max or p0")

integrals = [r["integral"] for r in results]
total = sum(integrals)
percentages_list = [i/total*100 for i in integrals]
for i, (r, pct) in enumerate(zip(results, percentages_list)):
    print(f"Component {i+1}: {pct:.1f}% of total area")

# --- Eta hypothesis comparison (robustness test) ---
if ETA_COMPARISON is not None:
    gi = ETA_COMPARISON["group_idx"]
    ci = ETA_COMPARISON["component_idx"]
    group_cfg = PEAKS[gi]
    n_peaks_group = len(group_cfg["p0"])
    eta_fixed_base = list(group_cfg.get("eta_fixed", [None] * n_peaks_group))

    print(f"\n=== Eta comparison — group {gi+1}, component {ci+1} ===")
    print("  (RSS/RMS computed on the residual of the WHOLE group, not just this component)")
    best = None
    for eta_test in ETA_COMPARISON["eta_values"]:
        eta_fixed_test = list(eta_fixed_base)
        eta_fixed_test[ci] = eta_test
        try:
            r_test = fit_group(
                delta, signal, group_cfg["ppm_min"], group_cfg["ppm_max"],
                group_cfg["p0"], eta_fixed_list=eta_fixed_test,
                width_bounds_list=group_cfg.get("width_bounds", None),
                position_bounds_list=group_cfg.get("position_bounds", None)
            )
            rss, rms = evaluate_fit_quality(r_test)
            integrals_test = [rr["integral"] for rr in r_test]
            total_test = sum(integrals_test)
            split = ", ".join(f"{i/total_test*100:.1f}%" for i in integrals_test)
            print(f"  eta = {eta_test:.2f}: RSS = {rss:.4e}, residual RMS = {rms:.4e}, split = [{split}]")
            if best is None or rss < best[1]:
                best = (eta_test, rss)
        except RuntimeError:
            print(f"  eta = {eta_test:.2f}: fit failed")
    if best is not None:
        print(f"  --> best residual for eta = {best[0]:.2f} "
              f"(the lowest RSS doesn't necessarily mean the correct physical model — "
              f"also check whether the gap between hypotheses is significant or marginal)")

# --- Export (only if a fit was actually run — DISPLAY_ONLY exits before this point) ---
export(results, OUTPUT_NAME, delta, signal, percentages=percentages_list, zoom=ZOOM)
