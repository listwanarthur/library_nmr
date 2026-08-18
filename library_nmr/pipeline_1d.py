import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.fitting import pseudo_voigt, fit_group, evaluate_fit_quality
from library_nmr.agr_export import export_agr

# ============================================================
# 1D NMR PROCESSING PIPELINE — library_nmr
# Usage: edit the CONFIGURATION block below, then run
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\202"  # Bruker experiment folder
LB = 10  # line broadening in Hz (10-200 Hz typical for solids)
PH0_MANUAL = -103.394  # PHC0 in degrees, used only if AUTO_PH0 = False
PH1 = -49.524  # PHC1 in degrees (first-order phase correction)
AUTO_PH0 = True  # True: automatic PH0 search by maximizing the real part
READ_PHASE_FROM_PROCS = False  # True: read ph0/ph1 from TopSpin (procs) instead
REFERENCE_SHIFT_PPM = 2  # additive ppm-axis shift = literature_position - measured_position
ZF_FACTOR = 1  # zero-filling multiplier: total FFT length = N*(1+ZF_FACTOR)

PEAKS = [
    # One group = a ppm window fitted with one or more pseudo-Voigt profiles
    # simultaneously. eta_fixed/width_bounds/position_bounds keep components
    # from swapping roles or latching onto the wrong feature during the fit.
    {
        "ppm_min": -23, "ppm_max": 27,
        "p0": [
            [3.5e7, 0.6, 5.8, 0.99],  # narrow component
            [1.0e7, 0.6, 15, 0.5],  # broad component
        ],
        "eta_fixed": [None, 0.0],  # broad fixed to pure Gaussian (static disorder)
        "width_bounds": [(0, 8), (8, 100)],
        "position_bounds": [(-4, 5), (-4, 5)],
    },
]
BASELINE_CORRECTION = False  # set True if the baseline is poor
BASELINE_REGIONS = [(-500, -100), (200, 500)]  # signal-free regions in ppm
BASELINE_METHOD = "polynomial"  # "polynomial" or "spline"
POLYNOMIAL_DEGREE = 3
SPLINE_SMOOTHING = 1e18
DISPLAY_ONLY = False  # set True to just display the spectrum without fitting
ZOOM = (30, -30)
OUTPUT_NAME = "results"

# Robustness check: re-fits the same group varying only one component's eta,
# to compare residuals across hypotheses. Set to None to disable.
ETA_COMPARISON = {
    "group_idx": 0,
    "component_idx": 1,
    "eta_values": [0.0, 0.25, 0.5, 0.75, 1.0],
}
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, read_phase_from_procs=False, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.

    PH0 priority: read_phase_from_procs > auto_ph0 > ph0_manual.
    Returns (ppm_axis, phased_spectrum, bruker_dic).
    """
    dic, data = ng.bruker.read(path, read_procs=read_phase_from_procs)

    grpdly_shift = find_grpdly_shift(dic)
    if grpdly_shift > 0:
        print(f"GRPDLY corrected: shifted by {grpdly_shift} points")
    else:
        print("GRPDLY not found or zero — no correction applied")

    N = data.shape[0]
    dt = 1 / dic["acqus"]["SW_h"]
    data_zf = np.concatenate([data, np.zeros(zf_factor * N, dtype=complex)])

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
        # PH1 must be included in the search score, otherwise PH0 is optimized
        # as if PH1=0, which is wrong as soon as PH1 isn't negligible.
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1_deg))
        print(f"Optimal PH0 found (auto, PH1={ph1_deg:.3f}° accounted for): {ph0_deg:.3f}°")

    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1_deg), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm

    return delta, spectrum, dic


def export(results, base_name, delta, signal, percentages=None, zoom=None):
    """Exports results to CSV, a PDF figure, and a Grace (.agr) file."""
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

    agr_series = [dict(x=delta, y=signal, mode="line", color="blue", legend="spectrum")]
    for i, r in enumerate(results):
        agr_series.append(dict(
            x=r["delta_peak"], y=pseudo_voigt(r["delta_peak"], *r["popt"]),
            mode="line", color=colors[i % len(colors)],
            legend=f"fit {i+1} ({percentages[i]:.1f}%)" if percentages else f"fit {i+1}",
        ))
    export_agr(f"{base_name}.agr", agr_series,
               xlabel="Chemical shift (ppm)", ylabel="Intensity (a.u.)",
               invert_x=True, xlim=zoom)


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

# Fit — each entry in PEAKS is a group, potentially multi-component
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
        print(f"  --> best residual for eta = {best[0]:.2f} (lowest RSS isn't necessarily "
              f"the correct physical model — check whether the gap is significant)")

# --- Export (skipped in DISPLAY_ONLY mode, which exits earlier) ---
export(results, OUTPUT_NAME, delta, signal, percentages=percentages_list, zoom=ZOOM)
