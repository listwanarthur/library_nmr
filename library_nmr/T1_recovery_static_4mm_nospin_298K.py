import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T1_global_4mm_nospin_298K(_provisoire)
# Adapted from T1_recovery_static_298K.py for the NEW campaign: same 4mm
# MAS probe (H8906_0087) as the MAS measurements, but rotor NOT spinning
# -- isolates the spin/no-spin variable from any probe-to-probe difference,
# to test whether the static/MAS T1_slow discrepancy is a probe artefact.
#
# MAGNITUDE mode (|spectrum|), not real part + fixed phase -- same reason
# as the static-probe script: the wide powder pattern (no MAS averaging)
# makes AUTO_PH0/real-part unstable. PH0/PH1 below are irrelevant
# placeholders.
#
# Single RG (107.86) for the whole series -- unlike the original static
# script, no RG-mixing / empirical correction factor needed here, since
# exp717-736 were all acquired in one continuous session at the same RG.
#
# p1=1us (NOT the 2.5us MAS value) -- confirmed via CT-selective
# excitation calibration on this probe with the rotor stopped, see
# session notes. Recalibrate if you ever change RG/power/shim again.
#
# NS NORMALIZATION (fixed 15/09/2026): exp717-720 (D1=0.02,0.2,0.35,0.5s)
# were acquired with NS=128, exp721-736 (D1=0.7s onward) with NS=32.
# Raw magnitude scales linearly with NS (coherent averaging), so without
# dividing by NS the factor-of-4 discontinuity produced a spurious
# "V"-shaped drop in the recovery curve right after D1=0.5s. get_intensity()
# now returns intensity/NS -- do not remove this or re-introduce raw
# intensities into the fit.
# ============================================================

# === CONFIGURATION -- only section to edit ===
DATASETS = {
    0.02: r"D:\Postdoc\Datas\LLZO-400-sep26\717",
    0.2:  r"D:\Postdoc\Datas\LLZO-400-sep26\718",
    0.35: r"D:\Postdoc\Datas\LLZO-400-sep26\719",
    0.5:  r"D:\Postdoc\Datas\LLZO-400-sep26\720",
    0.7:  r"D:\Postdoc\Datas\LLZO-400-sep26\721",
    1:    r"D:\Postdoc\Datas\LLZO-400-sep26\722",
    1.5:  r"D:\Postdoc\Datas\LLZO-400-sep26\723",
    2:    r"D:\Postdoc\Datas\LLZO-400-sep26\724",
    3:    r"D:\Postdoc\Datas\LLZO-400-sep26\725",
    4:    r"D:\Postdoc\Datas\LLZO-400-sep26\726",
    6:    r"D:\Postdoc\Datas\LLZO-400-sep26\727",
    8:    r"D:\Postdoc\Datas\LLZO-400-sep26\728",
    15:   r"D:\Postdoc\Datas\LLZO-400-sep26\729",
    30:   r"D:\Postdoc\Datas\LLZO-400-sep26\730",
    60:   r"D:\Postdoc\Datas\LLZO-400-sep26\731",
    120:  r"D:\Postdoc\Datas\LLZO-400-sep26\732",
    150:  r"D:\Postdoc\Datas\LLZO-400-sep26\733",
    250:  r"D:\Postdoc\Datas\LLZO-400-sep26\734",
    400:  r"D:\Postdoc\Datas\LLZO-400-sep26\735",
    500:  r"D:\Postdoc\Datas\LLZO-400-sep26\736",
}

LB = 10               # line broadening in Hz -- same as the static-probe script
PH0_MANUAL = 0.0       # irrelevant in magnitude mode, kept for API compatibility
PH1 = 0.0              # irrelevant in magnitude mode
AUTO_PH0 = False       # irrelevant in magnitude mode -- do not turn on
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2   # TODO: check this still centers the peak correctly on
                          # THIS probe/dataset before trusting the fit -- it was
                          # tuned for the 7mm static probe, not verified here yet.
ZF_FACTOR = 1
PEAK_PPM_WINDOW = (100, -100)   # wide window, same logic as the static-probe script
                                 # since the rotor isn't spinning -- widen further if
                                 # the peak in your spectra sits outside +/-100ppm.
OUTPUT_NAME = r"D:\Postdoc\Figures\T1_global_4mm_nospin_298K_provisoire"
# Rename to drop _provisoire once you're satisfied the plateau is closed
# and the fit is stable.
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=False, read_phase_from_procs=False, reference_shift_ppm=0.0):
    dic, data = ng.bruker.read(path, read_procs=read_phase_from_procs)

    grpdly_shift = find_grpdly_shift(dic)
    if grpdly_shift > 0:
        print(f"  GRPDLY corrected: shifted by {grpdly_shift} points")
    else:
        print("  GRPDLY not found or zero — no correction applied")

    N = data.shape[0]
    dt = 1 / dic["acqus"]["SW_h"]
    data_zf = np.concatenate([data, np.zeros(zf_factor * N, dtype=complex)])

    ph0_deg = ph0_manual
    ph1_deg = ph1
    if read_phase_from_procs:
        try:
            ph0_deg = dic["procs"]["PHC0"]
            ph1_deg = dic["procs"]["PHC1"]
        except Exception:
            print("  Could not read phase from procs — using manual/auto value instead")
    elif auto_ph0:
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1_deg))

    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1_deg), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm

    return delta, spectrum, dic, ph0_deg


def get_peak_intensity_magnitude(delta, spectrum, ppm_window):
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    window = np.abs(spectrum[mask])
    idx = int(np.argmax(window))
    return float(window[idx]), float(delta[mask][idx])


def biexp_recovery(t, M0, f, T1a, T1b):
    return M0 * (1 - f * np.exp(-t / T1a) - (1 - f) * np.exp(-t / T1b))


def triexp_recovery(t, M0, f1, f2, T1a, T1b, T1c):
    return M0 * (1 - f1 * np.exp(-t / T1a) - f2 * np.exp(-t / T1b) - (1 - f1 - f2) * np.exp(-t / T1c))


def get_intensity(path):
    delta, spectrum, dic, _ = process_1d_spectrum(
        path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM,
    )
    intensity_raw, peak_ppm = get_peak_intensity_magnitude(delta, spectrum, PEAK_PPM_WINDOW)
    ns = dic["acqus"]["NS"]
    # NS normalization: coherent averaging over NS scans scales the raw
    # magnitude linearly with NS. This campaign used NS=128 for the first
    # few short-D1 points and NS=32 for the rest (see acqus of exp717-736)
    # -- without dividing by NS, that factor-of-4 discontinuity produces a
    # spurious "V"-shaped jump in the recovery curve right at D1=0.7s.
    # Caught by Arthur on 15/09/2026 after the first (buggy) run of this
    # script -- do not remove this normalization.
    intensity_per_scan = intensity_raw / ns
    return intensity_per_scan, peak_ppm, ns, intensity_raw


if __name__ == "__main__":
    print("=== T1 series (4mm probe, no spin, magnitude mode) ===")
    D1_list, I_list, ppm_list, ns_list = [], [], [], []
    for d1, path in sorted(DATASETS.items()):
        intensity, peak_ppm, ns, intensity_raw = get_intensity(path)
        print(f"  D1={d1:>6.2f}s  NS={ns:>4}  peak at {peak_ppm:7.2f} ppm"
              f"  I_raw={intensity_raw:.4e}  I_per_scan={intensity:.4e}")
        D1_list.append(d1)
        I_list.append(intensity)
        ppm_list.append(peak_ppm)
        ns_list.append(ns)

    ns_arr_check = np.array(ns_list)
    if len(np.unique(ns_arr_check)) > 1:
        print(f"\n  NOTE: NS varies across this series ({sorted(set(ns_list))}) -- "
              f"intensities below are already normalized per-scan (I/NS), so they "
              f"remain directly comparable across the NS change.")

    D1 = np.array(D1_list)
    I = np.array(I_list)
    ppm_arr = np.array(ppm_list)

    # Quick sanity check -- flag any point whose peak position looks like a
    # mispicked noise spike rather than the real 7Li line (same logic as the
    # static-probe script's calibration-pair check, applied here directly to
    # the main series since there's no separate calibration set this time).
    median_ppm = float(np.median(ppm_arr))
    PPM_OUTLIER_THRESHOLD = 5.0
    for d1, ppm in zip(D1, ppm_arr):
        dev = abs(ppm - median_ppm)
        if dev > PPM_OUTLIER_THRESHOLD:
            print(f"  WARNING: D1={d1}s peak at {ppm:.2f} ppm is {dev:.1f} ppm from the median"
                  f" ({median_ppm:.2f}) -- check this point before trusting it in the fit.")

    # === Fit -- biexponential, and triexponential for comparison ===
    print("\n--- Biexponential fit (sigma=I weighted) ---")
    p0_bi = [1.1 * I.max(), 0.1, 0.05, 20]
    bounds_bi = ([0.5 * I.max(), 0, 0.0001, 1], [5 * I.max(), 1, 5, 500])
    popt_bi, pcov_bi = curve_fit(biexp_recovery, D1, I, p0=p0_bi, sigma=I, maxfev=50000, bounds=bounds_bi)
    perr_bi = np.sqrt(np.diag(pcov_bi))
    M0_bi, f_bi, T1fast_bi, T1slow_bi = popt_bi
    print(f"  T1_slow = {T1slow_bi:.3g} +/- {perr_bi[3]:.2g} s  ({(1-f_bi)*100:.1f}%)")
    print(f"  T1_fast = {T1fast_bi:.3g} +/- {perr_bi[2]:.2g} s  ({f_bi*100:.1f}%)")

    print("\n--- Triexponential fit (sigma=I weighted, for comparison) ---")
    try:
        p0_tri = [1.1 * I.max(), 0.05, 0.05, 0.02, 2, 28]
        bounds_tri = ([0.5 * I.max(), 0, 0, 0.0001, 0.01, 1], [5 * I.max(), 0.5, 0.5, 1, 20, 500])
        popt_tri, pcov_tri = curve_fit(triexp_recovery, D1, I, p0=p0_tri, sigma=I, maxfev=50000, bounds=bounds_tri)
        perr_tri = np.sqrt(np.diag(pcov_tri))
        M0_tri, f1_tri, f2_tri, T1a_tri, T1b_tri, T1c_tri = popt_tri
        print(f"  T1a = {T1a_tri:.3g} +/- {perr_tri[3]:.2g} s  ({f1_tri*100:.1f}%)")
        print(f"  T1b = {T1b_tri:.3g} +/- {perr_tri[4]:.2g} s  ({f2_tri*100:.1f}%)")
        print(f"  T1c = {T1c_tri:.3g} +/- {perr_tri[5]:.2g} s  ({(1-f1_tri-f2_tri)*100:.1f}%)")
        use_tri = True
    except RuntimeError as e:
        print(f"  Triexponential fit failed ({e}) -- not enough points/too unstable, keep biexponential.")
        use_tri = False

    USE_TRIEXP_FOR_FIGURE = False  # flip to True by hand after comparing the two fits above

    print("\n--- Biexponential fit robustness (leave-one-out / reweighting) ---")
    for label, mask in [("tous", np.ones_like(D1, dtype=bool)),
                         ("sans D1=500", D1 != 500),
                         ("sans D1=400,500", ~np.isin(D1, [400, 500])),
                         ("sans D1=250,400,500", ~np.isin(D1, [250, 400, 500]))]:
        try:
            popt_i, pcov_i = curve_fit(biexp_recovery, D1[mask], I[mask], p0=p0_bi, sigma=I[mask],
                                        maxfev=50000, bounds=bounds_bi)
            print(f"  {label:<22} T1_slow = {popt_i[3]:.3g} +/- {np.sqrt(np.diag(pcov_i))[3]:.2g} s")
        except RuntimeError:
            print(f"  {label:<22} fit failed")
    for label, sigma in [("non pondere", None), ("sigma=sqrt(I)", np.sqrt(I))]:
        try:
            popt_i, pcov_i = curve_fit(biexp_recovery, D1, I, p0=p0_bi, sigma=sigma, maxfev=50000,
                                        bounds=bounds_bi)
            print(f"  {label:<22} T1_slow = {popt_i[3]:.3g} +/- {np.sqrt(np.diag(pcov_i))[3]:.2g} s")
        except RuntimeError:
            print(f"  {label:<22} fit failed")

    resid_bi = 100 * (I - biexp_recovery(D1, *popt_bi)) / I
    print("\nBiexponential relative residuals (%):", np.round(resid_bi, 2))

    df = pd.DataFrame({"D1_s": D1, "Intensity_per_scan": I, "NS": ns_list, "peak_ppm": ppm_arr})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    t_fit = np.logspace(np.log10(D1.min() / 2), np.log10(D1.max() * 1.3), 400)
    if USE_TRIEXP_FOR_FIGURE and use_tri:
        y_fit = triexp_recovery(t_fit, *popt_tri)
        fit_legend = (f"tri-exp fit: T1c={T1c_tri:.1f}+/-{perr_tri[5]:.1f}s ({(1-f1_tri-f2_tri)*100:.1f}%), "
                      f"T1b={T1b_tri:.2f}+/-{perr_tri[4]:.2f}s ({f2_tri*100:.1f}%), "
                      f"T1a={T1a_tri*1000:.2f}+/-{perr_tri[3]*1000:.2f}ms ({f1_tri*100:.1f}%)")
        title_suffix = "triexponential"
    else:
        y_fit = biexp_recovery(t_fit, *popt_bi)
        fit_legend = (f"biexp fit (PROVISOIRE): T1s={T1slow_bi:.1f}+/-{perr_bi[3]:.1f}s ({(1-f_bi)*100:.1f}%), "
                      f"T1f={T1fast_bi*1000:.1f}+/-{perr_bi[2]*1000:.1f}ms ({f_bi*100:.1f}%)")
        title_suffix = "biexponential"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(D1, I, color="blue", s=55, zorder=3, label="data")
    ax.plot(t_fit, y_fit, color="red", lw=1.5, zorder=2, label=f"{title_suffix} fit")
    ax.set_xscale("log")
    ax.set_xlabel("Recovery delay D1 (s)")
    ax.set_ylabel("Intensity (a.u., magnitude)")
    ax.set_title(r"$^7$Li T$_1$ recovery — 4mm probe, no spin, 298K")
    ax.text(0.97, 0.05, fit_legend, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    agr_series = [
        dict(x=D1, y=I, mode="symbol", color="blue", legend="data"),
        dict(x=t_fit, y=y_fit, mode="line", color="red", legend=fit_legend),
    ]
    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Recovery delay D1 (s)", ylabel="Intensity (a.u.)",
               xlog=True, title="7Li T1 recovery -- 4mm probe, no spin, 298K")
    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")
