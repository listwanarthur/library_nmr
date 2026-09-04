import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T1_global_statique_298K(_provisoire)
# Companion to relaxation_T1_onepulse_series.py, adapted for the static
# probe. MAGNITUDE mode (|spectrum|), not real part + fixed phase: the
# wide static powder pattern makes AUTO_PH0/real-part unstable, so
# PH0/PH1 below are irrelevant placeholders.
#
# Mixes two RG blocks: archive (RG=107.86, D1<=150s) and a later
# addition closing the plateau (D1=250,400s, RG=94.34). Raw intensities
# aren't comparable across RG, so an empirical correction factor is
# derived from matched cross-calibration points (RG_CALIBRATION_PAIRS)
# and applied to the new-RG points (NEW_RG_D1_POINTS) before fitting.
# ============================================================

# === CONFIGURATION -- only section to edit ===
DATASETS = {
    0.02:  r"D:\Postdoc\Datas\LLZO-400-aug26\320",
    0.05:  r"D:\Postdoc\Datas\LLZO-400-aug26\321",
    0.2:   r"D:\Postdoc\Datas\LLZO-400-aug26\322",
    0.5:   r"D:\Postdoc\Datas\LLZO-400-aug26\323",
    1:     r"D:\Postdoc\Datas\LLZO-400-aug26\324",
    2:     r"D:\Postdoc\Datas\LLZO-400-aug26\325",
    4:     r"D:\Postdoc\Datas\LLZO-400-aug26\326",
    8:     r"D:\Postdoc\Datas\LLZO-400-aug26\327",
    15:    r"D:\Postdoc\Datas\LLZO-400-aug26\328",
    30:    r"D:\Postdoc\Datas\LLZO-400-aug26\329",
    60:    r"D:\Postdoc\Datas\LLZO-400-aug26\330",
    120:   r"D:\Postdoc\Datas\LLZO-400-aug26\331",
    150:   r"D:\Postdoc\Datas\LLZO-400-aug26\307",
    # plateau-closing points, RG=94.34 (corrected below)
    250:   r"D:\Postdoc\Datas\LLZO-400-aug26\386",
    400:   r"D:\Postdoc\Datas\LLZO-400-aug26\387",
}

# Matched D1 points at both RG values, same NS=32 each side, used to
# derive the empirical correction factor automatically.
# D1=0.5 uses exp409 (clean redo) instead of exp381 (DS=9 bug,
# under-equilibrated) at RG=94.34.
RG_CALIBRATION_PAIRS = {
    # D1 (s): (archive path RG=107.86, new path RG=94.34)
    0.5:  (r"D:\Postdoc\Datas\LLZO-400-aug26\323", r"D:\Postdoc\Datas\LLZO-400-aug26\409"),
    15:   (r"D:\Postdoc\Datas\LLZO-400-aug26\328", r"D:\Postdoc\Datas\LLZO-400-aug26\380"),
    30:   (r"D:\Postdoc\Datas\LLZO-400-aug26\329", r"D:\Postdoc\Datas\LLZO-400-aug26\382"),
    60:   (r"D:\Postdoc\Datas\LLZO-400-aug26\330", r"D:\Postdoc\Datas\LLZO-400-aug26\383"),
    120:  (r"D:\Postdoc\Datas\LLZO-400-aug26\331", r"D:\Postdoc\Datas\LLZO-400-aug26\384"),
    150:  (r"D:\Postdoc\Datas\LLZO-400-aug26\307", r"D:\Postdoc\Datas\LLZO-400-aug26\385"),
}
# D1 values in DATASETS that were acquired at the NEW RG (94.34) and need
# the correction factor applied before fitting.
NEW_RG_D1_POINTS = [250, 400]

LB = 10               # line broadening in Hz -- same as relaxation_T1_onepulse_series.py
PH0_MANUAL = 0.0       # irrelevant in magnitude mode, kept for API compatibility
PH1 = 0.0              # irrelevant in magnitude mode
AUTO_PH0 = False       # irrelevant in magnitude mode -- do not turn on (unstable on this static line)
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
PEAK_PPM_WINDOW = (100, -100)   # static powder pattern is wide -- matches relaxation_T1_vs_temperature.py
OUTPUT_NAME = r"D:\Postdoc\Figures\T1_global_statique_298K_provisoire"
# Rename to T1_global_statique_298K (drop _provisoire) once you're satisfied
# the plateau is genuinely closed and the fit is stable.
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
    delta, spectrum, _, _ = process_1d_spectrum(
        path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM,
    )
    intensity, peak_ppm = get_peak_intensity_magnitude(delta, spectrum, PEAK_PPM_WINDOW)
    return intensity, peak_ppm


if __name__ == "__main__":
    # === STEP 1: derive the empirical RG correction factor ===
    # Auto-flags/excludes any calibration point whose peak was found far
    # from where the 7Li line sits (likely mispicked noise in magnitude mode).
    print("=== RG correction factor (empirical, from matched cross-cal points) ===")
    cal_records = {}  # d1 -> {"archive": (I, ppm), "new": (I, ppm)}
    for d1, (archive_path, new_path) in sorted(RG_CALIBRATION_PAIRS.items()):
        i_archive, ppm_archive = get_intensity(archive_path)
        i_new, ppm_new = get_intensity(new_path)
        cal_records[d1] = {"archive": (i_archive, ppm_archive), "new": (i_new, ppm_new)}

    all_ppm = [v[1] for rec in cal_records.values() for v in rec.values()]
    median_ppm = float(np.median(all_ppm))
    PPM_OUTLIER_THRESHOLD = 5.0  # ppm from the median

    excluded = {}
    for d1, rec in cal_records.items():
        for side, (intensity, ppm) in rec.items():
            dev = abs(ppm - median_ppm)
            if dev > PPM_OUTLIER_THRESHOLD:
                excluded.setdefault(d1, []).append(side)
                print(f"  D1={d1:>6}s ({side}): peak at {ppm:.2f} ppm, {dev:.1f} ppm from median"
                      f" ({median_ppm:.2f}) -- looks like a mispicked noise spike, excluding this D1 pair.")

    ratios, used_d1 = [], []
    for d1, rec in sorted(cal_records.items()):
        if d1 in excluded:
            continue
        i_archive, _ = rec["archive"]
        i_new, _ = rec["new"]
        ratio = i_archive / i_new
        ratios.append(ratio)
        used_d1.append(d1)
        print(f"  D1={d1:>6}s   archive={i_archive:.4e}   new={i_new:.4e}   ratio={ratio:.4f}")

    ratios = np.array(ratios)
    RG_CORRECTION = float(np.mean(ratios))
    RG_CORRECTION_STD = float(np.std(ratios))
    print(f"\n  Used {len(ratios)}/{len(RG_CALIBRATION_PAIRS)} calibration points"
          f" (excluded D1={sorted(excluded)})" if excluded else
          f"\n  Used all {len(ratios)} calibration points (none excluded)")
    print(f"  Mean correction factor (archive/new) = {RG_CORRECTION:.4f} +/- {RG_CORRECTION_STD:.4f}")
    print(f"  (theoretical RG ratio 107.86/94.34 = {107.86/94.34:.4f} -- compare, don't assume they must match)")
    if RG_CORRECTION_STD / RG_CORRECTION > 0.1:
        print("  WARNING: >10% spread across calibration points -- check individual ratios above"
              " before trusting this factor.")

    # D1=250/400s sit beyond the calibrated range (max 150s) -- extrapolation.
    # Compare the mean factor against the factor from the single highest-D1
    # calibration point (closer to what an extrapolated trend would need).
    if used_d1:
        d1_max = max(used_d1)
        factor_nearest = ratios[used_d1.index(d1_max)]
        print(f"  Sensitivity check -- factor from D1={d1_max}s alone (nearest to the extrapolation): "
              f"{factor_nearest:.4f}")
        rel_diff = abs(factor_nearest - RG_CORRECTION) / RG_CORRECTION
        if rel_diff > 0.05:
            print(f"  NOTE: {rel_diff*100:.1f}% different from the mean -- the ratio trend with D1 isn't flat,"
                  f" keep this in mind since 250/400s are extrapolated beyond D1=150s.")

    # === STEP 2: process the main D1 series, correcting the new-RG points ===
    print("\n=== T1 series (static, magnitude mode) ===")
    D1_list, I_list = [], []
    for d1, path in sorted(DATASETS.items()):
        intensity, peak_ppm = get_intensity(path)
        if d1 in NEW_RG_D1_POINTS:
            intensity_corrected = intensity * RG_CORRECTION
            print(f"  D1={d1:>6.2f}s  peak at {peak_ppm:7.2f} ppm  raw={intensity:.4e}"
                  f"  -> corrected={intensity_corrected:.4e}  (RG=94.34, factor applied)")
            intensity = intensity_corrected
        else:
            print(f"  D1={d1:>6.2f}s  peak at {peak_ppm:7.2f} ppm  intensity={intensity:.4e}  (RG=107.86, archive)")
        D1_list.append(d1)
        I_list.append(intensity)

    D1 = np.array(D1_list)
    I = np.array(I_list)

    # === STEP 3: fit -- biexponential, and triexponential for comparison
    # now that the plateau should be closed. ===
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

    if use_tri:
        print("\n--- Triexponential fit robustness (leave-one-out / reweighting) ---")
        print("  (same battery as used for the MAS T1c figure -- if T1c collapses or swings wildly")
        print("   on any variant below, don't trust the headline T1c=%.1fs yet)" % T1c_tri)
        for label, mask in [("tous", np.ones_like(D1, dtype=bool)),
                             ("sans D1=400", D1 != 400),
                             ("sans D1=250,400", ~np.isin(D1, [250, 400])),
                             ("sans D1=150", D1 != 150)]:
            try:
                popt_i, pcov_i = curve_fit(triexp_recovery, D1[mask], I[mask], p0=p0_tri, sigma=I[mask],
                                            maxfev=50000, bounds=bounds_tri)
                print(f"  {label:<20} T1c = {popt_i[5]:.3g} +/- {np.sqrt(np.diag(pcov_i))[5]:.2g} s")
            except RuntimeError:
                print(f"  {label:<20} fit failed")
        for label, sigma in [("non pondere", None), ("sigma=sqrt(I)", np.sqrt(I))]:
            try:
                popt_i, pcov_i = curve_fit(triexp_recovery, D1, I, p0=p0_tri, sigma=sigma, maxfev=50000,
                                            bounds=bounds_tri)
                print(f"  {label:<20} T1c = {popt_i[5]:.3g} +/- {np.sqrt(np.diag(pcov_i))[5]:.2g} s")
            except RuntimeError:
                print(f"  {label:<20} fit failed")

    # === STEP 4: pick which fit to plot/export -- default to biexp unless
    # triexp is justified after comparing the two printouts + residuals. ===
    USE_TRIEXP_FOR_FIGURE = False  # flip to True by hand after comparing the two fits above

    print("\n--- Biexponential fit robustness (leave-one-out / reweighting) ---")
    for label, mask in [("tous", np.ones_like(D1, dtype=bool)),
                         ("sans D1=400", D1 != 400),
                         ("sans D1=250,400", ~np.isin(D1, [250, 400])),
                         ("sans D1=150,250,400", ~np.isin(D1, [150, 250, 400]))]:
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

    df = pd.DataFrame({"D1_s": D1, "Intensity": I})
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
    ax.scatter(D1, I, color="blue", s=55, zorder=3, label="data (RG-corrected)")
    ax.plot(t_fit, y_fit, color="red", lw=1.5, zorder=2, label=f"{title_suffix} fit")
    ax.set_xscale("log")
    ax.set_xlabel("Recovery delay D1 (s)")
    ax.set_ylabel("Intensity (a.u., magnitude, RG-corrected)")
    ax.set_title(r"$^7$Li T$_1$ recovery — static probe, 298K")
    ax.text(0.97, 0.05, fit_legend, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    agr_series = [
        dict(x=D1, y=I, mode="symbol", color="blue", legend="data (RG-corrected)"),
        dict(x=t_fit, y=y_fit, mode="line", color="red", legend=fit_legend),
    ]
    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Recovery delay D1 (s)", ylabel="Intensity (a.u.)",
               xlog=True, title="7Li T1 recovery -- static probe, 298K")
    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")
