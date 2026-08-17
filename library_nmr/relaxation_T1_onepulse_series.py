import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0

# ============================================================
# T1 RECOVERY FIT — library_nmr (onepulse D1 series)
# Usage: edit the CONFIGURATION block below, then run
#
# Why one spectrum per D1 instead of a pseudo-2D saturation-recovery:
# a shared-reference two-component decomposition across many rows in
# one long acquisition turned out unstable and drift-prone in this
# project. A series of short, independent, individually-phased
# spectra proved far more robust, using a simple total peak
# intensity (not a fragile two-component decomposition) per D1.
# ============================================================

# === CONFIGURATION — only section to edit ===
# NOTE (filled in 18/08 from the working Library_nmr copy, fit_T1_recovery.py):
# this is the CURRENT VALIDATED series (exp281-286 for short D1, exp227-236 for
# the rest, NS=32) — identified by scanning acqus D1/PULPROG/NS across the
# whole dataset folder. The previous 0.3->251 / 0.5->252 entries here were from
# an OLDER, ABANDONED short series (exp250-252, D1=0.1/0.3/0.5 only, never
# completed past D1=0.5) — NOT the same acquisitions as 284/285 below, and not
# usable on their own for a full biexponential. Do not mix them back in.
#
# CAUTION / HISTORY (15-17/08): a systematic ~15-18% residual trend at short
# D1 was first observed and (wrongly) attributed to DS=0 (no dummy scans,
# insufficient pre-equilibration for D1 short compared to T1_slow~28s).
# exp266-269 were reacquired with DS=16, then DS scaled per-point to target
# ~5xT1_slow of pre-equilibration time (exp281-286, DS=3000/1000/750/500/
# 300/215 for D1=0.05/0.15/0.2/0.3/0.5/0.7) — the residual pattern was
# UNCHANGED either time. The real cause: the fit was UNWEIGHTED (no `sigma=`
# in curve_fit), so ordinary least squares minimizes ABSOLUTE residuals and
# is dominated by the large-intensity long-D1 points; the much smaller
# short-D1 points (exactly where T1_fast lives) were effectively free to be
# fit poorly without penalty. Adding `sigma=I` (see below) fixed the residual
# pattern immediately, with no further acquisition changes needed. D1=0.1
# (exp224) has no clean replacement and is left out — it looked like an
# outright outlier independent of the weighting issue.
#
# VERIFIED FINAL NUMBERS (17/08, weighted fit, this DATASETS):
#   T1_fast = 0.0164 +/- 0.0044 s (2.87%), T1_slow = 25.5 +/- 0.84 s (97.13%)
DATASETS = {
    # D1 (s) : path to the Bruker experiment folder
    0.05: r"D:\Postdoc\Datas\LLZO-400-aug26\281",
    0.15: r"D:\Postdoc\Datas\LLZO-400-aug26\282",
    0.2:  r"D:\Postdoc\Datas\LLZO-400-aug26\283",
    0.3:  r"D:\Postdoc\Datas\LLZO-400-aug26\284",
    0.5:  r"D:\Postdoc\Datas\LLZO-400-aug26\285",
    0.7:  r"D:\Postdoc\Datas\LLZO-400-aug26\286",
    1:   r"D:\Postdoc\Datas\LLZO-400-aug26\227",
    2:   r"D:\Postdoc\Datas\LLZO-400-aug26\228",
    4:   r"D:\Postdoc\Datas\LLZO-400-aug26\229",
    8:   r"D:\Postdoc\Datas\LLZO-400-aug26\230",
    10:  r"D:\Postdoc\Datas\LLZO-400-aug26\231",
    20:  r"D:\Postdoc\Datas\LLZO-400-aug26\232",
    40:  r"D:\Postdoc\Datas\LLZO-400-aug26\233",
    80:  r"D:\Postdoc\Datas\LLZO-400-aug26\234",
    120: r"D:\Postdoc\Datas\LLZO-400-aug26\235",
    150: r"D:\Postdoc\Datas\LLZO-400-aug26\236",
}
LB = 10  # line broadening in Hz (10-200 Hz typical for solids)
PH0_MANUAL = -103.394  # PHC0 in degrees, used only if AUTO_PH0 = False
PH1 = -49.524  # PHC1 in degrees (first-order phase correction)
AUTO_PH0 = True  # True: automatic PH0 search by maximizing the real part
READ_PHASE_FROM_PROCS = False  # set True to read ph0/ph1 from TopSpin (procs) — takes priority over AUTO_PH0
REFERENCE_SHIFT_PPM = 2  # additive shift applied to the ppm axis (referencing) — same convention as pipeline_1d.py
ZF_FACTOR = 1  # zero-filling multiplier: total FFT length = N*(1+ZF_FACTOR) — so
                 # 0=none, 1=double, 3=quadruple (NOT 1=none/2=double/4=quadruple;
                 # fixed 18/08, same convention as pipeline_1d.py)
PEAK_PPM_WINDOW = (30, -30)  # window to search for the peak max (same span as your ZOOM)
OUTPUT_NAME = "T1_recovery_fit"
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, read_phase_from_procs=False, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.

    Built on the shared core.process_row / core.find_best_ph0 — same pattern
    as pipeline_1d.py, so both scripts treat a given dataset identically.
    Returns delta, spectrum, dic, AND the PH0 actually used (needed here to
    log it per D1 point, as a phasing-quality diagnostic across the series).
    """
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
            print(f"  Phase read from procs: PH0={ph0_deg:.3f}°, PH1={ph1_deg:.3f}°")
        except Exception:
            print("  Could not read phase from procs — using manual/auto value instead")
    elif auto_ph0:
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1_deg))
        print(f"  Optimal PH0 found (auto, PH1={ph1_deg:.3f}° accounted for): {ph0_deg:.3f}°")

    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1_deg), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm

    return delta, spectrum, dic, ph0_deg


def get_peak_intensity(delta, signal, ppm_window):
    """Simple total peak height within ppm_window — deliberately NOT a
    two-component decomposition. For a D1 series we only need a robust,
    reproducible total intensity per spectrum, and the shared-reference
    pseudo-Voigt decomposition proved too unstable run-to-run for this."""
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    window = signal[mask]
    idx = int(np.argmax(window))
    return float(window[idx]), float(delta[mask][idx])


def biexp_recovery(t, M0, f, T1a, T1b):
    """M0 * (1 - f*exp(-t/T1a) - (1-f)*exp(-t/T1b))"""
    return M0 * (1 - f * np.exp(-t / T1a) - (1 - f) * np.exp(-t / T1b))


if __name__ == "__main__":
    # === PROCESSING ===
    # Guarded under __main__ (fixed 18/08) so this module can be imported
    # (e.g. by tests, for biexp_recovery / get_peak_intensity / process_1d_spectrum)
    # without immediately trying to read the hardcoded DATASETS paths — same
    # convention already applied to relaxation_T2_components.py.
    D1_list, I_list, ph0_list = [], [], []

    for d1, path in sorted(DATASETS.items()):
        print(f"\nD1 = {d1} s  ({path})")
        try:
            delta, spectrum, bruker_dic, ph0_used = process_1d_spectrum(
                path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
                auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
                reference_shift_ppm=REFERENCE_SHIFT_PPM
            )
        except OSError as e:
            print(f"  SKIPPED — could not read dataset: {e}")
            continue
        intensity, peak_ppm = get_peak_intensity(delta, spectrum.real, PEAK_PPM_WINDOW)
        print(f"  peak at {peak_ppm:.3f} ppm, intensity = {intensity:.4e}")
        D1_list.append(d1)
        I_list.append(intensity)
        ph0_list.append(ph0_used)

    D1 = np.array(D1_list)
    I = np.array(I_list)

    if len(D1) < 5:
        # biexp_recovery has 4 free parameters (M0, f, T1a, T1b) — need strictly
        # more data points than parameters for curve_fit to be well-posed.
        print("\nNeed at least 5 usable points for a stable biexponential fit.")
        raise SystemExit

    p0 = [1.1 * I.max(), 0.2, 0.5, D1[D1 > D1.max() / 4].mean()]
    bounds = ([0.5 * I.max(), 0, 0.001, 1], [5 * I.max(), 1, 20, 500])
    # CAUTION (fixed 18/08): sigma=I was missing here. The T1 recovery curve spans
    # a wide dynamic range (weak signal at short D1, near-full recovery at long
    # D1) — an unweighted fit is dominated by the large-amplitude long-D1 points
    # and effectively ignores the short-D1 points where T1_fast lives. This is
    # the exact bug class already found and fixed once in this project (see
    # fit_T1_recovery.py in the working Library_nmr scripts) that had regressed
    # here in the Portfolio copy. Do not remove sigma=I without re-checking the
    # residuals at short D1.
    popt, pcov = curve_fit(biexp_recovery, D1, I, p0=p0, sigma=I, maxfev=50000, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    M0, f, T1fast, T1slow = popt
    M0e, fe, T1faste, T1slowe = perr

    print("\n--- T1 biexponential recovery fit ---")
    print(f"M0        = {M0:.4e} +/- {M0e:.2e}")
    print(f"T1_fast   = {T1fast:.3g} s  +/- {T1faste:.2g} s   (fraction = {f*100:.1f}% +/- {fe*100:.1f}%)")
    print(f"T1_slow   = {T1slow:.3g} s  +/- {T1slowe:.2g} s   (fraction = {(1-f)*100:.1f}%)")

    resid = I - biexp_recovery(D1, *popt)
    rel_resid = 100 * resid / I
    print("\nrelative residuals (%):", np.round(rel_resid, 2))
    if np.any(np.abs(rel_resid) > 15):
        print("WARNING: some points have >15% residual — check those spectra "
              "(bad phasing, wrong RG, or a point that needs excluding).")

    # --- export (same pandas/CSV convention as pipeline_1d.py) ---
    df = pd.DataFrame({"D1_s": D1, "Intensity": I, "PH0_deg": ph0_list, "residual_pct": rel_resid})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    # --- plot ---
    t_fit = np.logspace(np.log10(D1.min() / 2), np.log10(D1.max() * 1.3), 400)
    y_fit = biexp_recovery(t_fit, *popt)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(D1, I, color="blue", s=55, zorder=3, label="data")
    ax.plot(t_fit, y_fit, color="red", lw=1.5, zorder=2, label="biexponential fit")
    ax.set_xscale("log")
    ax.set_xlabel("Recovery delay D1 (s)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(r"$^7$Li T$_1$ recovery (onepulse series)")

    textstr = (
        f"$T_1$ slow = {T1slow:.1f} +/- {T1slowe:.1f} s  ({(1-f)*100:.1f}%)\n"
        f"$T_1$ fast = {T1fast:.2f} +/- {T1faste:.2f} s  ({f*100:.1f}%)"
    )
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=10.5,
             va="bottom", ha="right",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()
