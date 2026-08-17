import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0

# ============================================================
# T1(T) — VARIABLE-TEMPERATURE T1 RECOVERY, 7Li LLZO, STATIC PROBE 300M
# library_nmr — companion to relaxation_T1_onepulse_series.py
# Usage: fill in the DATA paths in the TEMPERATURES dict below (one onepulse
# D1 series per temperature, exactly like relaxation_T1_onepulse_series.py but
# repeated 7 times), then run once. Produces one CSV + two figures for the
# whole VT-T1 series instead of 7 manual re-runs.
#
# Follows protocole_T1_temperature.docx (prepared 13/08, checklist there
# still valid — recalibrate p1 on this probe, spot-check nutation per
# temperature, retune per temperature, etc. This script only replaces the
# "run relaxation_T1_onepulse_series.py by hand at each T" step).
#
# Processing (GRPDLY correction, apodization, FFT, phasing) is built on the
# same shared core.process_row / core.find_best_ph0 as every other script in
# this package, so a given D1 point processed by either script gives the
# same intensity.
#
# Model selection per temperature follows the same rule already stated in
# the article's Materials and method (Relaxation curve fitting): biexponential
# if >=5 usable points, monoexponential if 3-4 points, a simple two-point
# estimate if only 2 points are usable. The short "survol" grids (7 points)
# should support a biexponential fit but may leave T1_fast poorly constrained
# at some temperatures — check the printed uncertainties, don't just trust
# convergence.
#
# IMPORTANT — reference value for the RT sanity check below:
# the 298K point in this VT series is meant to reproduce the 400M MAS
# room-temperature result under the new static-probe configuration. Compare
# it against the CORRECTED, weighted-fit value from
# relaxation_T1_onepulse_series.py (17/08 fix) — NOT the older, unweighted
# value quoted in protocole_T1_temperature.docx, which predates that fix
# and is superseded. Fill in REF_298K_T1_SLOW / REF_298K_T1_FAST (and their
# error bars) below from your own verified fit before running this script —
# deliberately left blank here since this repository is public and those
# numbers are part of an unpublished manuscript. The script checks this
# automatically once filled in.
# ============================================================

# === CONFIGURATION — only section to edit ===
#
# One entry per temperature. D1 grids and NS below are copied verbatim from
# protocole_T1_temperature.docx (13/08) — edit only if the actual grid used
# on the day differed. Fill in "TODO" with the Bruker experiment folder for
# each D1 point once acquired (same convention as relaxation_T1_onepulse_series.py's
# DATASETS dict: D1 in seconds -> path).
TEMPERATURES = {
    200: {
        "T_C": -73,
        "NS": 16,
        "note": "limite basse",
        "D1_paths": {
            0.05: r"D:\Postdoc\Datas\TODO_LLZO-300-VT\200K\TODO",
            0.3:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\200K\TODO",
            1.5:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\200K\TODO",
            6:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\200K\TODO",
            20:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\200K\TODO",
            60:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\200K\TODO",
            150:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\200K\TODO",
        },
    },
    230: {
        "T_C": -43,
        "NS": 16,
        "note": "proche limite basse 400M MAS (-40C) -- recoupement possible",
        "D1_paths": {
            0.05: r"D:\Postdoc\Datas\TODO_LLZO-300-VT\230K\TODO",
            0.3:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\230K\TODO",
            1.5:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\230K\TODO",
            6:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\230K\TODO",
            20:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\230K\TODO",
            60:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\230K\TODO",
            150:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\230K\TODO",
        },
    },
    260: {
        "T_C": -13,
        "NS": 16,
        "note": "proche -10C prevu sur 400M",
        "D1_paths": {
            0.05: r"D:\Postdoc\Datas\TODO_LLZO-300-VT\260K\TODO",
            0.3:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\260K\TODO",
            1.5:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\260K\TODO",
            6:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\260K\TODO",
            20:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\260K\TODO",
            60:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\260K\TODO",
            150:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\260K\TODO",
        },
    },
    298: {
        "T_C": 25,
        "NS": 32,
        "note": "RT -- reference, grille complete, mesuree en premier",
        "D1_paths": {
            0.05: r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            0.2:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            0.5:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            1:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            2:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            4:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            8:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            15:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            30:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            60:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            120:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
            200:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\298K\TODO",
        },
    },
    330: {
        "T_C": 57,
        "NS": 16,
        "note": "proche limite haute 400M MAS (60C) -- recoupement possible",
        "D1_paths": {
            0.05: r"D:\Postdoc\Datas\TODO_LLZO-300-VT\330K\TODO",
            0.3:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\330K\TODO",
            1.5:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\330K\TODO",
            6:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\330K\TODO",
            20:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\330K\TODO",
            60:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\330K\TODO",
            150:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\330K\TODO",
        },
    },
    360: {
        "T_C": 87,
        "NS": 16,
        "note": "",
        "D1_paths": {
            0.05: r"D:\Postdoc\Datas\TODO_LLZO-300-VT\360K\TODO",
            0.3:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\360K\TODO",
            1.5:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\360K\TODO",
            6:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\360K\TODO",
            20:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\360K\TODO",
            60:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\360K\TODO",
            150:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\360K\TODO",
        },
    },
    400: {
        "T_C": 127,
        "NS": 16,
        "note": "limite haute",
        "D1_paths": {
            0.05: r"D:\Postdoc\Datas\TODO_LLZO-300-VT\400K\TODO",
            0.3:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\400K\TODO",
            1.5:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\400K\TODO",
            6:    r"D:\Postdoc\Datas\TODO_LLZO-300-VT\400K\TODO",
            20:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\400K\TODO",
            60:   r"D:\Postdoc\Datas\TODO_LLZO-300-VT\400K\TODO",
            150:  r"D:\Postdoc\Datas\TODO_LLZO-300-VT\400K\TODO",
        },
    },
}

# Reference RT values for the sanity check (weighted fit, 17/08 — see
# relaxation_T1_onepulse_series.py). Do NOT replace with the older unweighted
# numbers from protocole_T1_temperature.docx. Left as None here (public
# repo, unpublished numbers) — fill in from your own results before running;
# the sanity check below is skipped automatically while these are None.
REF_298K_T1_SLOW = None      # s -- e.g. 25.5
REF_298K_T1_SLOW_ERR = None  # s
REF_298K_T1_FAST = None      # s
REF_298K_T1_FAST_ERR = None  # s
RT_SANITY_TOLERANCE = 0.20   # warn if the new 298K T1_slow differs by more than 20%

LB = 10  # line broadening in Hz — same as relaxation_T1_onepulse_series.py; revisit if the
         # static-probe lineshape needs a different value
PH0_MANUAL = 0.0   # PHC0 fallback, degrees — recalibrate once you have real spectra
PH1 = 0.0          # PHC1 fallback, degrees
AUTO_PH0 = True    # automatic PH0 search by maximizing the real part
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 0.0   # re-referencing offset — recalibrate for the static probe/300M
ZF_FACTOR = 1  # zero-filling multiplier: total FFT length = N*(1+ZF_FACTOR) — so
                 # 0=none, 1=double, 3=quadruple (NOT 1=none/2=double/4=quadruple;
                 # same convention as pipeline_1d.py)
PEAK_PPM_WINDOW = (100, -100)  # static powder pattern is much wider than the MAS
                                 # spectrum (see protocol checklist) — widen/narrow
                                 # after looking at the first real spectrum
MIN_POINTS_BIEXP = 5   # matches the article's Materials and method rule
MIN_POINTS_MONOEXP = 3
OUTPUT_NAME = "T1_vs_temperature"
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, read_phase_from_procs=False, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.
    Built on the shared core.process_row / core.find_best_ph0 -- same pattern as
    pipeline_1d.py / relaxation_T1_onepulse_series.py.
    """
    dic, data = ng.bruker.read(path, read_procs=read_phase_from_procs)

    grpdly_shift = find_grpdly_shift(dic)
    if grpdly_shift > 0:
        print(f"    GRPDLY corrected: shifted by {grpdly_shift} points")
    else:
        print("    GRPDLY not found or zero -- no correction applied")

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
            print("    Could not read phase from procs -- using manual/auto value instead")
    elif auto_ph0:
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1_deg))

    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1_deg), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm

    return delta, spectrum, dic, ph0_deg


def get_peak_intensity(delta, signal, ppm_window):
    """Simple total peak height within ppm_window (see relaxation_T1_onepulse_series.py)."""
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    window = signal[mask]
    idx = int(np.argmax(window))
    return float(window[idx]), float(delta[mask][idx])


def biexp_recovery(t, M0, f, T1a, T1b):
    return M0 * (1 - f * np.exp(-t / T1a) - (1 - f) * np.exp(-t / T1b))


def monoexp_recovery(t, M0, T1):
    return M0 * (1 - np.exp(-t / T1))


def fit_one_temperature(T_K, cfg):
    """Process one temperature's D1 series and fit T1, following the same
    biexp -> monoexp -> two-point fallback rule as the article's Materials
    and method (Relaxation curve fitting)."""
    print(f"\n=== T = {T_K} K ({cfg['T_C']} C) ===  {cfg.get('note', '')}")
    D1_list, I_list = [], []
    for d1, path in sorted(cfg["D1_paths"].items()):
        if "TODO" in path:
            print(f"  D1={d1}s: path not filled in yet -- skipping")
            continue
        try:
            delta, spectrum, _, ph0_used = process_1d_spectrum(
                path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
                auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
                reference_shift_ppm=REFERENCE_SHIFT_PPM,
            )
        except OSError as e:
            print(f"  D1={d1}s: SKIPPED -- could not read dataset: {e}")
            continue
        intensity, peak_ppm = get_peak_intensity(delta, spectrum.real, PEAK_PPM_WINDOW)
        print(f"  D1={d1:>6.2f}s  peak at {peak_ppm:7.2f} ppm  intensity={intensity:.4e}")
        D1_list.append(d1)
        I_list.append(intensity)

    D1 = np.array(D1_list)
    I = np.array(I_list)
    n = len(D1)

    result = {
        "T_K": T_K, "T_C": cfg["T_C"], "n_points": n, "model": None,
        "M0": np.nan, "T1_fast": np.nan, "T1_fast_err": np.nan, "frac_fast": np.nan,
        "T1_slow": np.nan, "T1_slow_err": np.nan, "frac_slow": np.nan,
    }
    if n == 0:
        print("  No usable data yet for this temperature.")
        return result

    if n >= MIN_POINTS_BIEXP:
        try:
            p0 = [1.1 * I.max(), 0.2, 0.5, D1[D1 > D1.max() / 4].mean()]
            bounds = ([0.5 * I.max(), 0, 0.001, 1], [5 * I.max(), 1, 20, 500])
            # sigma=I weighting is required -- see the module docstring in
            # relaxation_T1_onepulse_series.py / relaxation_T1_components.py
            # for the history of this exact bug (unweighted fits bias
            # T1_fast, previously found and fixed, once regressed).
            popt, pcov = curve_fit(biexp_recovery, D1, I, p0=p0, sigma=I, maxfev=50000, bounds=bounds)
            perr = np.sqrt(np.diag(pcov))
            M0, f, T1fast, T1slow = popt
            M0e, fe, T1faste, T1slowe = perr
            result.update(model="biexponential", M0=M0,
                           T1_fast=T1fast, T1_fast_err=T1faste, frac_fast=f * 100,
                           T1_slow=T1slow, T1_slow_err=T1slowe, frac_slow=(1 - f) * 100)
            print(f"  biexponential fit: T1_fast={T1fast:.3g}+/-{T1faste:.2g}s ({f*100:.1f}%), "
                  f"T1_slow={T1slow:.3g}+/-{T1slowe:.2g}s ({(1-f)*100:.1f}%)")
        except RuntimeError as e:
            print(f"  biexponential fit FAILED ({e}) -- falling back to monoexponential")
            n = MIN_POINTS_MONOEXP  # force fallthrough to the monoexp branch below

    if result["model"] is None and n >= MIN_POINTS_MONOEXP:
        popt, pcov = curve_fit(monoexp_recovery, D1, I, p0=[1.1 * I.max(), D1.mean()], sigma=I, maxfev=50000)
        perr = np.sqrt(np.diag(pcov))
        M0, T1 = popt
        M0e, T1e = perr
        result.update(model="monoexponential", M0=M0, T1_slow=T1, T1_slow_err=T1e, frac_slow=100.0)
        print(f"  monoexponential fit: T1={T1:.3g}+/-{T1e:.2g}s")
    elif result["model"] is None and n == 2:
        # Simple two-point estimate, as used in the article for the sparsest series.
        # Needs an assumed fully-relaxed plateau Iinf strictly above both points --
        # with only 2 points there is no way to estimate Iinf from the data itself
        # (unlike I.max(), which just equals the longer-D1 point here and makes the
        # log() below blow up). Uses a fixed 20% headroom above the longer-D1 point
        # as a rough plateau; treat the result as indicative only and refine by hand.
        t1, t2 = D1
        i1, i2 = I
        Iinf = 1.2 * max(i1, i2)
        T1_est = np.nan
        try:
            with np.errstate(divide="raise", invalid="raise"):
                T1_est = (t2 - t1) / np.log((Iinf - i1) / (Iinf - i2))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            T1_est = np.nan
        if np.isfinite(T1_est) and T1_est > 0:
            result.update(model="two-point estimate", T1_slow=T1_est, frac_slow=100.0)
            print(f"  two-point estimate: T1~={T1_est:.3g}s (rough, assumed Iinf=1.2*max -- check manually)")
        else:
            print("  two-point estimate failed (non-physical result -- check for a plateau/outlier issue)")
    elif result["model"] is None:
        print(f"  Only {n} usable point(s) -- not enough to fit anything yet.")

    return result


if __name__ == "__main__":
    # === PROCESSING ===
    results = [fit_one_temperature(T_K, cfg) for T_K, cfg in sorted(TEMPERATURES.items())]
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    # --- RT sanity check against the corrected (weighted-fit) reference ---
    rt_row = df[df["T_K"] == 298]
    if REF_298K_T1_SLOW is None:
        print("\n--- RT (298K) sanity check --- SKIPPED (REF_298K_T1_SLOW is None; "
              "fill it in from your own verified fit -- see module docstring)")
    elif not rt_row.empty and not np.isnan(rt_row["T1_slow"].iloc[0]):
        t1s = rt_row["T1_slow"].iloc[0]
        dev = abs(t1s - REF_298K_T1_SLOW) / REF_298K_T1_SLOW
        print(f"\n--- RT (298K) sanity check ---")
        print(f"  New T1_slow = {t1s:.2f} s  vs  reference (weighted fit, 17/08) = "
              f"{REF_298K_T1_SLOW} +/- {REF_298K_T1_SLOW_ERR} s")
        if dev > RT_SANITY_TOLERANCE:
            print(f"  WARNING: deviates by {dev*100:.0f}% from the corrected RT reference -- "
                  f"check probe calibration/phasing before trusting the rest of the series. "
                  f"(Do NOT compare against the older, superseded unweighted value from "
                  f"protocole_T1_temperature.docx -- that number predates the 17/08 weighting fix.)")
        else:
            print(f"  OK -- within {RT_SANITY_TOLERANCE*100:.0f}% of the corrected reference.")

    # --- Plot 1: T1 vs temperature ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    valid = df.dropna(subset=["T1_slow"])
    ax.errorbar(valid["T_K"], valid["T1_slow"], yerr=valid["T1_slow_err"],
                fmt="o-", color="purple", label="T1 slow (or single T1)", capsize=3)
    biexp = valid[valid["model"] == "biexponential"]
    if not biexp.empty:
        ax.errorbar(biexp["T_K"], biexp["T1_fast"], yerr=biexp["T1_fast_err"],
                    fmt="s-", color="orange", label="T1 fast", capsize=3)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("T1 (s)")
    ax.set_yscale("log")
    ax.set_title(r"$^7$Li T$_1$(T) — static probe, 300M")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}_vs_T.pdf")

    # --- Plot 2: Arrhenius-style plot (ln(1/T1) vs 1000/T) ---
    # Deliberately NOT auto-fitting an activation energy here: with only 7
    # temperatures spanning 200-400K, it is unknown a priori whether a T1
    # minimum falls inside this window. An Arrhenius/BPP activation energy is
    # only meaningful on ONE side of the minimum (if any) -- fit that by hand
    # on whichever subset looks linear once you see the real data shape,
    # don't regress blindly through all 7 points.
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    inv_T1_slow = 1.0 / valid["T1_slow"]
    ax2.plot(1000.0 / valid["T_K"], np.log(inv_T1_slow), "o-", color="purple", label="ln(1/T1 slow)")
    ax2.set_xlabel("1000 / T (K$^{-1}$)")
    ax2.set_ylabel(r"ln(1/T$_1$)")
    ax2.set_title("Arrhenius-style plot — inspect shape before fitting Ea by hand")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}_arrhenius.pdf")

    plt.show()
    print("\nDone. Figures saved as "
          f"{OUTPUT_NAME}_vs_T.pdf and {OUTPUT_NAME}_arrhenius.pdf")
