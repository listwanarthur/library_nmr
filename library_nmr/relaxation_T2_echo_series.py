import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T2 SOLID-ECHO FIT — library_nmr (echosolide_v2, discrete L0 series)
# Usage: edit the CONFIGURATION block below, then run
#
# Independent, individually-phased onepulse-style spectra (like
# relaxation_T1_onepulse_series.py), total peak intensity vs echo delay L0
# (tau_echo = L0 * tau_rotor), using the calibrated p2 (not p1*2) for the
# refocusing pulse.
#
# The T2 decay spans ~2-3 orders of magnitude and is not single-exponential
# — an unweighted fit would be dominated by the first few large points and
# barely constrain the slow tail, so this fits with sigma=I. Do not remove
# without checking the residuals on the longest delays.
#
# Phase: PH0 is determined ONCE on the reference (best S/N) spectrum and
# FROZEN for the whole series — on a real overnight series, re-running
# AUTO_PH0 per spectrum let two low-S/N points lock onto noise instead of
# the real signal, silently corrupting their intensity.
# ============================================================

# === CONFIGURATION — only section to edit ===
TAU_ROTOR_US = 80.0  # rotor period (µs), MAS 12.5 kHz -> 80 µs

DATASETS = {
    # L0 (integer number of rotor periods) : path to the Bruker experiment folder
    1:   r"D:\Postdoc\Datas\LLZO-400-aug26\253",
    2:   r"D:\Postdoc\Datas\LLZO-400-aug26\254",
    4:   r"D:\Postdoc\Datas\LLZO-400-aug26\255",
    8:   r"D:\Postdoc\Datas\LLZO-400-aug26\256",
    16:  r"D:\Postdoc\Datas\LLZO-400-aug26\257",
    32:  r"D:\Postdoc\Datas\LLZO-400-aug26\258",
    40:  r"D:\Postdoc\Datas\LLZO-400-aug26\259",
    50:  r"D:\Postdoc\Datas\LLZO-400-aug26\260",
    80:  r"D:\Postdoc\Datas\LLZO-400-aug26\262",
    90:  r"D:\Postdoc\Datas\LLZO-400-aug26\263",
    100: r"D:\Postdoc\Datas\LLZO-400-aug26\264",
    # Extended tau points added 14-15/08 (exp270-275), also used to extend
    # relaxation_T2_components.py. NOTE: at long tau this GLOBAL fit mixes
    # narrow's and broad's differently-decaying tails, so treat T2_slow here
    # as a rough cross-check only — relaxation_T2_components.py's
    # per-component values are the reliable source.
    125: r"D:\Postdoc\Datas\LLZO-400-aug26\270",
    150: r"D:\Postdoc\Datas\LLZO-400-aug26\271",
    188: r"D:\Postdoc\Datas\LLZO-400-aug26\272",
    250: r"D:\Postdoc\Datas\LLZO-400-aug26\273",
    313: r"D:\Postdoc\Datas\LLZO-400-aug26\274",
    375: r"D:\Postdoc\Datas\LLZO-400-aug26\275",
}

# Points known to be at/below the noise floor (e.g. old 243 at L0=100 with
# NS=32): kept OUT of DATASETS/the fit, but you can list them here just to
# show them as "ND" markers on the plot. Leave empty if not needed.
BELOW_DETECTION = {
    # L0 : path
    # 100: r"D:\Postdoc\Datas\LLZO-400-aug26\243",
}

LB = 10  # line broadening in Hz (10-200 Hz typical for solids)
PH0_MANUAL = -103.394  # PHC0 in degrees, used only if AUTO_PH0 = False
PH1 = -49.524  # PHC1 in degrees (first-order phase correction)
AUTO_PH0 = True  # True: automatic PH0 search by maximizing the real part
READ_PHASE_FROM_PROCS = False  # set True to read ph0/ph1 from TopSpin (procs) — takes priority over AUTO_PH0
REFERENCE_SHIFT_PPM = 2  # additive shift applied to the ppm axis (referencing) — same convention as pipeline_1d.py
ZF_FACTOR = 1  # zero-filling multiplier: total FFT length = N*(1+ZF_FACTOR); 0=none, 1=double, 3=quadruple
PEAK_PPM_WINDOW = (6, -4)  # kept TIGHT around the real peak on purpose — a wide window
    # lets argmax lock onto a noise spike once S/N drops, corrupting both intensity
    # and position. Re-check against your actual peak position before running.
PHASE_REFERENCE_L0 = None  # L0 to determine PH0 from (frozen for the whole series). None = use the smallest L0 in DATASETS (best S/N).
OUTPUT_NAME = "T2_echosolide_fit"
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, read_phase_from_procs=False, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.
    Built on the shared core.process_row / core.find_best_ph0 — same pattern as
    pipeline_1d.py / relaxation_T1_onepulse_series.py, kept in sync on purpose.
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
    """Integrated intensity (trapezoidal area) within ppm_window, not argmax —
    a single-point max is noise-sensitive and biased upward at low S/N (the
    T2 series' weak tail); integrating averages the noise down instead."""
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    window_ppm = delta[mask]
    window_sig = signal[mask]
    peak_ppm = float(window_ppm[int(np.argmax(window_sig))])  # kept only for diagnostic printing
    sort_idx = np.argsort(window_ppm)
    integral = np.trapezoid(window_sig[sort_idx], window_ppm[sort_idx])
    return float(integral), peak_ppm


def biexp_decay(t, A1, T2fast, A2, T2slow):
    """A1*exp(-t/T2fast) + A2*exp(-t/T2slow)"""
    return A1 * np.exp(-t / T2fast) + A2 * np.exp(-t / T2slow)


if __name__ == "__main__":
    # === PROCESSING ===

    # --- determine PH0 once, on the reference spectrum, and freeze it ---
    ref_l0 = min(DATASETS.keys()) if PHASE_REFERENCE_L0 is None else PHASE_REFERENCE_L0
    ref_path = DATASETS[ref_l0]
    print(f"=== Determining phase from reference spectrum (L0={ref_l0}, best S/N) ===")
    _, _, _, ph0_frozen = process_1d_spectrum(
        ref_path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM
    )
    print(f"  PH0 frozen at {ph0_frozen:.3f}° for the whole series\n")

    L0_list, tau_list, I_list, ph0_list = [], [], [], []

    for l0, path in sorted(DATASETS.items()):
        tau_us = l0 * TAU_ROTOR_US
        print(f"L0 = {l0}  (tau_echo = {tau_us:.0f} µs)  ({path})")
        try:
            delta, spectrum, bruker_dic, ph0_used = process_1d_spectrum(
                path, LB, ph0_frozen, PH1, ZF_FACTOR,
                auto_ph0=False, read_phase_from_procs=False,
                reference_shift_ppm=REFERENCE_SHIFT_PPM
            )
        except OSError as e:
            print(f"  SKIPPED — could not read dataset: {e}")
            continue
        intensity, peak_ppm = get_peak_intensity(delta, spectrum.real, PEAK_PPM_WINDOW)
        print(f"  peak at {peak_ppm:.3f} ppm, intensity = {intensity:.4e}")
        L0_list.append(l0)
        tau_list.append(tau_us)
        I_list.append(intensity)
        ph0_list.append(ph0_used)

    tau = np.array(tau_list)
    I = np.array(I_list)

    if len(tau) < 5:
        # biexp_decay has 4 free params — need >4 points for curve_fit to be well-posed.
        print("\nNeed at least 5 usable points for a stable biexponential fit.")
        raise SystemExit

    p0 = [0.9 * I.max(), 150, 0.1 * I.max(), tau[tau > tau.max() / 4].mean()]
    bounds = ([0, 1, 0, 100], [5 * I.max(), 2000, 5 * I.max(), 50000])
    popt, pcov = curve_fit(biexp_decay, tau, I, p0=p0, sigma=I, maxfev=50000, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    A1, T2fast, A2, T2slow = popt
    A1e, T2faste, A2e, T2slowe = perr

    frac_fast = A1 / (A1 + A2) * 100
    frac_slow = A2 / (A1 + A2) * 100

    print("\n--- T2 biexponential decay fit (echo solide) ---")
    print(f"T2_fast = {T2fast:.1f} +/- {T2faste:.1f} us   (fraction = {frac_fast:.1f}%)")
    print(f"T2_slow = {T2slow:.1f} +/- {T2slowe:.1f} us   (fraction = {frac_slow:.1f}%)")

    resid = I - biexp_decay(tau, *popt)
    rel_resid = 100 * resid / I
    print("\nrelative residuals (%):", np.round(rel_resid, 2))
    if np.any(np.abs(rel_resid) > 15):
        print("WARNING: some points have >15% residual — check those spectra "
              "(bad phasing, wrong RG, or a point close to the noise floor).")

    # --- export ---
    df = pd.DataFrame({"L0": L0_list, "tau_echo_us": tau, "Intensity": I,
                        "PH0_deg": ph0_list, "residual_pct": rel_resid})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    # --- below-detection points (plotted only, never fitted) ---
    nd_tau, nd_I = [], []
    for l0, path in sorted(BELOW_DETECTION.items()):
        tau_us = l0 * TAU_ROTOR_US
        try:
            delta, spectrum, _, _ = process_1d_spectrum(
                path, LB, ph0_frozen, PH1, ZF_FACTOR,
                auto_ph0=False, read_phase_from_procs=False,
                reference_shift_ppm=REFERENCE_SHIFT_PPM
            )
            intensity, _ = get_peak_intensity(delta, spectrum.real, PEAK_PPM_WINDOW)
        except OSError:
            intensity = np.nan
        nd_tau.append(tau_us)
        nd_I.append(intensity)

    # --- plot (log-log: the decay spans ~2-3 decades, linear axes would hide the tail) ---
    t_fit = np.logspace(np.log10(tau.min() / 1.5), np.log10(tau.max() * 1.3), 400)
    y_fit = biexp_decay(t_fit, *popt)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t_fit, y_fit, color="red", lw=1.5, zorder=2, label="biexponential fit")
    ax.scatter(tau, I, color="blue", s=55, zorder=3, label="data")
    if nd_tau:
        ax.scatter(nd_tau, nd_I, marker="x", color="gray", s=70, zorder=3, label="below detection (ND)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Echo delay tau (us) = L0 x tau_rotor")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(r"$^7$Li T$_2$ solid-echo decay (discrete L0 series)")

    textstr = (
        f"$T_2$ slow = {T2slow:.0f} +/- {T2slowe:.0f} us  ({frac_slow:.1f}%)\n"
        f"$T_2$ fast = {T2fast:.0f} +/- {T2faste:.0f} us  ({frac_fast:.1f}%)"
    )
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=10.5,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    agr_series = [
        dict(x=tau, y=I, mode="symbol", color="blue", legend="data"),
        dict(x=t_fit, y=y_fit, mode="line", color="red", legend="biexponential fit"),
    ]
    if nd_tau:
        agr_series.append(dict(x=nd_tau, y=nd_I, mode="symbol", color="grey", legend="below detection (ND)"))
    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Echo delay tau (us) = L0 x tau_rotor", ylabel="Intensity (a.u.)",
               xlog=True, ylog=True, title="7Li T2 solid-echo decay (discrete L0 series)")
