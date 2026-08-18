import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit, nnls

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.fitting import pseudo_voigt, sum_pseudo_voigt, fit_group
from library_nmr.agr_export import export_agr

# ============================================================
# T2 PER-COMPONENT FIT — library_nmr (echosolide_v2, discrete L0 series)
# Usage: edit the CONFIGURATION block below, then run
#
# Why not a free two-component pseudo-Voigt fit on every spectrum: that
# decomposition (pipeline_1d.py's fit_group) is already flagged as unstable
# run-to-run (shape/eta drift, amplitudes going negative) -- worse here since
# S/N drops fast with increasing echo delay.
#
# Strategy used instead:
#   1) Fit ONE reference spectrum with the normal free fit_group to pin down
#      position/width/eta for each component.
#   2) FREEZE those shape parameters. For every other spectrum, only the two
#      amplitudes are then unknown -- linear in amplitude for fixed shape, so
#      solved via NNLS (non-negative least squares) instead of a nonlinear
#      fit, avoiding the unphysical negative amplitudes seen with a free fit.
#   3) Each component's amplitude vs echo delay is its own T2 decay curve,
#      fit independently (biexponential).
#
# Reference spectrum is NOT taken from this T2 series: the minimum echo delay
# achievable under MAS is one full rotor period (tau_rotor=80us, must be
# rotor-synchronized), and broad's own T2 is short enough that even the
# first T2 spectrum has already lost a large fraction of broad before it's
# measured -- using it as shape reference gave a distorted fit (eta_narrow
# pinned near 1, near-degenerate NNLS basis, broad vanishing beyond L0=2).
# FIX: the reference shape is now taken from a onepulse spectrum (longest
# D1, fully relaxed, best S/N -- only the receiver's us-scale dead time, not
# a rotor-period floor), which gives a balanced eta_narrow/integral split
# and keeps broad above zero on 10/11 T2 points instead of 2/11. Exact split
# percentages not reproduced here (public repo, unpublished manuscript).
# ============================================================

# === CONFIGURATION — only section to edit ===
TAU_ROTOR_US = 80.0  # rotor period (µs), MAS 12.5 kHz -> 80 µs

DATASETS = {
    # L0 (integer number of rotor periods) : path to the Bruker experiment folder
    # L0=65 (exp 261) excluded -- same outlier dropped in relaxation_T2_echo_series.py.
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
    # Extended tau points added 14-15/08 (exp270-275) to better constrain
    # T2_slow, fragile in the original series; higher NS (128/192) to
    # compensate for weaker signal at these long delays.
    125: r"D:\Postdoc\Datas\LLZO-400-aug26\270",
    150: r"D:\Postdoc\Datas\LLZO-400-aug26\271",
    188: r"D:\Postdoc\Datas\LLZO-400-aug26\272",
    250: r"D:\Postdoc\Datas\LLZO-400-aug26\273",
    313: r"D:\Postdoc\Datas\LLZO-400-aug26\274",
    375: r"D:\Postdoc\Datas\LLZO-400-aug26\275",
}

# Reference spectrum for the one-time free shape fit (STEP 1), deliberately
# OUTSIDE this T2 series -- exp236 = relaxation_T1_onepulse_series.py's
# D1=150s point (fully relaxed, best S/N, no rotor-period dead time). See
# module docstring.
REFERENCE_PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\236"

# Per-component decay model for STEP 3 -- both components need TWO
# relaxation times (biexp), not one. broad becomes undetectable (NNLS
# zeroes it) beyond L0=188 (tau~15ms) -- a genuine physical limit (broad's
# slow tail is below the noise floor past ~15ms), not a fitting problem.
# Exact fitted T2_fast/T2_slow values not reproduced here (public repo,
# unpublished manuscript). Figure: T2_components_fit_v3.png.
COMPONENT_DECAY_MODEL = {"narrow (fine)": "biexp", "broad (large)": "biexp"}

LB = 10
PH0_MANUAL = -103.394
PH1 = -49.524
AUTO_PH0 = True
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
OUTPUT_NAME = "T2_components_fit"

# Same convention as pipeline_1d.py's PEAKS -- used ONCE, on the reference
# spectrum only, to determine position/width/eta for each component. The
# split from an ECHO spectrum (dead-time-limited) was wrong, skewed toward
# narrow; the corrected split (onepulse reference) shows broad as the
# MAJORITY population. Exact percentages not reproduced here (public repo,
# unpublished manuscript).
REFERENCE_PEAKS = {
    "ppm_min": -23, "ppm_max": 27,
    "p0": [
        [3.5e7, 0.6, 5.8,  0.99],  # narrow component (fine, corrected reference)
        [1.0e7, 0.6, 15,   0.5 ],  # broad component (large, eta=0, majority population, corrected reference)
    ],
    "eta_fixed": [None, 0.0],
    "width_bounds": [(0, 8), (8, 100)],
    "position_bounds": [(-4, 5), (-4, 5)],
}
COMPONENT_NAMES = ["narrow (fine)", "broad (large)"]
COMPONENT_COLORS = ["darkorange", "purple"]

# Points known to be at/below the noise floor: plotted as ND markers, never fitted.
BELOW_DETECTION = {
    # L0 : path
}
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, read_phase_from_procs=False, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.
    Built on the shared core.process_row / core.find_best_ph0 -- same pattern as
    pipeline_1d.py / relaxation_T1_onepulse_series.py / relaxation_T2_echo_series.py.
    """
    dic, data = ng.bruker.read(path, read_procs=read_phase_from_procs)

    grpdly_shift = find_grpdly_shift(dic)
    if grpdly_shift > 0:
        print(f"  GRPDLY corrected: shifted by {grpdly_shift} points")
    else:
        print("  GRPDLY not found or zero -- no correction applied")

    N = data.shape[0]
    dt = 1 / dic["acqus"]["SW_h"]
    data_zf = np.concatenate([data, np.zeros(zf_factor * N, dtype=complex)])

    ph0_deg = ph0_manual
    ph1_deg = ph1
    if read_phase_from_procs:
        try:
            ph0_deg = dic["procs"]["PHC0"]
            ph1_deg = dic["procs"]["PHC1"]
            print(f"  Phase read from procs: PH0={ph0_deg:.3f} deg, PH1={ph1_deg:.3f} deg")
        except Exception:
            print("  Could not read phase from procs -- using manual/auto value instead")
    elif auto_ph0:
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1_deg))
        print(f"  Optimal PH0 found (auto, PH1={ph1_deg:.3f} deg accounted for): {ph0_deg:.3f} deg")

    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1_deg), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm

    return delta, spectrum, dic, ph0_deg


# pseudo_voigt / sum_pseudo_voigt / fit_group imported from library_nmr.fitting
# (drop-in replacement, no behavior change).

def fit_amplitudes_fixed_shape(delta, signal, ppm_min, ppm_max, components):
    """Solves for the component amplitudes ONLY, with position/width/eta
    frozen to the reference values. Linear in A -> solved with NNLS
    (non-negative least squares) instead of a nonlinear optimizer: faster,
    always converges, and can't return the unphysical negative amplitudes
    seen with the free per-spectrum decomposition."""
    mask = (delta >= ppm_min) & (delta <= ppm_max)
    x = delta[mask]
    y = signal[mask]
    basis = np.column_stack([
        pseudo_voigt(x, 1.0, c["position"], c["width"], c["eta"]) for c in components
    ])
    amps, _ = nnls(basis, y)
    return amps


def monoexp_decay(t, A0, T2):
    return A0 * np.exp(-t / T2)


def biexp_decay(t, A1, T2fast, A2, T2slow):
    """Same functional form as relaxation_T2_echo_series.py's global fit."""
    return A1 * np.exp(-t / T2fast) + A2 * np.exp(-t / T2slow)


def eval_decay(model, t, popt):
    if model == "biexp":
        return biexp_decay(t, *popt)
    return monoexp_decay(t, *popt)


if __name__ == "__main__":
    # === STEP 1: reference shape fit, from the EXTERNAL onepulse reference
    # (see module docstring). PH0 for the T2 series itself is frozen from
    # L0=1 of DATASETS -- phase is per-experiment/per-probe-tuning, not
    # something the onepulse reference on a different day can supply. ===
    print(f"\n=== Reference SHAPE fit (external onepulse, {REFERENCE_PATH}) ===")
    delta_ref, spectrum_ref, _, _ = process_1d_spectrum(
        REFERENCE_PATH, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM
    )

    first_l0 = min(DATASETS.keys())
    print(f"\n=== Reference PHASE fit (from this T2 series, L0={first_l0}, {DATASETS[first_l0]}) ===")
    _, _, _, ph0_frozen = process_1d_spectrum(
        DATASETS[first_l0], LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM
    )
    print(f"  PH0 frozen at {ph0_frozen:.3f} deg for the whole series")
    ref_results = fit_group(
        delta_ref, spectrum_ref.real,
        REFERENCE_PEAKS["ppm_min"], REFERENCE_PEAKS["ppm_max"], REFERENCE_PEAKS["p0"],
        eta_fixed_list=REFERENCE_PEAKS.get("eta_fixed"),
        width_bounds_list=REFERENCE_PEAKS.get("width_bounds"),
        position_bounds_list=REFERENCE_PEAKS.get("position_bounds"),
    )
    for name, r in zip(COMPONENT_NAMES, ref_results):
        print(f"  {name}: position={r['position']:.3f} ppm, width={r['width']:.3f} ppm, "
              f"eta={r['eta']:.3f}  (FROZEN for the rest of the fit)")

    components = [{"position": r["position"], "width": r["width"], "eta": r["eta"]} for r in ref_results]
    n_comp = len(components)

    # === STEP 2: fixed-shape amplitude fit on every spectrum of the series ===
    L0_list, tau_list = [], []
    amp_series = [[] for _ in range(n_comp)]

    for l0, path in sorted(DATASETS.items()):
        tau_us = l0 * TAU_ROTOR_US
        print(f"\nL0 = {l0}  (tau_echo = {tau_us:.0f} µs)  ({path})")
        try:
            delta, spectrum, _, _ = process_1d_spectrum(
                path, LB, ph0_frozen, PH1, ZF_FACTOR,
                auto_ph0=False, read_phase_from_procs=False,
                reference_shift_ppm=REFERENCE_SHIFT_PPM
            )
        except OSError as e:
            print(f"  SKIPPED — could not read dataset: {e}")
            continue
        amps = fit_amplitudes_fixed_shape(delta, spectrum.real, REFERENCE_PEAKS["ppm_min"],
                                           REFERENCE_PEAKS["ppm_max"], components)
        for name, a in zip(COMPONENT_NAMES, amps):
            print(f"  {name}: amplitude = {a:.4e}" +
                  ("  (zeroed by NNLS — below detection for this component)" if a <= 0 else ""))
        L0_list.append(l0)
        tau_list.append(tau_us)
        for i in range(n_comp):
            amp_series[i].append(amps[i])

    tau = np.array(tau_list)

    # === STEP 3: independent T2 fit per component (model per COMPONENT_DECAY_MODEL) ===
    # fit_params[i] = {"model": ..., "popt": (...), "perr": (...)} or None if not fitted.
    fit_params = []
    for i in range(n_comp):
        name = COMPONENT_NAMES[i]
        model = COMPONENT_DECAY_MODEL.get(name, "monoexp")
        y = np.array(amp_series[i])
        usable = y > 0  # NNLS can zero out a point -> drop it rather than fit log(0)
        n_needed = 5 if model == "biexp" else 3  # biexp has 4 free params -> need >4 points to constrain it
        if usable.sum() < n_needed:
            print(f"\n{name}: only {usable.sum()} usable (>0) points — cannot fit "
                  f"({model} needs >= {n_needed}), skipping.")
            if usable.sum() == 2:
                t_sub, y_sub = tau[usable][:2], y[usable][:2]
                try:
                    if y_sub[0] <= y_sub[1]:
                        raise ValueError("no decay between the two points")
                    T2_rough = (t_sub[1] - t_sub[0]) / np.log(y_sub[0] / y_sub[1])
                    print(f"  crude 2-point T2 estimate (NOT a real fit, huge uncertainty): ~{T2_rough:.0f} us")
                except (ValueError, ZeroDivisionError):
                    print("  crude 2-point T2 estimate: not computable (points equal or non-decaying)")
            fit_params.append(None)
            continue
        t_fit_in, y_fit_in = tau[usable], y[usable]
        try:
            if model == "biexp":
                p0 = [0.9 * y_fit_in.max(), 150, 0.1 * y_fit_in.max(),
                      t_fit_in[t_fit_in > t_fit_in.max() / 4].mean()]
                bounds = ([0, 1, 0, 100], [5 * y_fit_in.max(), 2000, 5 * y_fit_in.max(), 50000])
                popt, pcov = curve_fit(biexp_decay, t_fit_in, y_fit_in, p0=p0, sigma=y_fit_in,
                                        bounds=bounds, maxfev=50000)
                perr = np.sqrt(np.diag(pcov))
                fit_params.append({"model": model, "popt": tuple(popt), "perr": tuple(perr)})
                frac_fast = 100 * popt[0] / (popt[0] + popt[2])
                print(f"\n{name}: T2_fast = {popt[1]:.1f} +/- {perr[1]:.1f} us  (fraction {frac_fast:.1f}%)  "
                      f"T2_slow = {popt[3]:.1f} +/- {perr[3]:.1f} us  (fraction {100 - frac_fast:.1f}%)  "
                      f"(fit on {usable.sum()}/{len(y)} points)")
            else:
                p0 = [y_fit_in.max(), t_fit_in[len(t_fit_in)//2]]
                popt, pcov = curve_fit(monoexp_decay, t_fit_in, y_fit_in, p0=p0, sigma=y_fit_in,
                                        bounds=([0, 1], [10 * y_fit_in.max(), 200000]), maxfev=20000)
                perr = np.sqrt(np.diag(pcov))
                fit_params.append({"model": model, "popt": tuple(popt), "perr": tuple(perr)})
                print(f"\n{name}: T2 = {popt[1]:.1f} +/- {perr[1]:.1f} us  (fit on {usable.sum()}/{len(y)} points)")
            rel_resid = 100 * (y_fit_in - eval_decay(model, t_fit_in, popt)) / y_fit_in
            print(f"  relative residuals (%): {np.round(rel_resid, 2)}")
            if np.any(np.abs(rel_resid) > 20):
                print("  WARNING: some points have >20% residual — check phasing/S-N on those spectra.")
        except RuntimeError as e:
            print(f"\n{name}: fit failed ({e})")
            fit_params.append(None)

    # === export ===
    df = pd.DataFrame({"L0": L0_list, "tau_echo_us": tau})
    for i, name in enumerate(COMPONENT_NAMES):
        df[f"amplitude_{name.split()[0]}"] = amp_series[i]
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    # === plot ===
    fig, ax = plt.subplots(figsize=(8, 5))

    textlines = []
    agr_series = []
    for i, name in enumerate(COMPONENT_NAMES):
        y = np.array(amp_series[i])
        usable = y > 0
        ax.scatter(tau[usable], y[usable], color=COMPONENT_COLORS[i], s=55, zorder=3, label=f"{name} — data")
        agr_series.append(dict(x=tau[usable], y=y[usable], mode="symbol",
                                color=COMPONENT_COLORS[i], legend=f"{name} - data"))
        if fit_params[i] is not None:
            model, popt, perr = fit_params[i]["model"], fit_params[i]["popt"], fit_params[i]["perr"]
            t_fit = np.logspace(np.log10(tau.min() / 1.5), np.log10(tau.max() * 1.3), 400)
            ax.plot(t_fit, eval_decay(model, t_fit, popt), color=COMPONENT_COLORS[i], lw=1.5, zorder=2,
                    label=f"{name} — fit")
            agr_series.append(dict(x=t_fit, y=eval_decay(model, t_fit, popt), mode="line",
                                    color=COMPONENT_COLORS[i], legend=f"{name} - fit"))
            if model == "biexp":
                A1, T2fast, A2, T2slow = popt
                _, T2faste, _, T2slowe = perr
                frac_fast = 100 * A1 / (A1 + A2)
                textlines.append(f"{name}: T2fast={T2fast:.0f}+/-{T2faste:.0f}us ({frac_fast:.0f}%), "
                                  f"T2slow={T2slow:.0f}+/-{T2slowe:.0f}us ({100-frac_fast:.0f}%)")
            else:
                _, T2 = popt
                _, T2e = perr
                textlines.append(f"T2 {name} = {T2:.0f} +/- {T2e:.0f} us")

    # below-detection points (never fitted)
    nd_tau, nd_y = [], []
    for l0, path in sorted(BELOW_DETECTION.items()):
        tau_us = l0 * TAU_ROTOR_US
        try:
            delta, spectrum, _, _ = process_1d_spectrum(
                path, LB, ph0_frozen, PH1, ZF_FACTOR,
                auto_ph0=False, read_phase_from_procs=False,
                reference_shift_ppm=REFERENCE_SHIFT_PPM
            )
            amps_nd = fit_amplitudes_fixed_shape(delta, spectrum.real, REFERENCE_PEAKS["ppm_min"],
                                                  REFERENCE_PEAKS["ppm_max"], components)
            nd_tau.append(tau_us)
            nd_y.append(sum(amps_nd))
        except OSError:
            pass
    if nd_tau:
        ax.scatter(nd_tau, nd_y, marker="x", color="gray", s=70, zorder=3, label="below detection (ND)")
        agr_series.append(dict(x=nd_tau, y=nd_y, mode="symbol", color="grey", legend="below detection (ND)"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Echo delay tau (us) = L0 x tau_rotor")
    ax.set_ylabel("Component amplitude (a.u.)")
    ax.set_title(r"$^7$Li T$_2$ per component (fixed shape, NNLS amplitudes)")
    if textlines:
        ax.text(0.97, 0.05, "\n".join(textlines), transform=ax.transAxes, fontsize=10,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Echo delay tau (us) = L0 x tau_rotor", ylabel="Component amplitude (a.u.)",
               xlog=True, ylog=True,
               title="7Li T2 per component (fixed shape, NNLS amplitudes)")
