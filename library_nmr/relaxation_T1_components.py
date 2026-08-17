import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit, nnls

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.fitting import pseudo_voigt, sum_pseudo_voigt, fit_group

# ============================================================
# T1 PER-COMPONENT FIT — library_nmr (onepulse D1 series)
# Usage: edit the CONFIGURATION block below, then run
#
# Same strategy as relaxation_T2_components.py: fit ONE reference spectrum
# freely (pipeline_1d.py's two-component pseudo-Voigt), FREEZE
# position/width/eta, then solve for the two amplitudes at every D1 with
# NNLS (linear in amplitude for fixed shape), and fit each component's own
# amplitude vs D1 recovery curve independently.
#
# Reference spectrum: unlike relaxation_T2_components.py, there is no
# dead-time concern here — a onepulse experiment has only the receiver's
# hardware dead time (a few us), not a rotor-period floor, so the LONGEST
# D1 in this same series (best S/N, fully relaxed) is a perfectly good,
# undistorted reference. No external reference spectrum needed.
#
# Result you should expect (verified on the validated series, exp224-236,
# BEFORE the 18/08 sigma= weighting fix below — re-verify this range after
# re-running with the fix, the numbers will likely shift; see fit_T1_components.py
# in Library_nmr for what the equivalent fix changed there):
# narrow and broad come out with ESSENTIALLY THE SAME T1_slow (within
# error bars) and a similar small T1_fast fraction (~3.5-4.5%, PRE-FIX), even
# though their T2 is very different (T2_fast: narrow ~200-250us vs broad
# ~150us — see relaxation_T2_components.py). This is expected, not a bug:
# T1 is driven by fluctuations at the Larmor frequency, typically
# dominated by a few relaxation "sinks" (paramagnetic impurities/defects)
# rather than the local static coupling that sets T2. In a rigidly
# dipolar-coupled lattice, spin diffusion can homogenize T1 across
# structurally different sites on the (seconds-long) T1 timescale, even
# though it's far too slow to do so on the (microsecond) T2 timescale —
# hence T2 tells narrow and broad apart, T1 mostly doesn't. See project
# notes for the full discussion and the T1(T)/T2(T) comparison this
# motivates.
#
# CAUTION: the T1_fast VALUE (not its ~4% fraction) is poorly constrained
# by this D1 grid in every fit attempted so far (total intensity, narrow,
# broad) — it converges near its lower bound regardless of which signal
# it's fit to. The shortest D1 here (0.1s) is still a bit long relative to
# the true T1_fast (~0.37s from the original, more finely-spaced total-
# intensity characterization in relaxation_T1_onepulse_series.py). Trust
# the T1_fast FRACTION, not its time constant, from this script.
# ============================================================

# === CONFIGURATION — only section to edit ===
# Same validated series as relaxation_T1_onepulse_series.py (exp224-236, NS=32, onepulse.al).
DATASETS = {
    # D1 (s) : path to the Bruker experiment folder
    0.1: r"D:\Postdoc\Datas\LLZO-400-aug26\224",
    0.3: r"D:\Postdoc\Datas\LLZO-400-aug26\225",
    0.5: r"D:\Postdoc\Datas\LLZO-400-aug26\226",
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
REFERENCE_D1 = 150  # longest D1 in DATASETS -> best S/N, fully relaxed, undistorted

LB = 10
PH0_MANUAL = -103.394
PH1 = -49.524
AUTO_PH0 = True
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
OUTPUT_NAME = "T1_components_fit"

# Same convention as pipeline_1d.py's PEAKS / relaxation_T2_components.py's
# REFERENCE_PEAKS. Copy your latest finalized numbers here before running.
REFERENCE_PEAKS = {
    "ppm_min": -23, "ppm_max": 27,
    "p0": [
        [3.5e7, 0.6, 5.8,  0.99],  # narrow component
        [1.0e7, 0.6, 15,   0.5 ],  # broad component
    ],
    "eta_fixed": [None, 0.0],
    "width_bounds": [(0, 8), (8, 100)],
    "position_bounds": [(-4, 5), (-4, 5)],
}
COMPONENT_NAMES = ["narrow (fine)", "broad (large)"]
COMPONENT_COLORS = ["darkorange", "purple"]
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, read_phase_from_procs=False, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.
    Built on the shared core.process_row / core.find_best_ph0 -- same pattern as
    pipeline_1d.py / relaxation_T2_components.py.
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


# NOTE: pseudo_voigt / sum_pseudo_voigt / fit_group are imported from
# library_nmr.fitting above -- previously duplicated here, identical logic.

def fit_amplitudes_fixed_shape(delta, signal, ppm_min, ppm_max, components):
    """Solves for the component amplitudes ONLY, with position/width/eta
    frozen to the reference values -> NNLS (see relaxation_T2_components.py)."""
    mask = (delta >= ppm_min) & (delta <= ppm_max)
    x = delta[mask]
    y = signal[mask]
    basis = np.column_stack([
        pseudo_voigt(x, 1.0, c["position"], c["width"], c["eta"]) for c in components
    ])
    amps, _ = nnls(basis, y)
    return amps


def biexp_recovery(t, M0, f, T1a, T1b):
    """M0 * (1 - f*exp(-t/T1a) - (1-f)*exp(-t/T1b)) -- same form as
    relaxation_T1_onepulse_series.py."""
    return M0 * (1 - f * np.exp(-t / T1a) - (1 - f) * np.exp(-t / T1b))


if __name__ == "__main__":
    # === STEP 1: reference shape fit (best S/N = longest D1 = fully relaxed) ===
    ref_path = DATASETS[REFERENCE_D1]
    print(f"\n=== Reference shape fit (D1={REFERENCE_D1}s, {ref_path}) ===")
    delta_ref, spectrum_ref, _, ph0_ref = process_1d_spectrum(
        ref_path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM
    )
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
    # NOTE: unlike relaxation_T2_components.py, phase is NOT frozen from a
    # separate reference here -- AUTO_PH0 runs independently per D1 point,
    # same as relaxation_T1_onepulse_series.py. If amplitudes look
    # inconsistent, check PH0 per point first (see relaxation_T2_echo_series.py's
    # note on why PH0 freezing mattered there).
    D1_list = []
    amp_series = [[] for _ in range(n_comp)]

    for d1, path in sorted(DATASETS.items()):
        print(f"\nD1 = {d1} s  ({path})")
        try:
            delta, spectrum, _, ph0 = process_1d_spectrum(
                path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
                auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
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
        D1_list.append(d1)
        for i in range(n_comp):
            amp_series[i].append(amps[i])

    D1 = np.array(D1_list)

    # === STEP 3: independent biexponential T1 fit per component ===
    fit_params = []  # {"popt": (...), "perr": (...)} per component, or None
    for i in range(n_comp):
        name = COMPONENT_NAMES[i]
        y = np.array(amp_series[i])
        usable = y > 0
        if usable.sum() < 5:
            # biexp_recovery has 4 free parameters (M0, f, T1a, T1b) — need
            # strictly more data points than parameters for curve_fit to be
            # well-posed (matches the n_needed=5 guard in relaxation_T2_components.py).
            print(f"\n{name}: only {usable.sum()} usable (>0) points — cannot fit, skipping.")
            fit_params.append(None)
            continue
        t_in, y_in = D1[usable], y[usable]
        p0 = [1.1 * y_in.max(), 0.05, 0.4, t_in[t_in > t_in.max() / 4].mean()]
        bounds = ([0.5 * y_in.max(), 0, 0.01, 5], [5 * y_in.max(), 0.3, 5, 200])
        try:
            # CAUTION (fixed 18/08): sigma=y_in was missing here — same regression
            # as relaxation_T1_onepulse_series.py. Without it the fit is dominated
            # by the large-amplitude long-D1 points and effectively ignores the
            # short-D1 points where T1_fast lives — this was the real cause of a
            # ~15-18% systematic residual trend at short D1, previously fixed once
            # and documented in fit_T1_components.py (Library_nmr working scripts).
            # Do not remove sigma=y_in without re-checking residuals at short D1.
            popt, pcov = curve_fit(biexp_recovery, t_in, y_in, p0=p0, sigma=y_in, maxfev=50000, bounds=bounds)
            perr = np.sqrt(np.diag(pcov))
            fit_params.append({"popt": tuple(popt), "perr": tuple(perr)})
            M0, f, T1fast, T1slow = popt
            _, fe, T1faste, T1slowe = perr
            print(f"\n{name}: T1_fast = {T1fast:.3f} +/- {T1faste:.3f} s  (fraction {f*100:.1f}%, "
                  f"value poorly constrained by this D1 grid — see module docstring)  "
                  f"T1_slow = {T1slow:.2f} +/- {T1slowe:.2f} s  (fraction {(1-f)*100:.1f}%)  "
                  f"(fit on {usable.sum()}/{len(y)} points)")
            rel_resid = 100 * (y_in - biexp_recovery(t_in, *popt)) / y_in
            print(f"  relative residuals (%): {np.round(rel_resid, 2)}")
            if np.any(np.abs(rel_resid) > 20):
                print("  WARNING: some points have >20% residual — check phasing/S-N on those spectra.")
        except RuntimeError as e:
            print(f"\n{name}: fit failed ({e})")
            fit_params.append(None)

    # === export ===
    df = pd.DataFrame({"D1_s": D1})
    for i, name in enumerate(COMPONENT_NAMES):
        df[f"amplitude_{name.split()[0]}"] = amp_series[i]
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    # === plot ===
    fig, ax = plt.subplots(figsize=(8, 5.5))
    t_fit = np.logspace(np.log10(D1.min() / 2), np.log10(D1.max() * 1.3), 400)

    textlines = []
    for i, name in enumerate(COMPONENT_NAMES):
        y = np.array(amp_series[i])
        usable = y > 0
        ax.scatter(D1[usable], y[usable], color=COMPONENT_COLORS[i], s=55, zorder=3,
                   edgecolor="black", linewidth=0.4, label=f"{name} — data")
        if fit_params[i] is not None:
            popt, perr = fit_params[i]["popt"], fit_params[i]["perr"]
            M0, f, T1fast, T1slow = popt
            _, _, _, T1slowe = perr
            ax.plot(t_fit, biexp_recovery(t_fit, *popt), color=COMPONENT_COLORS[i], lw=1.8, zorder=2,
                    label=f"{name} — biexponential fit")
            textlines.append(f"{name}: T1 slow = {T1slow:.1f}±{T1slowe:.1f}s ({(1-f)*100:.1f}%)")

    ax.set_xscale("log")
    ax.set_xlabel("Recovery delay D1 (s)")
    ax.set_ylabel("Component amplitude (a.u.)")
    ax.set_title(r"$^7$Li T$_1$ per component (fixed shape, NNLS amplitudes)")
    if textlines:
        textstr = "\n".join(textlines) + "\n(T1 fast fraction ~4% in both, value poorly\nconstrained by this D1 grid — not shown)"
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()
