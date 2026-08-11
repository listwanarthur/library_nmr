import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit, nnls

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.fitting import pseudo_voigt, sum_pseudo_voigt, fit_group

# ============================================================
# T2 PER-COMPONENT FIT — nmr_library (echosolide_v2, discrete L0 series)
# Usage: edit the CONFIGURATION block below, then run
#
# Why NOT a free two-component pseudo-Voigt fit on every spectrum:
# that is exactly what pipeline_1d.py's fit_group does, and it is the
# decomposition already flagged as unstable run-to-run (shape and eta
# drifting, amplitudes sometimes going negative) — worse here because
# S/N drops fast with increasing echo delay, so a fully free fit has
# even less to constrain it on the later spectra.
#
# Strategy used here instead:
#   1) Fit ONE reference spectrum (best S/N, shortest delay) with the
#      normal free fit_group (from pipeline_1d.py) to pin down
#      position / width / eta for each component.
#   2) FREEZE those shape parameters. For every other spectrum in the
#      T2 series, only the two amplitudes are then unknown — and
#      because a pseudo-Voigt is LINEAR in amplitude for fixed
#      position/width/eta, that reduces to solving a small linear
#      least-squares problem per spectrum. Non-negative least squares
#      (NNLS) is used so an amplitude can go to exactly 0 instead of
#      the unphysical negative values seen with the free fit.
#   3) Each component's amplitude vs echo delay is then its own T2
#      decay curve, fit independently (mono-exponential; the whole
#      point of separating the components is that each is expected to
#      have ONE relaxation time, unlike the mixed total-intensity
#      decay in fit_T2_echo.py).
# ============================================================

# === CONFIGURATION — only section to edit ===
TAU_ROTOR_US = 80.0  # rotor period (µs), MAS 12.5 kHz -> 80 µs

DATASETS = {
    # L0 (integer number of rotor periods) : path to the Bruker experiment folder
    1:   r"D:\Postdoc\Datas\LLZO-400-aug26\237",
    2:   r"D:\Postdoc\Datas\LLZO-400-aug26\238",
    4:   r"D:\Postdoc\Datas\LLZO-400-aug26\239",
    8:   r"D:\Postdoc\Datas\LLZO-400-aug26\240",
    16:  r"D:\Postdoc\Datas\LLZO-400-aug26\241",
    32:  r"D:\Postdoc\Datas\LLZO-400-aug26\242",
    40:  r"D:\Postdoc\Datas\LLZO-400-aug26\...",
    50:  r"D:\Postdoc\Datas\LLZO-400-aug26\...",
    65:  r"D:\Postdoc\Datas\LLZO-400-aug26\...",
    80:  r"D:\Postdoc\Datas\LLZO-400-aug26\...",
    90:  r"D:\Postdoc\Datas\LLZO-400-aug26\...",
    100: r"D:\Postdoc\Datas\LLZO-400-aug26\...",
}
REFERENCE_L0 = 1  # which DATASETS entry to use for the one-time free shape fit (best S/N -> shortest delay)

LB = 10
PH0_MANUAL = -103.394
PH1 = -49.524
AUTO_PH0 = True
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
OUTPUT_NAME = "T2_components_fit"

# Same convention as pipeline_1d.py's PEAKS — used ONCE, on the reference
# spectrum only, to determine position/width/eta for each component.
# Copy your latest finalized numbers here before running.
REFERENCE_PEAKS = {
    "ppm_min": -23, "ppm_max": 27,
    "p0": [
        [3.5e7, 0.6, 5.8,  0.99],  # narrow component (fine, ~78% in earlier runs)
        [1.0e7, 0.6, 15,   0.5 ],  # broad component (large, eta=0 in earlier runs)
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

    return delta, spectrum, dic


# NOTE: pseudo_voigt / sum_pseudo_voigt / fit_group are imported from
# library_nmr.fitting above (identical logic -- this script previously
# defined its own copies). fit_group's returned dicts carry more fields
# (err_position, integral, popt, ...) than this script actually uses
# (only position/width/eta), so the shared version is a drop-in
# replacement with no behavior change here.


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


# === STEP 1: reference shape fit ===
ref_path = DATASETS[REFERENCE_L0]
print(f"\n=== Reference shape fit (L0={REFERENCE_L0}, {ref_path}) ===")
delta_ref, spectrum_ref, _ = process_1d_spectrum(
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
    print(f"  {name}: position={r['position']:.3f} ppm, width={r['width']:.3f} ppm, eta={r['eta']:.3f}  (FROZEN for the rest of the fit)")

components = [{"position": r["position"], "width": r["width"], "eta": r["eta"]} for r in ref_results]
n_comp = len(components)

# === STEP 2: fixed-shape amplitude fit on every spectrum of the series ===
L0_list, tau_list = [], []
amp_series = [[] for _ in range(n_comp)]

for l0, path in sorted(DATASETS.items()):
    tau_us = l0 * TAU_ROTOR_US
    print(f"\nL0 = {l0}  (tau_echo = {tau_us:.0f} µs)  ({path})")
    try:
        delta, spectrum, _ = process_1d_spectrum(
            path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
            auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
            reference_shift_ppm=REFERENCE_SHIFT_PPM
        )
    except OSError as e:
        print(f"  SKIPPED — could not read dataset: {e}")
        continue
    amps = fit_amplitudes_fixed_shape(delta, spectrum.real, REFERENCE_PEAKS["ppm_min"], REFERENCE_PEAKS["ppm_max"], components)
    for name, a in zip(COMPONENT_NAMES, amps):
        print(f"  {name}: amplitude = {a:.4e}" + ("  (zeroed by NNLS — below detection for this component)" if a <= 0 else ""))
    L0_list.append(l0)
    tau_list.append(tau_us)
    for i in range(n_comp):
        amp_series[i].append(amps[i])

tau = np.array(tau_list)

# === STEP 3: independent mono-exponential T2 fit per component ===
fit_params = []  # (A0, T2, A0_err, T2_err) per component
for i in range(n_comp):
    y = np.array(amp_series[i])
    usable = y > 0  # NNLS can zero out a point -> drop it rather than fit log(0)
    if usable.sum() < 3:
        print(f"\n{COMPONENT_NAMES[i]}: only {usable.sum()} usable (>0) points — cannot fit, skipping.")
        fit_params.append(None)
        continue
    t_fit_in, y_fit_in = tau[usable], y[usable]
    p0 = [y_fit_in.max(), t_fit_in[len(t_fit_in)//2]]
    try:
        popt, pcov = curve_fit(monoexp_decay, t_fit_in, y_fit_in, p0=p0, sigma=y_fit_in,
                                bounds=([0, 1], [10 * y_fit_in.max(), 200000]), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        fit_params.append((popt[0], popt[1], perr[0], perr[1]))
        rel_resid = 100 * (y_fit_in - monoexp_decay(t_fit_in, *popt)) / y_fit_in
        print(f"\n{COMPONENT_NAMES[i]}: T2 = {popt[1]:.1f} +/- {perr[1]:.1f} us  "
              f"(fit on {usable.sum()}/{len(y)} points)")
        print(f"  relative residuals (%): {np.round(rel_resid, 2)}")
        if np.any(np.abs(rel_resid) > 20):
            print("  WARNING: some points have >20% residual — check phasing/S-N on those spectra.")
    except RuntimeError as e:
        print(f"\n{COMPONENT_NAMES[i]}: fit failed ({e})")
        fit_params.append(None)

# === export ===
df = pd.DataFrame({"L0": L0_list, "tau_echo_us": tau})
for i, name in enumerate(COMPONENT_NAMES):
    df[f"amplitude_{name.split()[0]}"] = amp_series[i]
df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
print(f"\nResults exported to {OUTPUT_NAME}.csv")

# === plot ===
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

textlines = []
for i, name in enumerate(COMPONENT_NAMES):
    y = np.array(amp_series[i])
    usable = y > 0
    ax.scatter(tau[usable], y[usable], color=COMPONENT_COLORS[i], s=55, zorder=3, label=f"{name} — data")
    if fit_params[i] is not None:
        A0, T2, A0e, T2e = fit_params[i]
        t_fit = np.logspace(np.log10(tau.min() / 1.5), np.log10(tau.max() * 1.3), 400)
        ax.plot(t_fit, monoexp_decay(t_fit, A0, T2), color=COMPONENT_COLORS[i], lw=1.5, zorder=2,
                label=f"{name} — fit")
        textlines.append(f"T2 {name} = {T2:.0f} +/- {T2e:.0f} us")

# below-detection points (never fitted)
nd_tau, nd_y = [], []
for l0, path in sorted(BELOW_DETECTION.items()):
    tau_us = l0 * TAU_ROTOR_US
    try:
        delta, spectrum, _ = process_1d_spectrum(
            path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
            auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
            reference_shift_ppm=REFERENCE_SHIFT_PPM
        )
        amps_nd = fit_amplitudes_fixed_shape(delta, spectrum.real, REFERENCE_PEAKS["ppm_min"], REFERENCE_PEAKS["ppm_max"], components)
        nd_tau.append(tau_us)
        nd_y.append(sum(amps_nd))
    except OSError:
        pass
if nd_tau:
    ax.scatter(nd_tau, nd_y, marker="x", color="gray", s=70, zorder=3, label="below detection (ND)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Component amplitude (a.u.)")
ax.set_title(r"$^7$Li T$_2$ per component (fixed shape, NNLS amplitudes)")
if textlines:
    ax.text(0.97, 0.05, "\n".join(textlines), transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, name in enumerate(COMPONENT_NAMES):
    if fit_params[i] is None:
        continue
    y = np.array(amp_series[i])
    usable = y > 0
    A0, T2, A0e, T2e = fit_params[i]
    resid_pct = 100 * (y[usable] - monoexp_decay(tau[usable], A0, T2)) / y[usable]
    ax2.scatter(tau[usable], resid_pct, color=COMPONENT_COLORS[i], s=40)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xscale("log")
ax2.set_xlabel("Echo delay tau (us) = L0 x tau_rotor")
ax2.set_ylabel("residual (%)")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()