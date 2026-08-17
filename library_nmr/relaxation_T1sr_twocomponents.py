import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0, parse_bruker_delay
from library_nmr.fitting import pseudo_voigt, sum_pseudo_voigt

# ============================================================
# PSEUDO-2D T1 SATURATION-RECOVERY — TWO-COMPONENT VERSION — nmr_library
# Usage: edit the CONFIGURATION block below, then run.
#
# Purpose: same experiment as pseudo2d_T1_saturation_recovery.py, but tracks
# TWO overlapping lineshape components (e.g. a narrow + a broad Li site)
# separately across the relaxation series, instead of a single peak-height
# extraction. Each component gets its own T1.
#
# METHOD: the two pseudo-Voigt SHAPES (position, width, eta) are determined
# ONCE via a full nonlinear fit on the fully-relaxed (longest tau) row — the
# best-conditioned row, best S/N. For every other row, only the two
# AMPLITUDES are solved via ordinary linear least squares against those
# fixed shapes. This avoids re-fitting 8 nonlinear parameters on every
# noisy row.
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\222"  # Bruker folder containing the pseudo-2D saturation-recovery experiment
LB = 50  # line broadening in Hz
PH0_MANUAL = 0.0
PH1 = 0.0
AUTO_PH0 = True  # automatic PH0 search, performed on the fully recovered (longest tau) row
READ_PHASE_FROM_PROCS = False

# --- Two-component lineshape (fit once, on the reference row) ---
PPM_MIN, PPM_MAX = -23.7, 26.3  # fit window — keep consistent with the main 1D pipeline
COMPONENTS_P0 = [
    [3.5e7, 0.6, 5.8,  0.99],  # narrow component initial guess [A, nu0, FWHM, eta]
    [1.0e7, 0.6, 15,   0.5 ],  # broad component initial guess
]
ETA_FIXED = [None, 0.0]              # broad component fixed to pure Gaussian, as established on the 1D spectrum
WIDTH_BOUNDS = [(0, 8), (8, 100)]     # prevents role-swapping between components
POSITION_BOUNDS = [(-4, 5), (-4, 5)]  # keeps both components on the real peak

NOISE_REGION_PPM = (-40, -30)
FIT_SATURATION_FACTOR = True  # fit B (saturation factor) independently for each component's T1 curve.
    # Physically, both components experience the SAME saturation comb, so their fitted B values
    # should come out similar — this script reports both so you can check that consistency.
OUTPUT_NAME = "T1_sat_recovery_two_components"
# ================================================


# --- Shared processing functions (same as the other pseudo-2D scripts) ---

def t1_sat_recovery(tau, M0, T1, B):
    return M0 * (1 - B * np.exp(-tau / T1))


# --- Two-component lineshape fitting ---
# NOTE: this fit_group is intentionally a SIMPLER, LOCAL variant, not the one
# in library_nmr.fitting — it returns a flat [A,nu0,FWHM,eta]*n_peaks array
# (no uncertainties/integral/etc.), since it is only used ONCE here to fix
# the lineshape parameters on the reference row before the per-row linear
# amplitude extraction. pseudo_voigt/sum_pseudo_voigt themselves ARE shared
# (imported from library_nmr.fitting above).

def fit_group(delta, spectrum, ppm_min, ppm_max, p0_list, eta_fixed_list=None,
              width_bounds_list=None, position_bounds_list=None):
    """Nonlinear multi-pseudo-Voigt fit (see main pipeline for full docstring).
    Used ONCE here, on the reference (fully-relaxed) row, to fix the lineshapes."""
    mask = (delta >= ppm_min) & (delta <= ppm_max)
    delta_peak = delta[mask]
    signal_peak = spectrum[mask]

    n_peaks = len(p0_list)
    if eta_fixed_list is None:
        eta_fixed_list = [None] * n_peaks
    if width_bounds_list is None:
        width_bounds_list = [(0, 100)] * n_peaks
    if position_bounds_list is None:
        position_bounds_list = [(ppm_min, ppm_max)] * n_peaks

    p0_reduced, lower_bounds_reduced, upper_bounds_reduced = [], [], []
    for p0_i, eta_fixed, (fwhm_min, fwhm_max), (nu0_min, nu0_max) in zip(
        p0_list, eta_fixed_list, width_bounds_list, position_bounds_list
    ):
        A0, nu0_0, FWHM0, eta0 = p0_i
        p0_reduced += [A0, nu0_0, FWHM0]
        lower_bounds_reduced += [0, nu0_min, fwhm_min]
        upper_bounds_reduced += [np.inf, nu0_max, fwhm_max]
        if eta_fixed is None:
            p0_reduced.append(eta0)
            lower_bounds_reduced.append(0)
            upper_bounds_reduced.append(1)

    def reduced_model(x, *reduced_params):
        full_params = []
        idx = 0
        for eta_fixed in eta_fixed_list:
            A, nu0, FWHM = reduced_params[idx:idx+3]
            idx += 3
            if eta_fixed is None:
                eta = reduced_params[idx]
                idx += 1
            else:
                eta = eta_fixed
            full_params += [A, nu0, FWHM, eta]
        return sum_pseudo_voigt(x, *full_params)

    popt_reduced, _ = curve_fit(
        reduced_model, delta_peak, signal_peak, p0=p0_reduced,
        bounds=(lower_bounds_reduced, upper_bounds_reduced), maxfev=20000
    )

    popt, idx = [], 0
    for eta_fixed in eta_fixed_list:
        A, nu0, FWHM = popt_reduced[idx:idx+3]
        idx += 3
        if eta_fixed is None:
            eta = popt_reduced[idx]
            idx += 1
        else:
            eta = eta_fixed
        popt += [A, nu0, FWHM, eta]
    return np.array(popt)  # flat [A1,nu1,FWHM1,eta1, A2,nu2,FWHM2,eta2, ...]


# === PROCESSING ===

# --- Read data ---
dic, data = ng.bruker.read(PATH, read_procs=READ_PHASE_FROM_PROCS)
print(f"Data shape: {data.shape}  (n_rows, n_points)")

grpdly_shift = find_grpdly_shift(dic)
print(f"GRPDLY corrected: shifting each row by {grpdly_shift} points" if grpdly_shift > 0
      else "GRPDLY not found or zero — no correction applied")

SW_h = dic["acqus"]["SW_h"]
SFO1 = dic["acqus"]["SFO1"]
O1 = dic["acqus"]["O1"]
dt = 1 / SW_h

# --- vdlist ---
try:
    with open(f"{PATH}/vdlist", "r") as f:
        tau_values = np.array([parse_bruker_delay(line) for line in f if line.strip()])
    print(f"tau read from vdlist (seconds): {tau_values}")
except FileNotFoundError:
    raise FileNotFoundError(f"No vdlist found in {PATH}.")
if len(tau_values) != data.shape[0]:
    raise ValueError(f"vdlist has {len(tau_values)} values but data has {data.shape[0]} rows.")

idx_longest = int(np.argmax(tau_values))

# --- Phase (reference: fully recovered row) ---
if READ_PHASE_FROM_PROCS:
    try:
        ph0_deg = dic["procs"]["PHC0"]
        ph1_deg = dic["procs"]["PHC1"]
        print(f"Phase read from procs: PH0={ph0_deg:.3f}°, PH1={ph1_deg:.3f}°")
    except Exception:
        print("Could not read phase from procs — falling back to manual/auto")
        ph0_deg, ph1_deg = PH0_MANUAL, PH1
else:
    ph1_deg = PH1
    if AUTO_PH0:
        signal_ref = process_row(data[idx_longest], dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift)
        ph0_deg = find_best_ph0(signal_ref, np.deg2rad(ph1_deg))
        print(f"Optimal PH0 found (auto, row {idx_longest+1}, longest tau): {ph0_deg:.3f}°")
    else:
        ph0_deg = PH0_MANUAL

ph0_rad = np.deg2rad(ph0_deg)
ph1_rad = np.deg2rad(ph1_deg)

# --- Process all rows ---
n_points = data.shape[1]
f = np.fft.fftshift(np.fft.fftfreq(n_points, dt))
delta = (O1 - f) / SFO1

spectra_real = np.array([
    process_row(data[i], dt, LB, ph0_rad, ph1_rad, grpdly_shift).real
    for i in range(data.shape[0])
])

# --- Step 1: fix the lineshapes on the reference (fully relaxed) row ---
popt_ref = fit_group(delta, spectra_real[idx_longest], PPM_MIN, PPM_MAX, COMPONENTS_P0,
                      eta_fixed_list=ETA_FIXED, width_bounds_list=WIDTH_BOUNDS,
                      position_bounds_list=POSITION_BOUNDS)
n_components = len(COMPONENTS_P0)
print(f"\nReference lineshapes (fixed from row {idx_longest+1}, longest tau):")
shapes = []  # list of (nu0, FWHM, eta) per component
for c in range(n_components):
    A_ref, nu0, FWHM, eta = popt_ref[c*4:(c+1)*4]
    shapes.append((nu0, FWHM, eta))
    print(f"  Component {c+1}: position={nu0:.3f} ppm, width={FWHM:.3f} ppm "
          f"({FWHM*SFO1:.1f} Hz), eta={eta:.3f}")

# --- Step 2: build the fixed-shape design matrix (unit amplitude per component) ---
window_mask = (delta >= PPM_MIN) & (delta <= PPM_MAX)
delta_window = delta[window_mask]
sort_idx = np.argsort(delta_window)

unit_shapes = np.array([pseudo_voigt(delta_window, 1.0, nu0, FWHM, eta)
                         for (nu0, FWHM, eta) in shapes])  # shape (n_components, n_points_window)
unit_integrals = np.array([
    np.trapezoid(unit_shapes[c][sort_idx], delta_window[sort_idx]) for c in range(n_components)
])
X = unit_shapes.T  # (n_points_window, n_components) design matrix for lstsq

# --- Step 3: per-row amplitude extraction via linear least squares ---
amplitudes = np.zeros((data.shape[0], n_components))
for i in range(data.shape[0]):
    y = spectra_real[i][window_mask]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    amplitudes[i] = coeffs

integrals = amplitudes * unit_integrals[np.newaxis, :]  # area per component, per row

# --- S/N diagnostic (whole-row noise, same for all components of a given row) ---
noise_mask = (delta >= min(NOISE_REGION_PPM)) & (delta <= max(NOISE_REGION_PPM))
noise_std_per_row = np.array([np.std(row[noise_mask]) for row in spectra_real])

print()
for i in range(data.shape[0]):
    comp_str = ", ".join(f"comp{c+1}={integrals[i,c]:.3e}" for c in range(n_components))
    print(f"Row {i+1}: tau = {tau_values[i]*1000:.3f} ms, {comp_str}, noise std = {noise_std_per_row[i]:.3e}")

# --- Step 4: T1 fit per component ---
results_per_component = []
for c in range(n_components):
    y = integrals[:, c]
    M0_guess = y[idx_longest]
    if M0_guess <= 0:
        # lstsq over overlapping/near-collinear pseudo-Voigt basis shapes can
        # return a negative amplitude for a component even when the true
        # signal is positive. curve_fit's bounds below require M0 >= 0, so a
        # negative initial guess makes the starting point infeasible and
        # raises ValueError (not RuntimeError) — crashing the whole script
        # instead of just failing this component. Clamp to a small positive
        # fallback instead. (fixed 18/08)
        M0_guess = max(np.abs(y).max(), 1e-6)
        print(f"\nComponent {c+1}: NOTE — raw M0 guess from lstsq was <= 0 (near-collinear basis "
              f"shapes); using |y|.max() as a fallback initial guess instead.")
    target = M0_guess * (1 - 1 / np.e)
    idx_above = np.where(y >= target)[0]
    if len(idx_above) > 0 and idx_above[0] > 0:
        i1 = idx_above[0]
        t_a, t_b = tau_values[i1 - 1], tau_values[i1]
        y_a, y_b = y[i1 - 1], y[i1]
        T1_guess = t_a + (target - y_a) * (t_b - t_a) / (y_b - y_a)
        print(f"\nComponent {c+1}: (1-1/e) crossing between rows {i1}/{i1+1}, "
              f"T1 guess ~ {T1_guess*1000:.3f} ms")
    elif len(idx_above) > 0 and idx_above[0] == 0:
        # Already at/above (1-1/e) of the plateau on the very first tau point —
        # NOT "never reaches". True T1 may be shorter than this grid resolves.
        T1_guess = tau_values[0]
        print(f"\nComponent {c+1}: (1-1/e) of plateau already reached at the shortest tau tested "
              f"(~{T1_guess*1000:.3f} ms) — using it as an upper-bound initial guess.")
    else:
        T1_guess = np.median(tau_values)
        print(f"\nComponent {c+1}: WARNING — recovery never clearly reaches (1-1/e) of plateau "
              f"within the tested tau range. This component's true T1 may lie outside the range "
              f"tested, or the vdlist may need more points at short tau.")

    try:
        if FIT_SATURATION_FACTOR:
            popt, pcov = curve_fit(t1_sat_recovery, tau_values, y, p0=[M0_guess, T1_guess, 1.0],
                                    bounds=([0, 0, 0], [np.inf, np.inf, 2]), maxfev=20000)
            M0, T1, B = popt
            err_M0, err_T1, err_B = np.sqrt(np.diag(pcov))
        else:
            def t1_sat_recovery_fixed_B(tau, M0, T1):
                return t1_sat_recovery(tau, M0, T1, 1.0)
            popt, pcov = curve_fit(t1_sat_recovery_fixed_B, tau_values, y, p0=[M0_guess, T1_guess],
                                    bounds=([0, 0], [np.inf, np.inf]), maxfev=20000)
            M0, T1 = popt
            B = 1.0
            err_M0, err_T1 = np.sqrt(np.diag(pcov))
            err_B = 0.0
        print(f"Component {c+1} fit: M0={M0:.3e}, T1={T1*1000:.3f} +/- {err_T1*1000:.3f} ms, "
              f"B={B:.3f} +/- {err_B:.3f}")
        if err_T1 / T1 > 1:
            print(f"  WARNING: relative T1 uncertainty {err_T1/T1*100:.0f}% — unreliable fit "
                  f"for this component (check tau range / (1-1/e) crossing above).")
        results_per_component.append({"M0": M0, "T1": T1, "err_T1": err_T1, "B": B, "err_B": err_B})
    except (RuntimeError, ValueError) as e:
        # ValueError is included because an infeasible initial guess (e.g. any
        # x0 outside `bounds`) raises ValueError rather than RuntimeError —
        # without catching it here, one bad component crashes the entire
        # script instead of just being reported as failed. (fixed 18/08)
        print(f"Component {c+1}: T1 fit failed to converge ({type(e).__name__}: {e}).")
        results_per_component.append({"M0": np.nan, "T1": np.nan, "err_T1": np.nan, "B": np.nan, "err_B": np.nan})

# --- Consistency check: both components should show a similar B (same physical saturation comb) ---
if FIT_SATURATION_FACTOR and n_components >= 2 and all(not np.isnan(r["B"]) for r in results_per_component):
    B_values = [r["B"] for r in results_per_component]
    print(f"\nConsistency check — saturation factor B per component: "
          f"{', '.join(f'{b:.3f}' for b in B_values)}")
    if max(B_values) - min(B_values) > 0.3:
        print("  NOTE: components disagree notably on B — check that the reference-row lineshape "
              "fit and window are both sound before trusting either T1 individually.")

# --- Figure: both components on the same plot ---
colors = ["blue", "red", "green", "purple"]
fig, ax = plt.subplots(figsize=(9, 5.5))
tau_fit = np.logspace(np.log10(tau_values[tau_values > 0].min()), np.log10(tau_values.max()), 200)
for c in range(n_components):
    r = results_per_component[c]
    ax.scatter(tau_values * 1000, integrals[:, c], color=colors[c % len(colors)],
               label=f"component {c+1} data")
    if not np.isnan(r["T1"]):
        ax.plot(tau_fit * 1000, t1_sat_recovery(tau_fit, r["M0"], r["T1"], r["B"]),
                color=colors[c % len(colors)], linewidth=1.5,
                label=f"component {c+1} fit: T1={r['T1']*1000:.2f} ms, B={r['B']:.2f}")
ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Recovery time tau (ms)")
ax.set_ylabel("Integrated area (a.u.)")
ax.set_title("T1 relaxation — saturation recovery, two components")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()

# --- CSV export ---
df = pd.DataFrame({"tau_ms": tau_values * 1000})
for c in range(n_components):
    df[f"integral_comp{c+1}"] = integrals[:, c]
    df[f"amplitude_comp{c+1}"] = amplitudes[:, c]
df["noise_std"] = noise_std_per_row
for c in range(n_components):
    r = results_per_component[c]
    df[f"T1_comp{c+1}_ms"] = r["T1"] * 1000
    df[f"err_T1_comp{c+1}_ms"] = r["err_T1"] * 1000
    df[f"B_comp{c+1}"] = r["B"]
df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
print(f"\nResults exported to {OUTPUT_NAME}.csv")