import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0, parse_bruker_delay

# ============================================================
# PSEUDO-2D T1 INVERSION-RECOVERY EXTRACTION — nmr_library
# Usage: edit the CONFIGURATION block below, then run.
#
# Purpose: read a pseudo-2D inversion-recovery experiment (180-tau-90-acquire,
# varying recovery time tau from a vdlist), extract signed peak intensity per
# row, and fit the T1 recovery curve.
#
# IMPORTANT DIFFERENCE vs the T2 (spin-echo) script: intensity here changes
# SIGN across the series (negative right after inversion, positive once fully
# relaxed). This affects peak localization, extraction, and the fit model —
# do not reuse T2-script logic blindly, several steps below are intentionally
# different.
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\210"  # Bruker folder containing the pseudo-2D inversion-recovery experiment
LB = 50  # line broadening in Hz
PH0_MANUAL = 0.0  # PHC0 in degrees, used if AUTO_PH0=False and READ_PHASE_FROM_PROCS=False
PH1 = 0.0  # PHC1 in degrees
AUTO_PH0 = True  # automatic PH0 search, performed on the FULLY RECOVERED row (longest tau) —
                  # that row has an ordinary (non-inverted) absorptive lineshape, the most
                  # reliable reference for phasing.
READ_PHASE_FROM_PROCS = False
PPM_MIN = -10  # search window to locate the peak
PPM_MAX = 10
EXTRACTION_HALFWIDTH_PPM = 0.2  # intensity extracted as the mean over
    # [peak_position - halfwidth, peak_position + halfwidth]. Set to 0 for the single nearest point.
NOISE_REGION_PPM = (-40, -30)  # signal-free region, for S/N diagnostics per row
FIT_INVERSION_FACTOR = True  # True: fit the inversion factor A freely (recommended — real
    # inversion pulses are rarely perfect, A=2 only holds for an ideal 180°). False: fix A=2.0
    # (ideal inversion) and only fit M0, T1.
OUTPUT_NAME = "T1_inversion_recovery"
# ================================================


def t1_recovery(tau, M0, T1, A):
    """Inversion-recovery model: M(tau) = M0 * (1 - A * exp(-tau/T1)).
    A=2.0 for an ideal 180° inversion pulse; A<2 for an imperfect one.
    """
    return M0 * (1 - A * np.exp(-tau / T1))


# === PROCESSING ===

# --- Read the pseudo-2D data ---
dic, data = ng.bruker.read(PATH, read_procs=READ_PHASE_FROM_PROCS)
print(f"Data shape: {data.shape}  (n_rows, n_points)")

grpdly_shift = find_grpdly_shift(dic)
if grpdly_shift > 0:
    print(f"GRPDLY corrected: shifting each row by {grpdly_shift} points")
else:
    print("GRPDLY not found or zero — no correction applied")

SW_h = dic["acqus"]["SW_h"]
SFO1 = dic["acqus"]["SFO1"]
O1 = dic["acqus"]["O1"]
dt = 1 / SW_h

# --- Read the recovery time list (vdlist) ---
try:
    with open(f"{PATH}/vdlist", "r") as f:
        tau_values = np.array([parse_bruker_delay(line) for line in f if line.strip()])
    print(f"tau (recovery times) read from vdlist (converted to seconds): {tau_values}")
except FileNotFoundError:
    raise FileNotFoundError(f"No vdlist found in {PATH}.")
if len(tau_values) != data.shape[0]:
    raise ValueError(
        f"vdlist has {len(tau_values)} values but data has {data.shape[0]} rows — "
        f"check TD1 in posacqus matches the vdlist length, and that the experiment completed."
    )

# --- Determine phase, using the FULLY RECOVERED (longest tau) row as reference ---
# (An inverted/near-null row has an ambiguous or sign-flipped lineshape — not a
# reliable reference for automatic phasing.)
idx_longest = int(np.argmax(tau_values))
if READ_PHASE_FROM_PROCS:
    try:
        ph0_deg = dic["procs"]["PHC0"]
        ph1_deg = dic["procs"]["PHC1"]
        print(f"Phase read from procs: PH0={ph0_deg:.3f}°, PH1={ph1_deg:.3f}°")
    except Exception:
        print("Could not read phase from procs — falling back to manual/auto value")
        ph0_deg, ph1_deg = PH0_MANUAL, PH1
else:
    ph1_deg = PH1
    if AUTO_PH0:
        signal_ref = process_row(data[idx_longest], dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift)
        ph0_deg = find_best_ph0(signal_ref, np.deg2rad(ph1_deg))
        print(f"Optimal PH0 found (auto, on row {idx_longest+1}, longest tau): {ph0_deg:.3f}°")
    else:
        ph0_deg = PH0_MANUAL

ph0_rad = np.deg2rad(ph0_deg)
ph1_rad = np.deg2rad(ph1_deg)

# --- Process all rows with the SAME phase ---
n_points = data.shape[1]
f = np.fft.fftshift(np.fft.fftfreq(n_points, dt))
delta = (O1 - f) / SFO1

spectra = np.array([
    process_row(data[i], dt, LB, ph0_rad, ph1_rad, grpdly_shift)
    for i in range(data.shape[0])
])
spectra_real = spectra.real  # MUST be the real (signed) part — magnitude would destroy
                              # the sign information essential to inversion recovery.

# --- Locate the peak: use whichever row has the largest ABSOLUTE signal in the
#     window, since the largest-magnitude row could be the most inverted (short
#     tau) or the most recovered (long tau) depending on the experiment. ---
window_mask = (delta >= PPM_MIN) & (delta <= PPM_MAX)
abs_max_per_row = np.array([np.max(np.abs(row[window_mask])) for row in spectra_real])
idx_peak_row = int(np.argmax(abs_max_per_row))
idx_in_window = np.argmax(np.abs(spectra_real[idx_peak_row][window_mask]))
peak_position_ppm = delta[window_mask][idx_in_window]
print(f"Peak located at {peak_position_ppm:.3f} ppm (from row {idx_peak_row+1}, strongest |signal|)")

if EXTRACTION_HALFWIDTH_PPM > 0:
    extraction_mask = (delta >= peak_position_ppm - EXTRACTION_HALFWIDTH_PPM) & \
                       (delta <= peak_position_ppm + EXTRACTION_HALFWIDTH_PPM)
else:
    extraction_mask = np.zeros_like(delta, dtype=bool)
    extraction_mask[np.argmin(np.abs(delta - peak_position_ppm))] = True

# SIGNED mean — do not take abs() here, the sign carries the physics.
intensities = np.array([np.mean(row[extraction_mask]) for row in spectra_real])

# --- Signal-to-noise diagnostic ---
noise_mask = (delta >= min(NOISE_REGION_PPM)) & (delta <= max(NOISE_REGION_PPM))
noise_std_per_row = np.array([np.std(row[noise_mask]) for row in spectra_real])
snr_per_row = np.abs(intensities) / noise_std_per_row

for i in range(data.shape[0]):
    print(f"Row {i+1}: tau = {tau_values[i]*1000:.3f} ms, intensity = {intensities[i]:.3e}, "
          f"noise std = {noise_std_per_row[i]:.3e}, S/N = {snr_per_row[i]:.1f}")

# --- Estimate T1 from the null point (zero crossing) as an initial guess, and as
#     an independent sanity check even before the full fit. For an ideal
#     inversion recovery, tau_null = T1 * ln(2). ---
sign_changes = np.where(np.diff(np.sign(intensities)) != 0)[0]
if len(sign_changes) > 0:
    i0 = sign_changes[0]
    t_a, t_b = tau_values[i0], tau_values[i0 + 1]
    y_a, y_b = intensities[i0], intensities[i0 + 1]
    tau_null = t_a - y_a * (t_b - t_a) / (y_b - y_a)
    T1_guess = tau_null / np.log(2)
    print(f"\nNull point (zero crossing) found between rows {i0+1} and {i0+2}: "
          f"tau_null ~ {tau_null*1000:.3f} ms -> T1 guess ~ {T1_guess*1000:.3f} ms")
else:
    T1_guess = np.median(tau_values)
    print(f"\nWARNING: no sign change detected across the series — the null point was not "
          f"bracketed by this vdlist. The T1 fit below may be poorly constrained; consider "
          f"extending the tau range (both shorter and/or longer) if the fit looks unreliable.")

M0_guess = intensities[idx_longest]

# --- T1 fit ---
if FIT_INVERSION_FACTOR:
    popt, pcov = curve_fit(
        t1_recovery, tau_values, intensities,
        p0=[M0_guess, T1_guess, 2.0],
        bounds=([-np.inf, 0, 0], [np.inf, np.inf, 4]),
        maxfev=20000
    )
    M0, T1, A = popt
    err_M0, err_T1, err_A = np.sqrt(np.diag(pcov))
    print(f"\nM0 = {M0:.3e} +/- {err_M0:.3e}")
    print(f"T1 = {T1*1000:.3f} +/- {err_T1*1000:.3f} ms")
    print(f"A  = {A:.3f} +/- {err_A:.3f}  (2.0 = ideal 180° inversion pulse)")
else:
    def t1_recovery_fixed_A(tau, M0, T1):
        return t1_recovery(tau, M0, T1, 2.0)
    popt, pcov = curve_fit(
        t1_recovery_fixed_A, tau_values, intensities,
        p0=[M0_guess, T1_guess],
        bounds=([-np.inf, 0], [np.inf, np.inf]),
        maxfev=20000
    )
    M0, T1 = popt
    A = 2.0
    err_M0, err_T1 = np.sqrt(np.diag(pcov))
    print(f"\nM0 = {M0:.3e} +/- {err_M0:.3e}")
    print(f"T1 = {T1*1000:.3f} +/- {err_T1*1000:.3f} ms  (A fixed at 2.0)")

if err_T1 / T1 > 1:
    print(f"\nWARNING: T1 uncertainty exceeds the fitted value itself (relative error "
          f"{err_T1/T1*100:.0f}%). No reliable recovery was resolved — check whether the "
          f"vdlist actually brackets the null point (see sign-change check above), and "
          f"whether S/N is adequate on the shortest/longest rows.")

# --- Figure ---
tau_fit = np.logspace(np.log10(tau_values[tau_values > 0].min()), np.log10(tau_values.max()), 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(tau_values * 1000, intensities, color="blue", label="data")
if FIT_INVERSION_FACTOR:
    ax.plot(tau_fit * 1000, t1_recovery(tau_fit, M0, T1, A), color="red",
            label=f"fit T1={T1*1000:.3f} ms, A={A:.2f}")
else:
    ax.plot(tau_fit * 1000, t1_recovery(tau_fit, M0, T1, 2.0), color="red",
            label=f"fit T1={T1*1000:.3f} ms")
ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Recovery time tau (ms)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("T1 relaxation — inversion recovery")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()

# --- CSV export ---
df = pd.DataFrame({
    "tau_ms": tau_values * 1000,
    "intensity": intensities,
    "noise_std": noise_std_per_row,
    "SNR": snr_per_row,
    "intensity_fit": (t1_recovery(tau_values, M0, T1, A) if FIT_INVERSION_FACTOR
                       else t1_recovery(tau_values, M0, T1, 2.0))
})
df["M0"] = M0
df["T1_ms"] = T1 * 1000
df["err_T1_ms"] = err_T1 * 1000
df["A"] = A
df["peak_position_ppm"] = peak_position_ppm
df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
print(f"\nResults exported to {OUTPUT_NAME}.csv")