import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0, parse_bruker_delay

# ============================================================
# PSEUDO-2D T1 SATURATION-RECOVERY EXTRACTION — nmr_library
# Usage: edit the CONFIGURATION block below, then run.
#
# Purpose: read a pseudo-2D saturation-recovery experiment (comb-TAU-90-
# acquire, varying recovery time tau from a vdlist), extract peak intensity
# per row, and fit the T1 recovery curve.
#
# WHY SATURATION RECOVERY INSTEAD OF INVERSION RECOVERY: for quadrupolar
# nuclei (7Li included) with non-negligible quadrupolar coupling (satellites
# visible in the static/MAS spectrum), a single pulse calibrated as "180
# degrees" from a whole-powder nutation curve does not equally invert the
# central transition and the satellite transitions — it can fail to invert
# the central transition cleanly even when the nominal calibration looks
# correct. A saturation comb only needs to destroy Mz "well enough", not
# hit a precise flip angle, sidestepping that ambiguity. Unlike inversion
# recovery, the signal here NEVER changes sign — it grows monotonically
# from ~0 to M0.
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"C:\..."  # Bruker folder containing the pseudo-2D saturation-recovery experiment
LB = 50  # line broadening in Hz
PH0_MANUAL = 0.0
PH1 = 0.0
AUTO_PH0 = True  # automatic PH0 search, performed on the fully recovered (longest tau) row
READ_PHASE_FROM_PROCS = False
PPM_MIN = -10  # search window to locate the peak
PPM_MAX = 10
EXTRACTION_HALFWIDTH_PPM = 0.2
NOISE_REGION_PPM = (-40, -30)
FIT_SATURATION_FACTOR = True  # True: fit the saturation factor B freely (recommended — a real
    # saturation comb rarely destroys Mz perfectly, especially for quadrupolar satellites).
    # False: fix B=1.0 (ideal/complete saturation) and only fit M0, T1.
OUTPUT_NAME = "T1_saturation_recovery"
# ================================================


def t1_sat_recovery(tau, M0, T1, B):
    """Saturation-recovery model: M(tau) = M0 * (1 - B * exp(-tau/T1)).
    B=1.0 for ideal/complete saturation (Mz=0 right after the comb);
    B<1 if saturation is incomplete (some Mz survives the comb).
    """
    return M0 * (1 - B * np.exp(-tau / T1))


# === PROCESSING ===

dic, data = ng.bruker.read(PATH, read_procs=READ_PHASE_FROM_PROCS)
print(f"Data shape: {data.shape}  (n_rows, n_points)")

grpdly_shift = find_grpdly_shift(dic)
print(f"GRPDLY corrected: shifting each row by {grpdly_shift} points" if grpdly_shift > 0
      else "GRPDLY not found or zero — no correction applied")

SW_h = dic["acqus"]["SW_h"]
SFO1 = dic["acqus"]["SFO1"]
O1 = dic["acqus"]["O1"]
dt = 1 / SW_h

try:
    with open(f"{PATH}/vdlist", "r") as f:
        tau_values = np.array([parse_bruker_delay(line) for line in f if line.strip()])
    print(f"tau (recovery times) read from vdlist (seconds): {tau_values}")
except FileNotFoundError:
    raise FileNotFoundError(f"No vdlist found in {PATH}.")
if len(tau_values) != data.shape[0]:
    raise ValueError(f"vdlist has {len(tau_values)} values but data has {data.shape[0]} rows.")

idx_longest = int(np.argmax(tau_values))

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

n_points = data.shape[1]
f = np.fft.fftshift(np.fft.fftfreq(n_points, dt))
delta = (O1 - f) / SFO1

spectra_real = np.array([
    process_row(data[i], dt, LB, ph0_rad, ph1_rad, grpdly_shift).real
    for i in range(data.shape[0])
])

# --- Locate the peak on the strongest (longest tau, best S/N) row ---
window_mask = (delta >= PPM_MIN) & (delta <= PPM_MAX)
idx_in_window = np.argmax(spectra_real[idx_longest][window_mask])
peak_position_ppm = delta[window_mask][idx_in_window]
print(f"Peak located at {peak_position_ppm:.3f} ppm (from row {idx_longest+1}, longest tau)")

if EXTRACTION_HALFWIDTH_PPM > 0:
    extraction_mask = (delta >= peak_position_ppm - EXTRACTION_HALFWIDTH_PPM) & \
                       (delta <= peak_position_ppm + EXTRACTION_HALFWIDTH_PPM)
else:
    extraction_mask = np.zeros_like(delta, dtype=bool)
    extraction_mask[np.argmin(np.abs(delta - peak_position_ppm))] = True

intensities = np.array([np.mean(row[extraction_mask]) for row in spectra_real])

noise_mask = (delta >= min(NOISE_REGION_PPM)) & (delta <= max(NOISE_REGION_PPM))
noise_std_per_row = np.array([np.std(row[noise_mask]) for row in spectra_real])
snr_per_row = intensities / noise_std_per_row

for i in range(data.shape[0]):
    print(f"Row {i+1}: tau = {tau_values[i]*1000:.3f} ms, intensity = {intensities[i]:.3e}, "
          f"noise std = {noise_std_per_row[i]:.3e}, S/N = {snr_per_row[i]:.1f}")

# --- Initial T1 guess: time to reach (1 - 1/e) of the plateau value ---
M0_guess = intensities[idx_longest]
target = M0_guess * (1 - 1 / np.e)
idx_above = np.where(intensities >= target)[0]
if len(idx_above) > 0 and idx_above[0] > 0:
    i1 = idx_above[0]
    t_a, t_b = tau_values[i1 - 1], tau_values[i1]
    y_a, y_b = intensities[i1 - 1], intensities[i1]
    T1_guess = t_a + (target - y_a) * (t_b - t_a) / (y_b - y_a)
    print(f"\nT1 guess from (1-1/e) crossing: ~{T1_guess*1000:.3f} ms")
else:
    T1_guess = np.median(tau_values)
    print(f"\nWARNING: recovery never clearly reaches (1-1/e) of plateau within tested tau range — "
          f"the vdlist may not resolve this component's rise well. Using a fallback T1 guess.")

# --- T1 fit ---
if FIT_SATURATION_FACTOR:
    popt, pcov = curve_fit(t1_sat_recovery, tau_values, intensities,
                            p0=[M0_guess, T1_guess, 1.0],
                            bounds=([0, 0, 0], [np.inf, np.inf, 2]), maxfev=20000)
    M0, T1, B = popt
    err_M0, err_T1, err_B = np.sqrt(np.diag(pcov))
    print(f"\nM0 = {M0:.3e} +/- {err_M0:.3e}")
    print(f"T1 = {T1*1000:.3f} +/- {err_T1*1000:.3f} ms")
    print(f"B  = {B:.3f} +/- {err_B:.3f}  (1.0 = ideal/complete saturation)")
else:
    def t1_sat_recovery_fixed_B(tau, M0, T1):
        return t1_sat_recovery(tau, M0, T1, 1.0)
    popt, pcov = curve_fit(t1_sat_recovery_fixed_B, tau_values, intensities,
                            p0=[M0_guess, T1_guess], bounds=([0, 0], [np.inf, np.inf]), maxfev=20000)
    M0, T1 = popt
    B = 1.0
    err_M0, err_T1 = np.sqrt(np.diag(pcov))
    print(f"\nM0 = {M0:.3e} +/- {err_M0:.3e}")
    print(f"T1 = {T1*1000:.3f} +/- {err_T1*1000:.3f} ms  (B fixed at 1.0)")

if err_T1 / T1 > 1:
    print(f"\nWARNING: T1 uncertainty exceeds the fitted value itself (relative error "
          f"{err_T1/T1*100:.0f}%). Consider extending the tau range (shorter and/or longer) "
          f"or checking S/N on the shortest rows.")

# --- Figure ---
tau_fit = np.logspace(np.log10(tau_values[tau_values > 0].min()), np.log10(tau_values.max()), 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(tau_values * 1000, intensities, color="blue", label="data")
if FIT_SATURATION_FACTOR:
    ax.plot(tau_fit * 1000, t1_sat_recovery(tau_fit, M0, T1, B), color="red",
            label=f"fit T1={T1*1000:.3f} ms, B={B:.2f}")
else:
    ax.plot(tau_fit * 1000, t1_sat_recovery(tau_fit, M0, T1, 1.0), color="red",
            label=f"fit T1={T1*1000:.3f} ms")
ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Recovery time tau (ms)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("T1 relaxation — saturation recovery")
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
    "intensity_fit": (t1_sat_recovery(tau_values, M0, T1, B) if FIT_SATURATION_FACTOR
                       else t1_sat_recovery(tau_values, M0, T1, 1.0))
})
df["M0"] = M0
df["T1_ms"] = T1 * 1000
df["err_T1_ms"] = err_T1 * 1000
df["B"] = B
df["peak_position_ppm"] = peak_position_ppm
df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
print(f"\nResults exported to {OUTPUT_NAME}.csv")