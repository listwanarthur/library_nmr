import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0, parse_bruker_delay

# ============================================================
# PSEUDO-2D T2 EXTRACTION — nmr_library
# Usage: edit the CONFIGURATION block below, then run.
#
# Purpose: read a pseudo-2D spin-echo experiment (varying echo delay d4,
# from a vdlist), extract peak intensity per row, and fit a monoexponential
# T2 decay. Kept as a separate, single-purpose script (see main pipeline
# and multi-spectrum comparison scripts for other tasks).
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\221"  # Bruker folder containing the pseudo-2D experiment
LB = 50  # line broadening in Hz (higher than usual, common for solids / low S/N)
PH0_MANUAL = 0.0  # PHC0 in degrees, used if AUTO_PH0=False and READ_PHASE_FROM_PROCS=False
PH1 = 0.0  # PHC1 in degrees
AUTO_PH0 = True  # True: automatic PH0 search (see note below on which row is used)
READ_PHASE_FROM_PROCS = False  # read ph0/ph1 from TopSpin procs instead — priority over AUTO_PH0
USE_REAL_PART = True  # True: use Re(spectrum) (needs a properly phased spectrum, preferred).
                       # False: use |spectrum| (magnitude) — more forgiving if phase drifts
                       # slightly row to row, but loses lineshape information and can bias
                       # intensities upward at low S/N (magnitude never averages to zero noise).
PPM_MIN = -10  # search window to locate the peak (on the most intense row)
PPM_MAX = 10
EXTRACTION_HALFWIDTH_PPM = 0.2  # intensity extracted as the mean over
    # [peak_position - halfwidth, peak_position + halfwidth] rather than a single point,
    # for robustness against point-to-point noise. Set to 0 to use the single nearest point.
NOISE_REGION_PPM = (-40, -30)  # a signal-free region, used to estimate noise level and
    # report signal-to-noise ratio per row — critical diagnostic if the T2 fit looks flat/failed
OUTPUT_NAME = "T2_pseudo2D"
# ================================================


def t2_decay(tau, M0, T2):
    """Monoexponential T2 decay model. tau = d4 (single echo delay);
    total echo evolution time is 2*tau (refocusing before AND after)."""
    return M0 * np.exp(-2 * tau / T2)


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

# --- Read the echo delay list (vdlist) — REQUIRED, no silent fallback ---
# NOTE: dic["acqus"]["D"][4:4+n_rows] is NOT a valid substitute — D[4], D[5], D[6]...
# are distinct, unrelated Bruker delays (d4, d5, d6...), not successive values of a
# single varied delay. Using them would silently produce a physically meaningless
# T2 axis. If vdlist is missing, fix the acquisition/export rather than guessing.
try:
    with open(f"{PATH}/vdlist", "r") as f:
        d4_values = np.array([parse_bruker_delay(line) for line in f if line.strip()])
    print(f"d4 read from vdlist (converted to seconds): {d4_values}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"No vdlist found in {PATH}. The echo delay list must come from vdlist — "
        f"there is no safe substitute in acqus (D[4:4+n] are unrelated delays)."
    )
if len(d4_values) != data.shape[0]:
    raise ValueError(
        f"vdlist has {len(d4_values)} values but data has {data.shape[0]} rows — "
        f"check that the experiment finished correctly / matches this vdlist."
    )

# --- Determine phase (shared across all rows: same experiment, same receiver phase) ---
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
        # Use the row with the strongest signal (shortest echo delay = least decayed)
        # to determine PH0 — most reliable S/N for the phase search.
        idx_strongest = int(np.argmin(d4_values))
        signal_ref = process_row(data[idx_strongest], dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift)
        ph0_deg = find_best_ph0(signal_ref, np.deg2rad(ph1_deg))
        print(f"Optimal PH0 found (auto, on row {idx_strongest+1}, shortest d4): {ph0_deg:.3f}°")
    else:
        ph0_deg = PH0_MANUAL

ph0_rad = np.deg2rad(ph0_deg)
ph1_rad = np.deg2rad(ph1_deg)

# --- Process all rows with the SAME phase ---
# NOTE: process_row uses np.roll for GRPDLY (circular shift), which does NOT
# change the array length — so the ppm axis must use the ORIGINAL point count,
# not a shortened one.
n_points = data.shape[1]
f = np.fft.fftshift(np.fft.fftfreq(n_points, dt))
delta = (O1 - f) / SFO1

spectra = np.array([
    process_row(data[i], dt, LB, ph0_rad, ph1_rad, grpdly_shift)
    for i in range(data.shape[0])
])
spectra_real = spectra.real if USE_REAL_PART else np.abs(spectra)

# --- Locate the peak on the strongest row, then extract at that FIXED position
#     for every row (avoids picking up noise peaks at long/decayed delays) ---
idx_strongest = int(np.argmin(d4_values))
window_mask = (delta >= PPM_MIN) & (delta <= PPM_MAX)
idx_in_window = np.argmax(spectra_real[idx_strongest][window_mask])
peak_position_ppm = delta[window_mask][idx_in_window]
print(f"Peak located at {peak_position_ppm:.3f} ppm (from row {idx_strongest+1}, shortest d4)")

if EXTRACTION_HALFWIDTH_PPM > 0:
    extraction_mask = (delta >= peak_position_ppm - EXTRACTION_HALFWIDTH_PPM) & \
                       (delta <= peak_position_ppm + EXTRACTION_HALFWIDTH_PPM)
else:
    extraction_mask = np.zeros_like(delta, dtype=bool)
    extraction_mask[np.argmin(np.abs(delta - peak_position_ppm))] = True

intensities = np.array([np.mean(row[extraction_mask]) for row in spectra_real])

# --- Signal-to-noise diagnostic (important if the T2 fit looks flat / fails) ---
noise_mask = (delta >= min(NOISE_REGION_PPM)) & (delta <= max(NOISE_REGION_PPM))
noise_std_per_row = np.array([np.std(row[noise_mask]) for row in spectra_real])
snr_per_row = intensities / noise_std_per_row

for i in range(data.shape[0]):
    print(f"Row {i+1}: d4 = {d4_values[i]*1000:.3f} ms, intensity = {intensities[i]:.3e}, "
          f"noise std = {noise_std_per_row[i]:.3e}, S/N = {snr_per_row[i]:.1f}")

# --- T2 fit ---
popt, pcov = curve_fit(t2_decay, d4_values, intensities,
                        p0=[max(intensities), 0.1],
                        bounds=([0, 0], [np.inf, np.inf]))
uncertainties = np.sqrt(np.diag(pcov))
T2, err_T2 = popt[1], uncertainties[1]
print(f"\nT2 = {T2*1000:.3f} +/- {err_T2*1000:.3f} ms")

# --- Figure ---
d4_fit = np.logspace(np.log10(d4_values[0]), np.log10(d4_values[-1]), 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(2 * d4_values * 1000, intensities, color="blue", label="data")
ax.plot(2 * d4_fit * 1000, t2_decay(d4_fit, *popt),
        color="red", label=f"fit T2={T2*1000:.3f} ms")
ax.set_xscale("log")
ax.set_xlabel("Echo time 2×d4 (ms)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("T2 relaxation — solid echo")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()

# --- CSV export ---
df = pd.DataFrame({
    "d4_ms": d4_values * 1000,
    "echo_time_2d4_ms": 2 * d4_values * 1000,
    "intensity": intensities,
    "noise_std": noise_std_per_row,
    "SNR": snr_per_row,
    "intensity_fit": t2_decay(d4_values, *popt)
})
df["T2_ms"] = T2 * 1000
df["err_T2_ms"] = err_T2 * 1000
df["peak_position_ppm"] = peak_position_ppm
df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
print(f"Results exported to {OUTPUT_NAME}.csv")

if snr_per_row[idx_strongest] < 10:
    print(f"\nWARNING: S/N on the strongest row is only {snr_per_row[idx_strongest]:.1f}. "
          f"A flat/unresolved T2 fit is expected at this noise level regardless of the "
          f"true T2 — consider more scans (higher S/N) before trusting this fit.")
elif err_T2 / T2 > 1:
    print(f"\nWARNING: T2 uncertainty exceeds the fitted value itself (relative error "
          f"{err_T2/T2*100:.0f}%). No reliable decay was resolved in this delay range — "
          f"the true T2 is likely much longer than the maximum echo time probed "
          f"({2*max(d4_values)*1000:.2f} ms), or S/N is too low to see it.")