import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, parse_bruker_delay
from library_nmr.agr_export import export_agr

# ============================================================
# CHECK_DRIFT — standalone diagnostic for pseudo-2D Bruker datasets
# Usage: edit PATH below, then run.
# Purpose: checks whether peak position/intensity drift vs ROW INDEX (not vs
# delay) on long unlocked pseudo-2D acquisitions — a trend vs index, independent
# of vd/vc, signals B0/shim/temperature drift rather than real T1/T2 physics.
# Works on any 1D-per-row pseudo-2D dataset.
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\219"
LB = 50
PPM_MIN, PPM_MAX = -25, 25  # window covering the full lineshape (both components); widen if peaks are wider
OUTPUT_NAME = "drift_check"
# ================================================
# NOTE: works in MAGNITUDE mode (|spectrum|), not absorption — phase-independent,
# so it sidesteps per-row autophasing noise entirely.


def process_row_magnitude(fid, dt, LB, grpdly_shift):
    if grpdly_shift > 0:
        fid = np.roll(fid, -grpdly_shift)
    N = len(fid)
    t = np.arange(N) * dt
    w = np.exp(-t * LB)
    fid_apodized = w * fid
    signal = np.fft.fftshift(np.fft.fft(fid_apodized))
    return np.abs(signal)


# === PROCESSING ===

dic, data = ng.bruker.read(PATH, read_procs=False)
print(f"Data shape: {data.shape}  (n_rows, n_points)")

grpdly_shift = find_grpdly_shift(dic)
SW_h = dic["acqus"]["SW_h"]
SFO1 = dic["acqus"]["SFO1"]
O1 = dic["acqus"]["O1"]
dt = 1 / SW_h

# CAVEAT: "drift vs row index" only separates from "physics vs delay" if row
# index isn't monotonically correlated with delay — check monotonicity below.
delay_values = None
delay_source = None
for fname in ("vdlist", "vclist"):
    try:
        with open(f"{PATH}/{fname}", "r") as f:
            delay_values = np.array([parse_bruker_delay(line) for line in f if line.strip()])
        delay_source = fname
        break
    except FileNotFoundError:
        continue
if delay_values is None:
    print("NOTE: no vdlist/vclist found next to the dataset — cannot check whether "
          "delay order is monotonic. Interpret any 'drift' below with caution.")
elif len(delay_values) == data.shape[0]:
    diffs = np.diff(delay_values)
    is_monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)
    if is_monotonic:
        print(f"NOTE: {delay_source} is monotonic in row order (delays acquired in "
              f"strictly increasing/decreasing order). Row index and delay are therefore "
              f"confounded here: a real two-component T1/T2 relaxation curve WILL also "
              f"produce a trend vs row index, and this script CANNOT distinguish that from "
              f"genuine B0/shim drift. Treat any WARNING below as inconclusive unless you "
              f"independently know the physics shouldn't produce this trend (e.g. from the "
              f"already-fitted T1/T2 curve), or re-acquire with a randomized delay order to "
              f"get a clean drift test.")
    else:
        print(f"{delay_source} is NOT monotonic in row order — row-index trends below are "
              f"not confounded with relaxation and are a meaningful drift diagnostic.")
else:
    print(f"NOTE: {delay_source} has {len(delay_values)} values but data has {data.shape[0]} "
          f"rows — mismatch, skipping the monotonicity check.")

n_rows = data.shape[0]
n_points = data.shape[1]
f = np.fft.fftshift(np.fft.fftfreq(n_points, dt))
delta = (O1 - f) / SFO1
window_mask = (delta >= PPM_MIN) & (delta <= PPM_MAX)
delta_window = delta[window_mask]

positions = np.zeros(n_rows)
intensities = np.zeros(n_rows)
peak_heights = np.zeros(n_rows)

for i in range(n_rows):
    sig = process_row_magnitude(data[i], dt, LB, grpdly_shift)

    y = sig[window_mask]  # magnitude is always >= 0, no sign/phase issues
    positions[i] = np.sum(delta_window * y) / np.sum(y)
    intensities[i] = np.trapezoid(y, delta_window)
    peak_heights[i] = y.max()

    print(f"Row {i+1}/{n_rows}: peak position={positions[i]:.4f} ppm, "
          f"integral={intensities[i]:.3e}, height={peak_heights[i]:.3e}")

# --- Drift statistics: position/intensity vs ROW INDEX (not vs delay) ---
rows = np.arange(1, n_rows + 1)
valid = ~np.isnan(positions)

pos_slope, pos_intercept = np.polyfit(rows[valid], positions[valid], 1)
int_slope, int_intercept = np.polyfit(rows[valid], intensities[valid], 1)

pos_drift_total = pos_slope * (n_rows - 1)
int_drift_pct = (int_slope * (n_rows - 1)) / np.mean(intensities) * 100

print(f"\n--- Drift summary ---")
print(f"Peak position: {pos_slope:.5f} ppm/row -> total drift over the run = "
      f"{pos_drift_total:.4f} ppm ({pos_drift_total * SFO1:.1f} Hz)")
print(f"Integrated intensity: {int_slope:.3e}/row -> total drift over the run = "
      f"{int_drift_pct:.1f}% of the mean intensity")

if abs(pos_drift_total * SFO1) > 5:
    print("  WARNING: peak position drifted by more than 5 Hz over the course of the "
          "acquisition. This is consistent with B0/shim drift over a long unlocked run.")
if abs(int_drift_pct) > 10:
    print("  WARNING: integrated intensity shows a >10% systematic trend vs ROW INDEX "
          "(not vs delay) — this is scale/gain/shim drift, not T1/T2 physics, and could "
          "be masking or distorting the real relaxation curve.")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(rows, positions, "o-", color="tab:blue")
axes[0].plot(rows, pos_slope * rows + pos_intercept, "--", color="black",
             label=f"slope={pos_slope*SFO1:.2f} Hz/row")
axes[0].set_xlabel("Row index (acquisition order)")
axes[0].set_ylabel("Peak position (ppm)")
axes[0].set_title("Peak position vs row index")
axes[0].legend(fontsize=8)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(rows, intensities, "o-", color="tab:red")
axes[1].plot(rows, int_slope * rows + int_intercept, "--", color="black",
             label=f"drift={int_drift_pct:.1f}% total")
axes[1].set_xlabel("Row index (acquisition order)")
axes[1].set_ylabel("Integrated intensity (a.u.)")
axes[1].set_title("Intensity vs row index (drift check)")
axes[1].legend(fontsize=8)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()
print(f"\nPlot saved to {OUTPUT_NAME}.pdf")

export_agr(
    f"{OUTPUT_NAME}_position.agr",
    series=[
        dict(x=rows, y=positions, mode="symbol", color="blue", legend="position"),
        dict(x=rows, y=pos_slope * rows + pos_intercept, mode="line", color="black",
             legend=f"slope={pos_slope*SFO1:.2f} Hz/row"),
    ],
    xlabel="Row index (acquisition order)", ylabel="Peak position (ppm)",
    title="Peak position vs row index",
)
export_agr(
    f"{OUTPUT_NAME}_intensity.agr",
    series=[
        dict(x=rows, y=intensities, mode="symbol", color="red", legend="intensity"),
        dict(x=rows, y=int_slope * rows + int_intercept, mode="line", color="black",
             legend=f"drift={int_drift_pct:.1f}% total"),
    ],
    xlabel="Row index (acquisition order)", ylabel="Integrated intensity (a.u.)",
    title="Intensity vs row index (drift check)",
)