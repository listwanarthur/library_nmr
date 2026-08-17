import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng

from library_nmr.core import find_grpdly_shift

# ============================================================
# MQMAS (TRIPLE-QUANTUM) 2D PROCESSING — nmr_library
# Usage: edit the CONFIGURATION block below, then run.
#
# ASSUMED ACQUISITION SCHEME: States (hypercomplex) F1 acquisition, the most
# common for 2-pulse / z-filter split-t1 MQMAS. This means the ser file is
# assumed to contain 2*TD1 rows: for each t1 increment, one "cosine" (real)
# row followed by one "sine" (imaginary) row (standard Bruker States
# convention). If your actual pulse program uses TPPI, echo/antiecho, or a
# different row ordering, the F1 combination step below (Step 3) will need
# adjusting — send me the pulse program and I'll verify/adapt it.
#
# SPIN / COHERENCE ORDER: hardcoded for 7Li (I=3/2), which only has 3Q
# (triple-quantum) as a practically usable multiple-quantum order. The
# shearing ratio R=7/9 used below is the standard tabulated value for
# I=3/2, p=3 (Massiot et al. 1996 / Amoureux et al. conventions). This is a
# physical constant for this spin/order combination, not a free parameter.
# ============================================================

# === CONFIGURATION — only section to edit ===
PATH = r"C:\..."  # Bruker folder containing the 2D MQMAS experiment (ser file)
LB_F2 = 50   # line broadening in Hz, direct (F2) dimension
LB_F1 = 50   # line broadening in Hz, indirect (F1) dimension
PH0_F2, PH1_F2 = 0.0, 0.0  # F2 (direct dimension) phase, degrees — same convention as the 1D pipeline
ZF_FACTOR_F2 = 1  # zero-filling in F2 (direct)
ZF_FACTOR_F1 = 2  # zero-filling in F1 (indirect) — usually worth more since TD1 is typically small
SPIN_I = 1.5       # 7Li nuclear spin (3/2) — do not change unless processing a different nucleus
MQ_ORDER = 3       # triple-quantum, the only practical order for I=3/2
PPM_RANGE_F2 = (30, -30)   # display window, direct dimension
PPM_RANGE_F1 = None        # display window, indirect (sheared) dimension — None = full range
OUTPUT_NAME = "MQMAS_2D"
# ================================================


def shearing_ratio(I, p):
    """Standard shearing ratio R(I,p) for MQMAS, from the general formula relating
    p-quantum and single-quantum second-order quadrupolar shifts for a spin-I
    central transition. Correct closed-form values are tabulated in the MQMAS
    literature (Massiot et al. 1996; Amoureux, Fernandez, Frydman).
    Only I=3/2 (p=3) is implemented here since that is the only practically
    accessible order for 7Li; extend this dict if you later work with other
    half-integer quadrupolar nuclei (23Na, 27Al, etc.).
    """
    table = {
        (1.5, 3): 7 / 9,   # 7Li, 23Na, ... (I=3/2), triple-quantum
    }
    key = (I, p)
    if key not in table:
        raise NotImplementedError(
            f"Shearing ratio not tabulated here for I={I}, p={p}. "
            f"Add the correct literature value before proceeding — do not guess."
        )
    return table[key]


def process_f2_row(fid, dt, LB, ph0_rad, ph1_rad, grpdly_shift, zf_factor):
    """Same F2 (direct dimension) processing as the 1D pipeline: GRPDLY,
    apodization, FFT, phase. Returns the complex phased spectrum."""
    if grpdly_shift > 0:
        fid = np.roll(fid, -grpdly_shift)
    N = len(fid)
    t = np.arange(N) * dt
    w = np.exp(-t * LB)
    fid_apodized = w * fid
    signal_zf = np.concatenate([fid_apodized, np.zeros(zf_factor * N)])
    signal = np.fft.fftshift(np.fft.fft(signal_zf))
    n = np.arange(len(signal_zf))
    phase = ph0_rad + ph1_rad * (n / len(signal_zf))
    return signal * np.exp(1j * phase)


# === PROCESSING ===

dic, data = ng.bruker.read(PATH)
print(f"Raw data shape: {data.shape}")

TD1_rows = data.shape[0]  # 2 * number of t1 increments, under the States assumption above
n_t1 = TD1_rows // 2
if TD1_rows % 2 != 0:
    raise ValueError(
        f"Data has an odd number of rows ({TD1_rows}) — inconsistent with the assumed States "
        f"(cosine/sine pairs) hypercomplex F1 acquisition. Check the acquisition scheme."
    )
print(f"Interpreted as {n_t1} t1 increments x 2 (States hypercomplex)")

grpdly_shift = find_grpdly_shift(dic)
print(f"GRPDLY corrected: shifting each row by {grpdly_shift} points" if grpdly_shift > 0
      else "GRPDLY not found or zero — no correction applied")

SW_h_F2 = dic["acqus"]["SW_h"]
SFO1_F2 = dic["acqus"]["SFO1"]
O1_F2 = dic["acqus"]["O1"]
dt_F2 = 1 / SW_h_F2

# --- Step 1: process F2 (direct dimension) for every row ---
ph0_rad = np.deg2rad(PH0_F2)
ph1_rad = np.deg2rad(PH1_F2)
spectra_f2 = np.array([
    process_f2_row(data[i], dt_F2, LB_F2, ph0_rad, ph1_rad, grpdly_shift, ZF_FACTOR_F2)
    for i in range(TD1_rows)
])
n_points_f2 = spectra_f2.shape[1]

f2_freq = np.fft.fftshift(np.fft.fftfreq(n_points_f2, dt_F2))  # Hz
delta_f2_ppm = (O1_F2 - f2_freq) / SFO1_F2

# --- Step 2: build the complex F1 (t1) interferogram, States method ---
# Standard States reconstruction: the REAL part of the F2-FFT'd "cosine" row
# becomes Re(F1 series); the REAL part of the "sine" row becomes Im(F1 series).
cos_rows = spectra_f2[0::2].real  # (n_t1, n_points_f2)
sin_rows = spectra_f2[1::2].real
interferogram_f1 = cos_rows + 1j * sin_rows  # complex, shape (n_t1, n_points_f2)

# --- Step 3: F1 apodization ---
try:
    SW_h_F1 = dic["acqu2s"]["SW_h"]
    SFO1_F1 = dic["acqu2s"]["SFO1"]
    O1_F1 = dic["acqu2s"]["O1"]
except KeyError:
    raise KeyError(
        "Could not read F1 (indirect dimension) parameters from acqu2s. "
        "Check that this is really a 2D dataset processed with the correct nmrglue read."
    )
dt_F1 = 1 / SW_h_F1
t1_axis = np.arange(n_t1) * dt_F1
w_f1 = np.exp(-t1_axis * LB_F1)
interferogram_f1_apod = interferogram_f1 * w_f1[:, np.newaxis]

# --- Step 4: shearing — applied as a linear phase ramp in the mixed (t1, F2) domain. ---
# CAUTION (flagged 18/08, not silently changed — needs empirical verification against
# real data, not just a literature derivation, since the correct sign here depends on
# the coherence-pathway/echo-antiecho convention actually used during acquisition,
# which this script cannot know on its own):
#   By the Fourier shift theorem, multiplying a t1 column by exp(+i*2*pi*k*f2*t1)
#   shifts that point of the F1 spectrum by +k*f2 (Hz) after the FFT; multiplying by
#   exp(-i*2*pi*k*f2*t1) shifts it by -k*f2. The line below uses the MINUS sign
#   (shift by -k*f2), which previously disagreed with an old comment claiming a "+"
#   convention — that stale comment has been removed rather than "corrected", because
#   the actually-correct sign depends on details (P-type vs N-type / echo vs antiecho
#   selection in the pulse sequence used) that aren't recorded here.
#   HOW TO CHECK: after running this script, the isotropic F1 projection should show
#   NARROW, well-resolved ridges/peaks (that's the whole point of shearing). If instead
#   the 2D ridges run diagonally / the isotropic projection looks broadened or doubled
#   compared to the unsheared spectrum, flip the sign below (change -1j to +1j) and
#   re-run.
k = shearing_ratio(SPIN_I, MQ_ORDER)
print(f"Shearing ratio R(I={SPIN_I}, p={MQ_ORDER}) = {k:.4f}")

phase_ramp = np.exp(-1j * 2 * np.pi * k * f2_freq[np.newaxis, :] * t1_axis[:, np.newaxis])
interferogram_sheared = interferogram_f1_apod * phase_ramp

# --- Step 5: zero-fill and FFT along F1 ---
n_t1_zf = n_t1 * (ZF_FACTOR_F1 + 1)
interferogram_zf = np.concatenate(
    [interferogram_sheared, np.zeros((n_t1_zf - n_t1, interferogram_sheared.shape[1]), dtype=complex)],
    axis=0
)
spectrum_2d = np.fft.fftshift(np.fft.fft(interferogram_zf, axis=0), axes=0)

f1_freq = np.fft.fftshift(np.fft.fftfreq(n_t1_zf, dt_F1))  # Hz, SHEARED F1 axis
delta_f1_ppm = (O1_F1 - f1_freq) / SFO1_F1  # ppm, sheared indirect axis (isotropic dimension)

# --- Step 6: projections ---
# Sheared spectrum: isotropic peaks now line up at constant F1 -> a simple sum over
# F2 gives the (high-resolution) isotropic projection directly.
projection_f1_isotropic = np.sum(np.abs(spectrum_2d), axis=1)
# Sum over F1 gives the conventional MAS-averaged (broadened) 1D lineshape, for comparison.
projection_f2_mas = np.sum(np.abs(spectrum_2d), axis=0)

# === FIGURE ===
fig = plt.figure(figsize=(9, 8))
gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)

ax_2d = fig.add_subplot(gs[1:4, 0:3])
ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_2d)
ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_2d)

magnitude = np.abs(spectrum_2d)
levels = np.linspace(0.05, 1.0, 12) * magnitude.max()
ax_2d.contour(delta_f2_ppm, delta_f1_ppm, magnitude, levels=levels, colors="blue", linewidths=0.6)
ax_2d.invert_xaxis()
ax_2d.invert_yaxis()
ax_2d.set_xlabel("F2 — MAS dimension (ppm)")
ax_2d.set_ylabel("F1 — sheared isotropic dimension (ppm)")
if PPM_RANGE_F2 is not None:
    ax_2d.set_xlim(PPM_RANGE_F2[0], PPM_RANGE_F2[1])
if PPM_RANGE_F1 is not None:
    ax_2d.set_ylim(PPM_RANGE_F1[0], PPM_RANGE_F1[1])

ax_top.plot(delta_f2_ppm, projection_f2_mas, color="blue", linewidth=1)
ax_top.axis("off")

ax_right.plot(projection_f1_isotropic, delta_f1_ppm, color="blue", linewidth=1)
ax_right.axis("off")

plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()

print(f"\nF1 (sheared, isotropic) axis: {delta_f1_ppm.min():.2f} to {delta_f1_ppm.max():.2f} ppm, "
      f"{n_t1_zf} points")
print(f"F2 (MAS) axis: {delta_f2_ppm.min():.2f} to {delta_f2_ppm.max():.2f} ppm, {n_points_f2} points")