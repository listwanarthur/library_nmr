import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from library_nmr.core import find_grpdly_shift, process_row

# ============================================================
# ESTIMATION DE CQ (7Li) A PARTIR DES TRANSITIONS SATELLITES
# STATIQUES (exp307) -- script exploratoire, pas une figure finale.
# Physique : cf. discussion -- moyenne de poudre du motif quadripolaire
# au 1er ordre pour les transitions satellites d'un spin I=3/2.
# ============================================================

# === CONFIGURATION ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\307"
LB = 10
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
N_ORIENTATIONS = 4000
FIT_WINDOW_POS = (170, 350)   # cote +ppm, exclut la transition centrale
FIT_WINDOW_NEG = (-350, -170) # cote -ppm (symetrique)
SEED = 0
OUTPUT_NAME = "diagnostic_satellite_CQ_fit"
# ================================================

np.random.seed(SEED)

# --- lecture + FFT magnitude (identique a spectrum_static.py) ---
dic, data = ng.bruker.read(PATH)
grpdly_shift = find_grpdly_shift(dic)
N = data.shape[0]
dt = 1 / dic["acqus"]["SW_h"]
data_zf = np.concatenate([data, np.zeros(ZF_FACTOR * N, dtype=complex)])
spectrum = process_row(data_zf, dt, LB, 0.0, 0.0, grpdly_shift)
signal = np.abs(spectrum)
f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"] + REFERENCE_SHIFT_PPM
SFO1 = dic["acqus"]["SFO1"]  # MHz

# --- orientations de poudre (Monte-Carlo, isotrope) ---
cos_theta = np.random.uniform(-1, 1, N_ORIENTATIONS)
theta = np.arccos(cos_theta)
phi = np.random.uniform(0, 2 * np.pi, N_ORIENTATIONS)

HIST_RANGE = (-450, 450)
NBINS = 600


def unit_shape(ppm_axis, CQ_kHz, eta, broadening_ppm, sign):
    """Motif de poudre normalise (max=1) pour UNE transition satellite."""
    shift_kHz = -sign * (CQ_kHz / 4.0) * (3 * np.cos(theta) ** 2 - 1 - eta * np.sin(theta) ** 2 * np.cos(2 * phi))
    shift_ppm = shift_kHz * 1000.0 / SFO1
    hist, edges = np.histogram(shift_ppm, bins=NBINS, range=HIST_RANGE)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma_bins = max(broadening_ppm / (centers[1] - centers[0]), 0.5)
    hist_b = gaussian_filter1d(hist.astype(float), sigma_bins)
    shape = np.interp(ppm_axis, centers, hist_b)
    peak = shape.max()
    return shape / peak if peak > 0 else shape


def linear_fit(y, shape):
    """Resout y ~= a*shape + b (amplitude + baseline) par moindres carres lineaires."""
    A = np.vstack([shape, np.ones_like(shape)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b, a * shape + b


mask_pos = (delta >= FIT_WINDOW_POS[0]) & (delta <= FIT_WINDOW_POS[1])
mask_neg = (delta >= FIT_WINDOW_NEG[0]) & (delta <= FIT_WINDOW_NEG[1])


def objective(params):
    CQ_kHz, eta, broadening_ppm = params
    if CQ_kHz <= 0 or not (0 <= eta <= 1) or broadening_ppm <= 0:
        return 1e18
    sse = 0.0
    for sign, mask in [(+1, mask_pos), (-1, mask_neg)]:
        shape = unit_shape(delta[mask], CQ_kHz, eta, broadening_ppm, sign)
        _, _, model = linear_fit(signal[mask], shape)
        sse += np.sum((signal[mask] - model) ** 2)
    return sse


result = minimize(objective, x0=[60.0, 0.3, 15.0], method="Nelder-Mead",
                   options=dict(xatol=1e-2, fatol=1e2, maxiter=3000))
CQ_kHz, eta, broadening_ppm = result.x

print(f"Convergence : {result.success}, iterations : {result.nit}")
print(f"CQ = {CQ_kHz:.1f} kHz ({CQ_kHz/1000:.4f} MHz)")
print(f"eta = {eta:.3f}")
print(f"elargissement gaussien = {broadening_ppm:.1f} ppm")

# --- verification visuelle ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, sign, mask, title in [(axes[0], +1, mask_pos, "Satellite +ppm"),
                               (axes[1], -1, mask_neg, "Satellite -ppm")]:
    shape = unit_shape(delta[mask], CQ_kHz, eta, broadening_ppm, sign)
    a, b, model = linear_fit(signal[mask], shape)
    ax.plot(delta[mask], signal[mask], color="blue", lw=1, label="donnees")
    ax.plot(delta[mask], model, color="red", lw=1.5, label="modele poudre I=3/2")
    ax.set_title(title)
    ax.set_xlabel("ppm")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()