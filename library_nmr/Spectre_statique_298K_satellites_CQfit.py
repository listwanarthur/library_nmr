import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from library_nmr.core import find_grpdly_shift, process_row
from library_nmr.agr_export import export_agr

# ============================================================
# Spectre_statique_298K_satellites -- WITH CQ FIT OVERLAY
#
# Same physics/model as diagnostic_satellite_CQ_fit.py (unchanged -- CQ/eta/
# broadening fit on the two satellite windows, first-order quadrupolar
# powder pattern, Monte-Carlo orientation average for I=3/2). What's new:
# this turns it into an actual article figure -- full +/-400ppm spectrum
# (like Spectre_statique_298K_satellites.agr) with the fitted powder-pattern
# model drawn on top of the two satellite wings, exported via export_agr
# like the other reference figures (not just a diagnostic matplotlib PDF).
#
# Baseline : quadratique (b0+b1*x+b2*x^2). Un terme cubique testé faisait
# s'effondrer l'elargissement gaussien (absorbait une partie de la forme
# physique reelle au lieu de corriger la baseline) -- quadratique est la
# meilleure version, la plus proche de l'estimation GIPAW de reference
# (valeur exacte non reproduite ici -- depot public, manuscrit non publie).
#
# La fenetre de fit exclut l'epaulement de la transition centrale (donnee
# trop raide pour qu'un polynome de degre raisonnable la suive sans
# deformer la forme physique). La fenetre d'AFFICHAGE du modele est alignee
# sur la fenetre de fit -- pas d'extrapolation dans une zone mal reproduite ;
# seule la donnee brute (bleu) y reste visible.
#
# Still exploratory/preliminary (see resume): no error bar on CQ yet (single
# Nelder-Mead point estimate, no bootstrap), no independent GIPAW cross-
# check, and magnitude mode likely smooths the powder pattern a bit.
# Present as illustrative/preliminary in the article text, not a final value.
# ============================================================

# === CONFIGURATION ===
PATH = (r"D:\Postdoc\Datas\LLZO-400-aug26\853")
EXP_NUM = PATH.rstrip("\\").split("\\")[-1]  # derive run label from PATH -- avoid stale hardcoded exp number in legends
LB = 10
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
N_ORIENTATIONS = 4000
FIT_WINDOW_POS = (170, 350)      # exclut l'epaulement raide pres du centre
FIT_WINDOW_NEG = (-350, -170)    # symetrique
DISPLAY_WINDOW_POS = (170, 350)  # aligne sur la fenetre de fit -- pas d'extrapolation
DISPLAY_WINDOW_NEG = (-350, -170)
FULL_VIEW_PPM = (-400, 400)      # meme vue que Spectre_statique_298K_satellites.agr
SEED = 0
OUTPUT_NAME = r"D:\Postdoc\Figures\Spectre_statique_298K_satellites_CQfit"
# ================================================

np.random.seed(SEED)

# --- lecture + FFT magnitude (identique a spectrum_static.py / diagnostic_satellite_CQ_fit.py) ---
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


def linear_fit(y, shape, x):
    """Resout y ~= a*shape + b0 + b1*x + b2*x^2 (amplitude + baseline
    quadratique en ppm) par moindres carres lineaires."""
    A = np.vstack([shape, np.ones_like(shape), x, x**2]).T
    (a, b0, b1, b2), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b0, b1, b2, a * shape + b0 + b1 * x + b2 * x**2


mask_pos = (delta >= FIT_WINDOW_POS[0]) & (delta <= FIT_WINDOW_POS[1])
mask_neg = (delta >= FIT_WINDOW_NEG[0]) & (delta <= FIT_WINDOW_NEG[1])


def objective(params):
    CQ_kHz, eta, broadening_ppm = params
    if CQ_kHz <= 0 or not (0 <= eta <= 1) or broadening_ppm <= 0:
        return 1e18
    sse = 0.0
    for sign, mask in [(+1, mask_pos), (-1, mask_neg)]:
        shape = unit_shape(delta[mask], CQ_kHz, eta, broadening_ppm, sign)
        _, _, _, _, model = linear_fit(signal[mask], shape, delta[mask])
        sse += np.sum((signal[mask] - model) ** 2)
    return sse


# === STEP 1: fit CQ/eta/broadening on the two satellite windows ===
result = minimize(objective, x0=[60.0, 0.3, 15.0], method="Nelder-Mead",
                   options=dict(xatol=1e-2, fatol=1e2, maxiter=3000))
CQ_kHz, eta, broadening_ppm = result.x

print(f"Convergence : {result.success}, iterations : {result.nit}")
print(f"CQ = {CQ_kHz:.1f} kHz ({CQ_kHz/1000:.4f} MHz)")
print(f"eta = {eta:.3f}")
print(f"elargissement gaussien = {broadening_ppm:.1f} ppm")

# --- amplitude+baseline (a,b0,b1,b2) fixes par le fit -- reutilises pour
# dessiner le modele sur la fenetre d'AFFICHAGE (identique a la fenetre de
# fit -- pas d'extrapolation) ---
shape_fit_pos = unit_shape(delta[mask_pos], CQ_kHz, eta, broadening_ppm, +1)
a_pos, b0_pos, b1_pos, b2_pos, _ = linear_fit(signal[mask_pos], shape_fit_pos, delta[mask_pos])
shape_fit_neg = unit_shape(delta[mask_neg], CQ_kHz, eta, broadening_ppm, -1)
a_neg, b0_neg, b1_neg, b2_neg, _ = linear_fit(signal[mask_neg], shape_fit_neg, delta[mask_neg])

# === STEP 2: build the display curves ===
mask_full = (delta >= FULL_VIEW_PPM[0]) & (delta <= FULL_VIEW_PPM[1])
x_full, y_full = delta[mask_full], signal[mask_full]

mask_disp_pos = (delta >= DISPLAY_WINDOW_POS[0]) & (delta <= DISPLAY_WINDOW_POS[1])
mask_disp_neg = (delta >= DISPLAY_WINDOW_NEG[0]) & (delta <= DISPLAY_WINDOW_NEG[1])
x_model_pos = delta[mask_disp_pos]
y_model_pos = (a_pos * unit_shape(x_model_pos, CQ_kHz, eta, broadening_ppm, +1)
               + b0_pos + b1_pos * x_model_pos + b2_pos * x_model_pos**2)
x_model_neg = delta[mask_disp_neg]
y_model_neg = (a_neg * unit_shape(x_model_neg, CQ_kHz, eta, broadening_ppm, -1)
               + b0_neg + b1_neg * x_model_neg + b2_neg * x_model_neg**2)

# === STEP 3: export (matplotlib PDF/PNG + .agr, same convention as the other figures) ===
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(x_full, y_full, color="blue", lw=0.8, label=f"Static, D1=150s (exp{EXP_NUM})")
ax.plot(x_model_pos, y_model_pos, color="red", lw=1.8, label=f"CQ fit ({CQ_kHz:.0f} kHz, eta={eta:.2f})")
ax.plot(x_model_neg, y_model_neg, color="red", lw=1.8)
ax.set_xlim(FULL_VIEW_PPM[1], FULL_VIEW_PPM[0])  # inverted, NMR convention
ax.set_xlabel(r"$^7$Li NMR shift (ppm)")
ax.set_ylabel("Intensity (a.u., magnitude)")
ax.set_title(r"$^7$Li static spectrum -- satellite transitions, preliminary $C_Q$ fit")
ax.legend(loc="upper right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()

agr_series = [
    dict(x=x_full, y=y_full, mode="line", color="blue", legend=f"Static, D1=150s (exp{EXP_NUM})"),
    dict(x=x_model_pos, y=y_model_pos, mode="line", color="red",
         legend=f"CQ fit (preliminary): CQ={CQ_kHz:.0f}kHz, eta={eta:.2f}"),
    dict(x=x_model_neg, y=y_model_neg, mode="line", color="red"),
]
export_agr(f"{OUTPUT_NAME}.agr", agr_series,
           xlabel="7Li NMR shift (ppm)", ylabel="Intensity (a.u., magnitude)",
           xlog=False, invert_x=True,
           title="7Li static spectrum -- satellite transitions, preliminary CQ fit")

print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")
print("Reminder: preliminary estimate -- no error bar yet, no independent GIPAW cross-check.")