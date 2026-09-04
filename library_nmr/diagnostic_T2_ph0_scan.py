import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, process_row

# === CONFIGURATION ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\253"  # L0=1, tau=80us (meilleur S/N de la série T2)
LB = 10
PH1 = -49.524
ZF_FACTOR = 1

# Balayage large autour de la valeur trouvée par AUTO_PH0 (-133.100°) --
# on a vu ce matin sur exp236 que l'auto-recherche peut se tromper de ~30°.
PH0_CANDIDATES = np.arange(-160, -158, 1)  # -180, -170, ..., -90

# === lecture + FFT brute (sans phase) ===
dic, data = ng.bruker.read(PATH, read_procs=False)
grpdly_shift = find_grpdly_shift(dic)
N = data.shape[0]
dt = 1 / dic["acqus"]["SW_h"]
data_zf = np.concatenate([data, np.zeros(ZF_FACTOR * N, dtype=complex)])

f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
delta = delta + 2  # REFERENCE_SHIFT_PPM du script T2

# === superposition pour chaque PH0 candidat ===
fig, ax = plt.subplots(figsize=(9, 6))
for ph0 in PH0_CANDIDATES:
    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0), np.deg2rad(PH1), grpdly_shift)
    ax.plot(delta, spectrum.real, label=f"PH0 = {ph0:.0f}°")

ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("ppm")
ax.set_ylabel("Intensity (real part)")
ax.set_title("exp253 (L0=1, tau=80us) — balayage PH0 (référence auto = -133.1°)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("diagnostic_T2_ph0_scan.pdf")
plt.show()