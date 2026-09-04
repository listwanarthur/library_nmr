import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, process_row

PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\236"
LB = 10
ZF_FACTOR = 1
PH0 = 21  # deja valide sur le pic central
REFERENCE_SHIFT_PPM = 2

PH1_CANDIDATES = np.arange(-150, 60, 20)  # balayage large autour de la valeur actuelle -49.524

dic, data = ng.bruker.read(PATH, read_procs=False)
grpdly_shift = find_grpdly_shift(dic)
N = data.shape[0]
dt = 1 / dic["acqus"]["SW_h"]
data_zf = np.concatenate([data, np.zeros(ZF_FACTOR * N, dtype=complex)])

f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
delta = delta + REFERENCE_SHIFT_PPM

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ph1 in PH1_CANDIDATES:
    spectrum = process_row(data_zf, dt, LB, np.deg2rad(PH0), np.deg2rad(ph1), grpdly_shift)
    axes[0].plot(delta, spectrum.real, label=f"PH1={ph1:.0f}")
    axes[1].plot(delta, spectrum.real, label=f"PH1={ph1:.0f}")

axes[0].set_xlim(150, 90)    # bande de rotation vers +120 ppm (axe ppm inverse)
axes[0].set_title("Bande ~+120 ppm")
axes[1].set_xlim(-90, -150)  # bande de rotation vers -120 ppm
axes[1].set_title("Bande ~-120 ppm")
for ax in axes:
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("diagnostic_T1_ph1_sidebands_scan.pdf")
plt.show()