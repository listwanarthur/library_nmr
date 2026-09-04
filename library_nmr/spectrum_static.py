import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, process_row
from library_nmr.agr_export import export_agr

# ============================================================
# SPECTRE STATIQUE 1D — mode magnitude
# Deux exports du meme spectre traite une seule fois :
#   - vue resserree : argument narrow/broad, comparaison avec les figures MAS
#   - vue large : montre les transitions satellites quadripolaires a ~+/-200ppm
# ============================================================

# === CONFIGURATION ===
PATH = r"D:\Postdoc\Datas\LLZO-400-aug26\850"
LB = 10
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1

OUTPUTS = [
    (r"D:\Postdoc\Figures\Spectre_statique_298K", (150, -150),
     "7Li static spectrum, 298K (magnitude mode, exp307) - central transition"),
    (r"D:\Postdoc\Figures\Spectre_statique_298K_satellites", (400, -400),
     "7Li static spectrum, 298K (magnitude mode, exp307) - full range, satellite transitions"),
]
# ================================================

dic, data = ng.bruker.read(PATH)
grpdly_shift = find_grpdly_shift(dic)
print(f"GRPDLY: {'corrige, decalage de ' + str(grpdly_shift) + ' points' if grpdly_shift > 0 else 'non trouve ou nul'}")

N = data.shape[0]
dt = 1 / dic["acqus"]["SW_h"]
data_zf = np.concatenate([data, np.zeros(ZF_FACTOR * N, dtype=complex)])

spectrum = process_row(data_zf, dt, LB, 0.0, 0.0, grpdly_shift)
signal = np.abs(spectrum)

f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"] + REFERENCE_SHIFT_PPM

print(f"Etendue de l'axe ppm : {delta.min():.1f} a {delta.max():.1f}")

# CSV complet, une seule fois (donnee brute identique pour les deux vues)
pd.DataFrame({"ppm": delta, "intensity": signal}).to_csv(
    r"D:\Postdoc\Figures\Spectre_statique_298K_full.csv", index=False)

for output_name, zoom, title in OUTPUTS:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(delta, signal, color="blue", linewidth=1)
    ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Chemical shift (ppm)")
    ax.set_ylabel("Magnitude intensity (a.u.)")
    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1])
    plt.savefig(f"{output_name}.pdf")
    plt.show()

    export_agr(
        f"{output_name}.agr",
        series=[dict(x=delta, y=signal, mode="line", color="blue", legend="magnitude spectrum")],
        xlabel="Chemical shift (ppm)", ylabel="Magnitude intensity (a.u.)",
        invert_x=True, xlim=zoom,
        title=title,
    )