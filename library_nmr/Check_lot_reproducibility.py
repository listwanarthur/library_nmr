import numpy as np
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, process_row

# ============================================================
# CHECK_LOT_REPRODUCIBILITY -- diagnostic melange de lot NS / reproductibilite
# Mode magnitude, meme convention que T1_recovery_static_330K.py (sonde
# statique -- phasage documente comme instable sur cette raie, cf. AUTO_PH0
# = False dans ce script).
# ============================================================

BASE = r"D:\Postdoc\Datas\LLZO-400-aug26"
LB = 10  # meme LB que T1_recovery_static_330K.py
PEAK_PPM_WINDOW = (-10, 15)  # meme fenetre que 298K/330K (fix du 26/08)


def load_spectrum_magnitude(exp):
    """Lecture Bruker + GRPDLY + apodisation + FFT, magnitude uniquement
    (ph0=ph1=0, comme T1_recovery_static_330K.py)."""
    path = f"{BASE}\\{exp}"
    dic, data = ng.bruker.read(path, read_procs=False)
    dt = 1 / dic["acqus"]["SW_h"]
    grpdly_shift = find_grpdly_shift(dic)
    spectrum = process_row(data, dt, LB, ph0_rad=0.0, ph1_rad=0.0, grpdly_shift=grpdly_shift)
    f = np.fft.fftshift(np.fft.fftfreq(len(data), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    ns = dic["acqus"]["NS"]
    return delta, np.abs(spectrum), ns


def peak_intensity_per_scan(exp, ppm_window=PEAK_PPM_WINDOW):
    """Meme logique que get_peak_intensity_magnitude, /NS pour comparer
    des acquisitions a NS different."""
    delta, mag, ns = load_spectrum_magnitude(exp)
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    return float(mag[mask].max()) / ns


def noise_floor_per_scan(exp):
    """Moyenne sur tout le spectre / NS -- pour les points sans signal reel
    (Bloc B, plancher de bruit)."""
    _, mag, ns = load_spectrum_magnitude(exp)
    return float(mag.mean()) / ns


def compare(exp_ref, exp_test, label, metric):
    v_ref, v_test = metric(exp_ref), metric(exp_test)
    ecart = 100 * (v_test / v_ref - 1)
    print(f"{label} : exp{exp_ref}={v_ref:.4e}  exp{exp_test}={v_test:.4e}  ecart={ecart:+.1f}%")
    return ecart


if __name__ == "__main__":
    print("=== Bloc A : lot NS=32 (officiel, exp503) vs NS=128 (test, exp540), D1=0.5s ===")
    compare(503, 540, "peak/scan (-10,15ppm), magnitude", peak_intensity_per_scan)

    print("\n=== Bloc B : plancher de bruit T2 330K, NS=220 vs NS=64 ===")
    compare(524, 541, "L0=64  mean/scan", noise_floor_per_scan)
    compare(525, 542, "L0=256 mean/scan", noise_floor_per_scan)

    print("\n=== Reproductibilite grille T1 330K refaite (38 points, 2 series) ===")
    s1, s2 = list(range(543, 562)), list(range(562, 581))
    d1_labels = [0.02, 0.05, 0.2, 0.35, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8,
                 15, 30, 60, 120, 150, 250]
    for d1, e1, e2 in zip(d1_labels, s1, s2):
        compare(e1, e2, f"D1={d1}s", peak_intensity_per_scan)