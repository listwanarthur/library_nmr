import numpy as np
import nmrglue as ng
from library_nmr.core import find_grpdly_shift, process_row

LB = 10
PPM_WINDOW = (-10, 15)
REFERENCE_SHIFT_PPM = 2

def load_intensity(path, ns):
    dic, data = ng.bruker.read(path, read_procs=False)
    grpdly_shift = find_grpdly_shift(dic)
    dt = 1 / dic["acqus"]["SW_h"]
    spectrum = process_row(data.astype(complex), dt, LB, 0.0, 0.0, grpdly_shift)
    f = np.fft.fftshift(np.fft.fftfreq(len(data), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"] + REFERENCE_SHIFT_PPM
    lo, hi = sorted(PPM_WINDOW)
    mask = (delta >= lo) & (delta <= hi)
    mag = np.abs(spectrum[mask])
    return {
        "argmax": float(mag.max()) / ns,
        "argmax_ppm": float(delta[mask][np.argmax(mag)]),
        "integral": float(np.sum(mag)) / ns,
        "mean": float(np.mean(mag)) / ns,
        "rg": dic["acqus"].get("RG"), "te": dic["acqus"].get("TE"),
    }

points = {
    "L0=64  NS=64 ": (r"D:\Postdoc\Datas\LLZO-400-aug26\800", 64),
    "L0=64  NS=220": (r"D:\Postdoc\Datas\LLZO-400-aug26\802", 220),
    "L0=256 NS=64 ": (r"D:\Postdoc\Datas\LLZO-400-aug26\801", 64),
    "L0=256 NS=220": (r"D:\Postdoc\Datas\LLZO-400-aug26\803", 220),
}

results = {}
for label, (path, ns) in points.items():
    r = load_intensity(path, ns)
    results[label] = r
    print(f"{label}: RG={r['rg']} TE={r['te']:.2f}  peak_ppm={r['argmax_ppm']:7.2f}  "
          f"argmax/scan={r['argmax']:.4e}  mean/scan={r['mean']:.4e}  integral/scan={r['integral']:.4e}")

print()
for l0_label, a, b in [("L0=64", "L0=64  NS=64 ", "L0=64  NS=220"),
                        ("L0=256", "L0=256 NS=64 ", "L0=256 NS=220")]:
    for metric in ["argmax", "mean", "integral"]:
        va, vb = results[a][metric], results[b][metric]
        pct = 100 * (va - vb) / vb
        print(f"{l0_label} NS=64 vs NS=220, metric={metric:8s}: ecart = {pct:+.1f}%")
    print()

print("Prediction theorique plancher de bruit (NS=64 vs NS=220): "
      f"{100*((220/64)**0.5 - 1):.1f}%")