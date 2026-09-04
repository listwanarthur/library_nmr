import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CSV_PATH = "T2_components_fit.csv"
df = pd.read_csv(CSV_PATH)

def monoexp_decay(t, A0, T2):
    return A0 * np.exp(-t / T2)

def biexp_decay(t, A1, T2fast, A2, T2slow):
    return A1 * np.exp(-t / T2fast) + A2 * np.exp(-t / T2slow)

for comp, col in [("narrow", "amplitude_narrow"), ("broad", "amplitude_broad")]:
    tau = df["tau_echo_us"].values
    y_all = df[col].values
    usable = y_all > 0
    t_fit, y_fit = tau[usable], y_all[usable]
    n = usable.sum()

    print(f"\n=== {comp} (n={n}) ===")

    p0_mono = [y_fit.max(), t_fit[len(t_fit)//2]]
    popt_m, pcov_m = curve_fit(monoexp_decay, t_fit, y_fit, p0=p0_mono, sigma=y_fit,
                                bounds=([0, 1], [10*y_fit.max(), 50000]), maxfev=50000)
    perr_m = np.sqrt(np.diag(pcov_m))
    resid_m = 100 * (y_fit - monoexp_decay(t_fit, *popt_m)) / y_fit
    chi2_m = np.sum(((y_fit - monoexp_decay(t_fit, *popt_m)) / y_fit) ** 2)
    print(f"  monoexp: T2={popt_m[1]:.1f}+/-{perr_m[1]:.1f}us, chi2_reduit={chi2_m/(n-2):.3f}")
    print(f"    residus (%): {np.round(resid_m, 1)}")

    p0_bi = [0.9*y_fit.max(), 150, 0.1*y_fit.max(), t_fit[t_fit > t_fit.max()/4].mean()]
    bounds_bi = ([0, 1, 0, 100], [5*y_fit.max(), 2000, 5*y_fit.max(), 50000])
    popt_b, pcov_b = curve_fit(biexp_decay, t_fit, y_fit, p0=p0_bi, sigma=y_fit,
                                bounds=bounds_bi, maxfev=50000)
    perr_b = np.sqrt(np.diag(pcov_b))
    resid_b = 100 * (y_fit - biexp_decay(t_fit, *popt_b)) / y_fit
    chi2_b = np.sum(((y_fit - biexp_decay(t_fit, *popt_b)) / y_fit) ** 2)
    print(f"  biexp:   T2_fast={popt_b[1]:.1f}+/-{perr_b[1]:.1f}us, T2_slow={popt_b[3]:.1f}+/-{perr_b[3]:.1f}us, "
          f"chi2_reduit={chi2_b/(n-4):.3f}")
    print(f"    residus (%): {np.round(resid_b, 1)}")