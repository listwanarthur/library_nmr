import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CSV_PATH = "T2_components_fit.csv"  # doit être dans le dossier où tu lances ce script

df = pd.read_csv(CSV_PATH)

def biexp_decay(t, A1, T2fast, A2, T2slow):
    return A1 * np.exp(-t / T2fast) + A2 * np.exp(-t / T2slow)

def fit_and_report(t, y, label, sigma_mode="prop"):
    usable = y > 0
    t_fit, y_fit = t[usable], y[usable]
    if sigma_mode == "prop":
        sigma = y_fit
    elif sigma_mode == "sqrt":
        sigma = np.sqrt(y_fit)
    else:
        sigma = None
    p0 = [0.9 * y_fit.max(), 150, 0.1 * y_fit.max(), t_fit[t_fit > t_fit.max() / 4].mean()]
    bounds = ([0, 1, 0, 100], [5 * y_fit.max(), 2000, 5 * y_fit.max(), 50000])
    try:
        popt, pcov = curve_fit(biexp_decay, t_fit, y_fit, p0=p0, sigma=sigma, bounds=bounds, maxfev=50000)
        perr = np.sqrt(np.diag(pcov))
        frac_fast = 100 * popt[0] / (popt[0] + popt[2])
        print(f"  {label}: T2_fast={popt[1]:.1f}+/-{perr[1]:.1f}us ({frac_fast:.1f}%)  "
              f"T2_slow={popt[3]:.1f}+/-{perr[3]:.1f}us ({100 - frac_fast:.1f}%)  (n={usable.sum()})")
    except RuntimeError as e:
        print(f"  {label}: fit failed ({e})")

for comp, col, drop_candidates in [
    ("narrow", "amplitude_narrow", [16, 313]),
    ("broad",  "amplitude_broad",  [100, 150]),
]:
    tau = df["tau_echo_us"].values
    y_all = df[col].values
    print(f"\n=== {comp} ===")
    fit_and_report(tau, y_all, "tous les points (pondere, comme le script principal)")
    for l0 in drop_candidates:
        mask = (df["L0"] != l0).values
        fit_and_report(tau[mask], y_all[mask], f"sans L0={l0}")
    mask = (~df["L0"].isin(drop_candidates)).values
    fit_and_report(tau[mask], y_all[mask], f"sans L0={drop_candidates}")
    fit_and_report(tau, y_all, "non pondere (sigma=None)", sigma_mode="none")
    fit_and_report(tau, y_all, "sigma=sqrt(I)", sigma_mode="sqrt")