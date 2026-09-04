import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T2_global_statique_345K -- campagne a venir (semaine du 08/09 ou apres)
#
# Grille allegee (diagnostic 315K du 02/09) : L0=2,3,4,6,8,11,16,22,64
# (decho=20-220us + verif plancher a 640us), NS=64, D1=15s. Pas de
# L0=32/45 (zone ambigue/contaminee, cf. l'essai rate a 315K) ni de L0=90
# (64 suffit a confirmer le plateau).
#
# A FAIRE AVANT DE LANCER :
#  1. Remplir EXP_START ci-dessous avec le numero de la premiere
#     experience Bruker de la serie (9 experiences consecutives, meme
#     ordre que _L0_LABELS).
#  2. Verifier RG (rga refait juste avant si besoin).
#  3. NOISE_FLOOR_L0 = {64} par defaut -- A VERIFIER (le seuil peut
#     differer de 315K/330K/360K) : le point L0=64 doit etre net en
#     dessous de la tendance signal avant de faire confiance au T2 final.
# ============================================================

# === CONFIGURATION -- only section to edit ===
EXP_START = None   # <-- A REMPLIR (ex: 500)

_L0_LABELS = [2, 3, 4, 6, 8, 11, 16, 22, 64]
if EXP_START is None:
    raise ValueError("EXP_START n'est pas rempli -- indique le numero de la premiere "
                      "experience Bruker de la serie avant de lancer.")
_EXP_LIST = list(range(EXP_START, EXP_START + len(_L0_LABELS)))

DATA_DIR = r"D:\Postdoc\Datas\LLZO-400-sep26"   # <-- A VERIFIER/ADAPTER selon le dossier reel

DATASETS = {
    l0: (rf"{DATA_DIR}\{exp}", 64)
    for l0, exp in zip(_L0_LABELS, _EXP_LIST)
}

# Presume indiscernable du plancher de bruit -- garde dans DATASETS pour
# affichage/audit, exclu du fit par defaut. A VERIFIER (cf. tete de fichier).
NOISE_FLOOR_L0 = {64}

LB = 10
PH0_MANUAL = 0.0       # irrelevant en mode magnitude
PH1 = 0.0              # irrelevant en mode magnitude
AUTO_PH0 = False
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
PEAK_PPM_WINDOW = (-10, 15)
PPM_OUTLIER_THRESHOLD = 5.0
FORCE_INCLUDE_L0 = []   # a remplir si un point signal (L0<=22) sort avec
                         # un ppm loin de la mediane mais une intensite
                         # coherente avec ses voisins (meme cas qu'a 315K).
OUTPUT_NAME = r"D:\Postdoc\Figures\T2_global_statique_345K_provisoire"
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=False, read_phase_from_procs=False, reference_shift_ppm=0.0):
    dic, data = ng.bruker.read(path, read_procs=read_phase_from_procs)

    grpdly_shift = find_grpdly_shift(dic)
    if grpdly_shift > 0:
        print(f"  GRPDLY corrected: shifted by {grpdly_shift} points")
    else:
        print("  GRPDLY not found or zero — no correction applied")

    N = data.shape[0]
    dt = 1 / dic["acqus"]["SW_h"]
    data_zf = np.concatenate([data, np.zeros(zf_factor * N, dtype=complex)])

    ph0_deg = ph0_manual
    ph1_deg = ph1
    if read_phase_from_procs:
        try:
            ph0_deg = dic["procs"]["PHC0"]
            ph1_deg = dic["procs"]["PHC1"]
        except Exception:
            print("  Could not read phase from procs — using manual/auto value instead")
    elif auto_ph0:
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1_deg), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1_deg))

    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1_deg), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm

    te = dic["acqus"].get("TE", None)
    if te is not None:
        print(f"  TE = {te:.2f} K")

    rg = dic["acqus"].get("RG", None)
    return delta, spectrum, dic, ph0_deg, rg


def get_peak_intensity_magnitude(delta, spectrum, ppm_window):
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    window = np.abs(spectrum[mask])
    idx = int(np.argmax(window))
    return float(window[idx]), float(delta[mask][idx])


def monoexp_decay(t, M0, T2):
    return M0 * np.exp(-t / T2)


def get_intensity(path):
    delta, spectrum, dic, _, rg = process_1d_spectrum(
        path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM,
    )
    intensity, peak_ppm = get_peak_intensity_magnitude(delta, spectrum, PEAK_PPM_WINDOW)
    return intensity, peak_ppm, rg


if __name__ == "__main__":
    print("=== T2 series (static probe, 345K, grille allegee 2-64) ===")
    records = []
    for l0, (path, ns) in sorted(DATASETS.items()):
        intensity, peak_ppm, rg = get_intensity(path)
        intensity_per_scan = intensity / ns
        decho_us = l0 * 10
        print(f"  L0={l0:>4}  decho={decho_us:>4}us  NS={ns:>4}  RG={rg}  peak at {peak_ppm:7.2f} ppm"
              f"  raw={intensity:.4e}  per-scan={intensity_per_scan:.4e}")
        records.append((l0, decho_us, intensity_per_scan, peak_ppm, rg))

    rg_values = set(r[4] for r in records)
    if len(rg_values) > 1:
        print(f"\n  WARNING: RG is not constant across the series ({rg_values}) -- "
              f"cross-calibration needed before comparing amplitudes.")
    else:
        print(f"\n  RG constant across the series ({rg_values.pop()}) -- no cross-calibration needed.")

    all_ppm = [r[3] for r in records if r[0] not in NOISE_FLOOR_L0]
    median_ppm = float(np.median(all_ppm))
    is_ppm_outlier = lambda r: abs(r[3] - median_ppm) > PPM_OUTLIER_THRESHOLD and r[0] not in FORCE_INCLUDE_L0
    is_noise_floor = lambda r: r[0] in NOISE_FLOOR_L0

    ppm_bad = [r for r in records if is_ppm_outlier(r) and not is_noise_floor(r)]
    floor_excl = [r for r in records if is_noise_floor(r)]
    good = [r for r in records if not is_ppm_outlier(r) and not is_noise_floor(r)]

    if ppm_bad:
        print(f"\n  EXCLUDING {len(ppm_bad)} point(s) from the fit -- peak far from the median"
              f" ({median_ppm:.2f} ppm), likely mispicked noise:")
        for l0, decho_us, _, ppm, _ in ppm_bad:
            print(f"    L0={l0} (decho={decho_us}us): peak at {ppm:.2f} ppm"
                  f" ({abs(ppm-median_ppm):.1f} ppm from median) -- EXCLUDED (ppm)")
        print("  -- verifie l'intensite brute vs voisins avant de rajouter via FORCE_INCLUDE_L0.")
    else:
        print(f"\n  All signal-region peak positions within {PPM_OUTLIER_THRESHOLD} ppm of the median"
              f" ({median_ppm:.2f} ppm) -- no mispicked-peak red flag.")

    print(f"\n  EXCLUDING {len(floor_excl)} point(s) from the fit -- presumed indistinguishable from the"
          f" magnitude noise floor (A VERIFIER sur cette serie) :")
    for l0, decho_us, i_ps, ppm, _ in sorted(floor_excl):
        print(f"    L0={l0} (decho={decho_us}us): per-scan={i_ps:.4e} ppm={ppm:.2f} -- EXCLUDED (floor, a verifier)")

    decho_s = np.array([r[1] * 1e-6 for r in good])
    L0_arr = np.array([r[0] for r in good])
    I = np.array([r[2] for r in good])
    print(f"\n  Fitting with {len(I)}/{len(records)} points (signal-only region, L0 <= {max(L0_arr)}).")

    diffs = np.diff(I)
    if np.any(diffs > 0):
        rises = [(L0_arr[i], L0_arr[i+1]) for i in range(len(diffs)) if diffs[i] > 0]
        print(f"\n  WARNING: intensity increases somewhere between these consecutive L0 pairs: {rises}"
              f" -- physically shouldn't happen for a decay curve, worth a second look.")
    else:
        print("\n  Intensity is monotonically non-increasing with decho -- consistent with a clean decay curve.")

    print("\n--- Monoexponential decay fit (sigma=I weighted, region signal uniquement) ---")
    p0 = [1.1 * I.max(), 1e-4]
    bounds = ([0.5 * I.max(), 1e-6], [5 * I.max(), 1e-2])
    popt, pcov = curve_fit(monoexp_decay, decho_s, I, p0=p0, sigma=I, maxfev=50000, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    M0, T2 = popt
    print(f"  T2 = {T2*1e6:.3g} +/- {perr[1]*1e6:.2g} us")

    print("\n--- Robustness (leave-one-out on the shortest/longest points) ---")
    for label, mask in [("tous", np.ones_like(L0_arr, dtype=bool)),
                         ("sans le point le plus court", L0_arr != L0_arr.min()),
                         ("sans le point le plus long", L0_arr != L0_arr.max())]:
        try:
            popt_i, pcov_i = curve_fit(monoexp_decay, decho_s[mask], I[mask], p0=p0, sigma=I[mask],
                                        maxfev=50000, bounds=bounds)
            print(f"  {label:<28} T2={popt_i[1]*1e6:.3g}us")
        except RuntimeError:
            print(f"  {label:<28} fit failed")

    resid = 100 * (I - monoexp_decay(decho_s, *popt)) / I
    print("\nMonoexponential relative residuals (%):", np.round(resid, 2))
    print("  -- si un motif systematique apparait (S croissant/decroissant plutot que dispersion"
          "     aleatoire autour de 0), c'est le signe qu'un ou plusieurs points signal sont deja")
    print("     contamines par le plancher -- resserre NOISE_FLOOR_L0 (cf. l'essai rate a 315K le 02/09).")

    df = pd.DataFrame({"L0": L0_arr, "decho_us": L0_arr * 10, "Intensity_per_scan": I})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    t_fit = np.logspace(np.log10(decho_s.min() / 2), np.log10(decho_s.max() * 1.3), 400)
    y_fit = monoexp_decay(t_fit, *popt)
    fit_legend = f"monoexp fit: T2={T2*1e6:.1f}+/-{perr[1]*1e6:.1f}us (signal region only, L0<={max(L0_arr)})"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(decho_s * 1e6, I, color="blue", s=55, zorder=3, label=f"data (fit region, L0<={max(L0_arr)})")
    ax.plot(t_fit * 1e6, y_fit, color="red", lw=1.5, zorder=2, label="monoexponential fit")
    if floor_excl:
        px = np.array([r[1] for r in floor_excl])
        py = np.array([r[2] for r in floor_excl])
        ax.scatter(px, py, color="gray", marker="x", s=60, zorder=3,
                   label="plancher de bruit presume (hors fit)")
    ax.set_xscale("log")
    ax.set_xlabel("Echo delay decho (us)")
    ax.set_ylabel("Intensity per scan (a.u., magnitude)")
    ax.set_title(r"$^7$Li T$_2$ decay — static probe, 345K")
    ax.text(0.97, 0.95, fit_legend, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="lower left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    agr_series = [
        dict(x=decho_s * 1e6, y=I, mode="symbol", color="blue", legend="data (per-scan)"),
        dict(x=t_fit * 1e6, y=y_fit, mode="line", color="red", legend=fit_legend),
    ]
    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Echo delay decho (us)", ylabel="Intensity per scan (a.u.)",
               xlog=True, title="7Li T2 decay -- static probe, 345K")

    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")
