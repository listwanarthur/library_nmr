import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T2_global_statique_330K -- grille fine (26/08), L0=2,3,4,5,6,8,9,11,13,
# 16,19,22,32,45 (decho=20-450us), NS=64, DS=6, D1=15s.
# PLATEAU_CHECK : L0=64,256 (decho=640us,2.56ms), NS=220.
#
# Point cle (mode magnitude) : le plancher de bruit est ADDITIF au signal
# reel, pas une extrapolation vers zero de la decroissance ajustee. Le
# cross-check NS=64 vs NS=220 (Bloc B, exp541/542, meme diagnostic qu'a
# 360K) donne un ecart nettement positif aux deux L0, de signe oppose et
# bien plus grand que celui vu au T1 du meme palier (effet de lot/session)
# -- signature d'un plancher de bruit magnitude (~1/sqrt(NS)), pas un vrai
# signal qui decroit. NB : la position ppm du "pic" est quasi aleatoire a
# ce niveau (argmax instable sur du bruit pur) -- juger sur l'intensite et
# sa coherence entre points voisins, pas sur le ppm seul.
#
# DECISION : fit MONO-exponentiel restreint a la region signal reel
# (L0 <= 22, ~20-220us). L0=32,45 (deja dans DATASETS), PLATEAU_CHECK et
# NS_CROSSCHECK restent affiches pour reference mais hors fit.
# ============================================================

# === CONFIGURATION -- only section to edit ===
DATASETS = {
    2:   (r"D:\Postdoc\Datas\LLZO-400-aug26\514", 64),
    3:   (r"D:\Postdoc\Datas\LLZO-400-aug26\515", 64),
    4:   (r"D:\Postdoc\Datas\LLZO-400-aug26\516", 64),
    5:   (r"D:\Postdoc\Datas\LLZO-400-aug26\532", 64),
    6:   (r"D:\Postdoc\Datas\LLZO-400-aug26\517", 64),
    8:   (r"D:\Postdoc\Datas\LLZO-400-aug26\518", 64),
    9:   (r"D:\Postdoc\Datas\LLZO-400-aug26\533", 64),
    11:  (r"D:\Postdoc\Datas\LLZO-400-aug26\519", 64),
    13:  (r"D:\Postdoc\Datas\LLZO-400-aug26\534", 64),
    16:  (r"D:\Postdoc\Datas\LLZO-400-aug26\520", 64),
    19:  (r"D:\Postdoc\Datas\LLZO-400-aug26\535", 64),
    22:  (r"D:\Postdoc\Datas\LLZO-400-aug26\521", 64),
    32:  (r"D:\Postdoc\Datas\LLZO-400-aug26\522", 64),
    45:  (r"D:\Postdoc\Datas\LLZO-400-aug26\523", 64),
}

# Indiscernables du plancher de bruit (cf. diagnostic ci-dessus) -- gardes
# dans DATASETS pour affichage/audit, exclus du fit.
NOISE_FLOOR_L0 = {32, 45}

# Points de verification plateau bruit -- traces/imprimes a part, PAS dans le fit
PLATEAU_CHECK = {
    64:  (r"D:\Postdoc\Datas\LLZO-400-aug26\524", 220),
    256: (r"D:\Postdoc\Datas\LLZO-400-aug26\525", 220),
}

# Cross-check NS (27/08, Bloc B) -- memes L0 que PLATEAU_CHECK, NS=64 au
# lieu de 220.
NS_CROSSCHECK = {
    64:  (r"D:\Postdoc\Datas\LLZO-400-aug26\541", 64),
    256: (r"D:\Postdoc\Datas\LLZO-400-aug26\542", 64),
}

LB = 10
PH0_MANUAL = 0.0       # irrelevant en mode magnitude
PH1 = 0.0              # irrelevant en mode magnitude
AUTO_PH0 = False
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2
ZF_FACTOR = 1
PEAK_PPM_WINDOW = (-10, 15)
PPM_OUTLIER_THRESHOLD = 5.0
FORCE_INCLUDE_L0 = [2, 3, 5, 8, 16, 19, 32]  # raie large/plate -- intensite
                         # brute monotone decroissante malgre les ppm disperses.
OUTPUT_NAME = r"D:\Postdoc\Figures\T2_global_statique_330K_provisoire"
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
    # === STEP 1: process the fine grid, normalize by NS, sanity-check peak position/RG ===
    print("=== T2 series (static probe, 330K palier, grille fine) ===")
    records = []  # (L0, decho_us, intensity_per_scan, peak_ppm, rg)
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

    print(f"\n  EXCLUDING {len(floor_excl)} point(s) from the fit -- indistinguishable from the"
          f" magnitude noise floor (see NS cross-check further down for the quantitative check):")
    for l0, decho_us, i_ps, ppm, _ in sorted(floor_excl):
        print(f"    L0={l0} (decho={decho_us}us): per-scan={i_ps:.4e} ppm={ppm:.2f} -- EXCLUDED (floor)")

    decho_s = np.array([r[1] * 1e-6 for r in good])   # decho in seconds for the fit
    L0_arr = np.array([r[0] for r in good])
    I = np.array([r[2] for r in good])
    print(f"\n  Fitting with {len(I)}/{len(records)} points from the fine grid"
          f" (signal-only region, L0 <= {max(L0_arr)}).")

    diffs = np.diff(I)
    if np.any(diffs > 0):
        rises = [(L0_arr[i], L0_arr[i+1]) for i in range(len(diffs)) if diffs[i] > 0]
        print(f"\n  WARNING: intensity increases somewhere between these consecutive L0 pairs: {rises}"
              f" -- physically shouldn't happen for a decay curve, worth a second look.")
    else:
        print("\n  Intensity is monotonically non-increasing with decho -- consistent with a clean decay curve.")

    # === STEP 2: plateau-bruit check (printed/plotted only, NOT fitted) ===
    print("\n--- Vérification plateau bruit (hors fit) ---")
    plateau_records = []
    for l0, (path, ns) in sorted(PLATEAU_CHECK.items()):
        intensity, peak_ppm, rg = get_intensity(path)
        intensity_per_scan = intensity / ns
        decho_us = l0 * 10
        print(f"  L0={l0:>4}  decho={decho_us/1000:.2f}ms  NS={ns:>4}  RG={rg}  peak at {peak_ppm:7.2f} ppm"
              f"  per-scan={intensity_per_scan:.4e}")
        plateau_records.append((l0, decho_us, intensity_per_scan, peak_ppm))

    # === STEP 2.5: NS cross-check (meme L0, NS different) ===
    print("\n--- NS cross-check (meme L0, NS different) ---")
    plateau_by_l0 = {r[0]: r for r in plateau_records}
    for l0_check, (path, ns_check) in sorted(NS_CROSSCHECK.items()):
        intensity, peak_ppm, rg = get_intensity(path)
        intensity_per_scan = intensity / ns_check
        print(f"  L0={l0_check:>4}  NS={ns_check:>4}  peak at {peak_ppm:7.2f} ppm"
              f"  per-scan={intensity_per_scan:.4e}")
        ref = plateau_by_l0.get(l0_check)
        if ref is not None:
            ref_per_scan = ref[2]
            rel_diff = 100 * (intensity_per_scan - ref_per_scan) / ref_per_scan
            print(f"    vs L0={l0_check} NS={PLATEAU_CHECK[l0_check][1]} deja dans PLATEAU_CHECK "
                  f"(per-scan={ref_per_scan:.4e}) : ecart = {rel_diff:+.1f}%")
        else:
            print(f"    (pas de point de reference trouve dans PLATEAU_CHECK pour L0={l0_check})")
    print(
        "  Diagnostic retenu (28/08) : cet ecart (NS=64 > NS=220), de signe oppose et bien plus\n"
        "  grand que celui vu au T1 du meme palier (Bloc A, effet de lot/session -- voir\n"
        "  Check_lot_reproducibility.py), est la signature d'un plancher de bruit magnitude\n"
        "  (moyenne ~1/sqrt(NS), pas un vrai\n"
        "  signal qui decroit) -- meme si l'amplitude exacte de l'ecart est bruitee ici par\n"
        "  l'instabilite de l'argmax ppm sur un signal quasi nul (voir positions ppm ci-dessus,\n"
        "  tres dispersees). Ces points, comme L0>22 dans la grille fine, sont donc traites\n"
        "  comme du bruit, pas comme une composante T2 lente reelle."
    )

    # === STEP 3: fit -- MONOexponential decay, signal-only region (L0 <= 22) ===
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

    # === STEP 4: export ===
    df = pd.DataFrame({"L0": L0_arr, "decho_us": L0_arr * 10, "Intensity_per_scan": I})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    t_fit = np.logspace(np.log10(decho_s.min() / 2), np.log10(decho_s.max() * 1.3), 400)
    y_fit = monoexp_decay(t_fit, *popt)
    fit_legend = f"monoexp fit: T2={T2*1e6:.1f}+/-{perr[1]*1e6:.1f}us (signal region only, L0<=22)"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(decho_s * 1e6, I, color="blue", s=55, zorder=3, label="data (fit region, L0<=22)")
    ax.plot(t_fit * 1e6, y_fit, color="red", lw=1.5, zorder=2, label="monoexponential fit")
    if plateau_records:
        px = np.array([r[1] for r in plateau_records])
        py = np.array([r[2] for r in plateau_records])
        ax.scatter(px, py, color="gray", marker="x", s=60, zorder=3,
                   label="plateau bruit (hors fit)")
    ax.set_xscale("log")
    ax.set_xlabel("Echo delay decho (us)")
    ax.set_ylabel("Intensity per scan (a.u., magnitude)")
    ax.set_title(r"$^7$Li T$_2$ decay — static probe, 330K (RG=216.57)")
    ax.text(0.97, 0.95, fit_legend, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="lower left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    agr_series = [
        dict(x=decho_s * 1e6, y=I, mode="symbol", color="blue", legend="data (per-scan, RG=216.57)"),
        dict(x=t_fit * 1e6, y=y_fit, mode="line", color="red", legend=fit_legend),
    ]
    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Echo delay decho (us)", ylabel="Intensity per scan (a.u.)",
               xlog=True, title="7Li T2 decay -- static probe, 330K")

    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")
