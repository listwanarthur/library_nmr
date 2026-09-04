import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T2_global_statique_298K -- grille fine (exp352-361, 19-20/08).
# L0=2,3,4,6,8,11,16,22,32,45 (decho=L0*10us), NS=64, RG=107.86 constant
# sur toute la grille fine.
#
# PLATEAU_CHECK (exp362/363, meme RG=107.86) et NS_CROSSCHECK (exp800-803,
# RG=1290 different -- comparaison interne NS=64 vs NS=220 uniquement)
# servent a verifier NOISE_FLOOR_L0 ci-dessous (voir le print de l'etape
# 2.5 plus bas pour le detail du diagnostic quantitatif).
#
# Point cle (mode magnitude) : le plancher de bruit est ADDITIF au signal
# reel, pas une extrapolation vers zero de la decroissance ajustee. Pour
# juger si un point est plancher de bruit : comparer son intensite au
# plateau de bruit mesure (points profondement dans le bruit, quasi
# identiques entre eux) et verifier la stabilite de sa position ppm --
# jamais a une extrapolation zero-baseline du fit.
#
# NOISE_FLOOR_L0 = {32, 45} est une hypothese de depart (meme grille qu'a
# 330K, ou ces L0 etaient deja plancher de bruit ; T2 ici (~69us) plus
# court qu'a 330K (~105us), donc la contamination devrait apparaitre au
# moins aussi tot). Verifier les residus/monotonie imprimes a l'etape 1 :
# si L0=22 ressort anormal, il est probablement lui aussi plancher de
# bruit -- l'ajouter et relancer.
#
# exp388-390 (L0=50,55,60) existent mais sont a un RG different (94.34) et
# non calibrable -- volontairement exclus de DATASETS.
# ============================================================

# === CONFIGURATION -- only section to edit ===
# L0: (path, NS) -- decho = L0*10us (echosolid_p2.al)
DATASETS = {
    2:  (r"D:\Postdoc\Datas\LLZO-400-aug26\352", 64),
    3:  (r"D:\Postdoc\Datas\LLZO-400-aug26\353", 64),
    4:  (r"D:\Postdoc\Datas\LLZO-400-aug26\354", 64),
    6:  (r"D:\Postdoc\Datas\LLZO-400-aug26\355", 64),
    8:  (r"D:\Postdoc\Datas\LLZO-400-aug26\356", 64),
    11: (r"D:\Postdoc\Datas\LLZO-400-aug26\357", 64),
    16: (r"D:\Postdoc\Datas\LLZO-400-aug26\358", 64),
    22: (r"D:\Postdoc\Datas\LLZO-400-aug26\359", 64),
    32: (r"D:\Postdoc\Datas\LLZO-400-aug26\360", 64),
    45: (r"D:\Postdoc\Datas\LLZO-400-aug26\361", 64),
}

NOISE_FLOOR_L0 = {32, 45}  # hypothese de depart -- voir commentaire en tete de fichier

# Plateau-check, MEME RG que la grille fine (107.86) -- comparable
# directement, affiche/imprime a part, PAS dans le fit.
PLATEAU_CHECK = {
    64:  (r"D:\Postdoc\Datas\LLZO-400-aug26\362", 160),
    256: (r"D:\Postdoc\Datas\LLZO-400-aug26\363", 160),
}

# NS cross-check (28/08) -- RG=1290, DIFFERENT de tout le reste de ce
# fichier. Comparaison NS=64 vs NS=220 faite EN INTERNE a ce lot, jamais
# contre PLATEAU_CHECK (RG different, pas de facteur de correction connu).
NS_CROSSCHECK = {
    64:  {64: (r"D:\Postdoc\Datas\LLZO-400-aug26\800", 64),
          220: (r"D:\Postdoc\Datas\LLZO-400-aug26\802", 220)},
    256: {64: (r"D:\Postdoc\Datas\LLZO-400-aug26\801", 64),
          220: (r"D:\Postdoc\Datas\LLZO-400-aug26\803", 220)},
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
FORCE_INCLUDE_L0 = [2, 11, 16]   # a completer si un point echoue le controle ppm
                         # mais que l'intensite reste coherente avec ses
                         # voisins (verifier la monotonie brute avant de
                         # forcer, comme aux autres paliers)
OUTPUT_NAME = r"D:\Postdoc\Figures\T2_global_statique_298K_provisoire"
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


def get_mean_intensity_magnitude(delta, spectrum, ppm_window):
    # Plus robuste que l'argmax en regime quasi-bruit (raie noyee dans le
    # plancher, position du max devient quasi aleatoire) -- c'est cette
    # metrique qui a donne un resultat propre et coherent sur le cross-check
    # NS dedie (exp800-803, voir Check_T2_298K_NS_crosscheck.py).
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    window = np.abs(spectrum[mask])
    return float(window.mean())


def monoexp_decay(t, M0, T2):
    return M0 * np.exp(-t / T2)


def get_intensity(path):
    delta, spectrum, dic, _, rg = process_1d_spectrum(
        path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM,
    )
    intensity, peak_ppm = get_peak_intensity_magnitude(delta, spectrum, PEAK_PPM_WINDOW)
    mean_intensity = get_mean_intensity_magnitude(delta, spectrum, PEAK_PPM_WINDOW)
    return intensity, peak_ppm, rg, mean_intensity


if __name__ == "__main__":
    # === STEP 1: process the fine grid, normalize by NS, sanity-check peak position/RG ===
    print("=== T2 series (static probe, 298K palier, grille fine) ===")
    records = []  # (L0, decho_us, intensity_per_scan, peak_ppm, rg)
    for l0, (path, ns) in sorted(DATASETS.items()):
        intensity, peak_ppm, rg, _ = get_intensity(path)
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

    # === STEP 2: plateau-bruit check, MEME RG que la grille fine (printed/plotted only, NOT fitted) ===
    print("\n--- Verification plateau bruit, plus loin (meme RG=107.86, hors fit) ---")
    plateau_records = []
    for l0, (path, ns) in sorted(PLATEAU_CHECK.items()):
        intensity, peak_ppm, rg, _ = get_intensity(path)
        intensity_per_scan = intensity / ns
        decho_us = l0 * 10
        print(f"  L0={l0:>4}  decho={decho_us/1000:.2f}ms  NS={ns:>4}  RG={rg}  peak at {peak_ppm:7.2f} ppm"
              f"  per-scan={intensity_per_scan:.4e}")
        plateau_records.append((l0, decho_us, intensity_per_scan, peak_ppm))

    # === STEP 2.5: NS cross-check, RG=1290 (DIFFERENT du reste) -- comparaison EN INTERNE, NS=64 vs NS=220 ===
    print("\n--- NS cross-check (RG=1290, distinct de la grille fine -- comparaison interne uniquement) ---")
    theoretical_pct = 100 * ((220 / 64) ** 0.5 - 1)
    ns_crosscheck_records = []
    for l0_check, ns_dict in sorted(NS_CROSSCHECK.items()):
        vals = {}
        for ns_check, (path, ns_val) in sorted(ns_dict.items()):
            intensity, peak_ppm, rg, mean_intensity = get_intensity(path)
            i_ps_argmax = intensity / ns_val
            i_ps_mean = mean_intensity / ns_val
            vals[ns_check] = (i_ps_argmax, i_ps_mean, peak_ppm, rg)
            ns_crosscheck_records.append((l0_check, l0_check * 10, i_ps_mean, peak_ppm))
            print(f"  L0={l0_check:>4}  NS={ns_check:>4}  RG={rg}  peak at {peak_ppm:7.2f} ppm"
                  f"  per-scan(argmax)={i_ps_argmax:.4e}  per-scan(mean)={i_ps_mean:.4e}")
        if 64 in vals and 220 in vals:
            argmax_diff = 100 * (vals[64][0] - vals[220][0]) / vals[220][0]
            mean_diff = 100 * (vals[64][1] - vals[220][1]) / vals[220][1]
            print(f"    L0={l0_check}: NS=64 vs NS=220 -- argmax: {argmax_diff:+.1f}%  |"
                  f"  mean/integral: {mean_diff:+.1f}%  (theorie plancher de bruit pur: {theoretical_pct:+.1f}%)")
    print(
        "  Diagnostic retenu (28/08) : l'ecart mean/integral (NS=64 > NS=220), tres\n"
        "  proche de la prediction theorique plancher de bruit magnitude (voir\n"
        "  Check_T2_298K_NS_crosscheck.py pour les valeurs), confirme que le\n"
        "  'T2_lent' precedemment vu dans le fit biexponentiel de la grille fine est le meme artefact\n"
        "  deja resolu a 330K et 360K -- pas une vraie composante lente. NB : ces points sont a un RG\n"
        "  different (1290) de la grille fine (107.86) -- ecart non comparable directement a\n"
        "  PLATEAU_CHECK ci-dessus, cross-check interne (NS=64 vs NS=220 au meme RG) uniquement."
    )

    # === STEP 3: fit -- MONOexponential decay, signal-only region ===
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
    print("(Regarde ici en particulier L0=22 -- cf. commentaire en tete de fichier -- si son residu")
    print(" est nettement plus grand que ses voisins, c'est un signe qu'il faut l'ajouter a NOISE_FLOOR_L0.)")

    # === STEP 4: export ===
    df = pd.DataFrame({"L0": L0_arr, "decho_us": L0_arr * 10, "Intensity_per_scan": I})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    t_fit = np.logspace(np.log10(decho_s.min() / 2), np.log10(decho_s.max() * 1.3), 400)
    y_fit = monoexp_decay(t_fit, *popt)
    fit_legend = f"monoexp fit: T2={T2*1e6:.1f}+/-{perr[1]*1e6:.1f}us (signal region only)"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(decho_s * 1e6, I, color="blue", s=55, zorder=3, label="data (fit region)")
    ax.plot(t_fit * 1e6, y_fit, color="red", lw=1.5, zorder=2, label="monoexponential fit")
    floor_x, floor_y = [], []
    for l0, decho_us, i_ps, ppm, _ in floor_excl:
        floor_x.append(decho_us); floor_y.append(i_ps)
    for l0, decho_us, i_ps, ppm in plateau_records:
        floor_x.append(decho_us); floor_y.append(i_ps)
    if floor_x:
        ax.scatter(floor_x, floor_y, color="gray", marker="x", s=60, zorder=3,
                   label="plancher de bruit / hors fit (grille fine exclue + plateau-check)")
    ax.set_xscale("log")
    ax.set_xlabel("Echo delay decho (us)")
    ax.set_ylabel("Intensity per scan (a.u., magnitude)")
    ax.set_title(r"$^7$Li T$_2$ decay — static probe, 298K")
    ax.text(0.97, 0.95, fit_legend, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    agr_series = [
        dict(x=decho_s * 1e6, y=I, mode="symbol", color="blue", legend="data (fit region)"),
        dict(x=t_fit * 1e6, y=y_fit, mode="line", color="red", legend=fit_legend),
    ]
    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Echo delay decho (us)", ylabel="Intensity per scan (a.u.)",
               xlog=True, title="7Li T2 decay -- static probe, 298K")

    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")
