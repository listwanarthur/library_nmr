import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T1_global_statique_315K -- campagne du 01-02/09 (LLZO-400-sep26)
# Meme pipeline que T1_recovery_static_330K.py, deux series de 19 points
# (D1 jusqu'a 250s) pour verifier la reproductibilite.
#
# A VERIFIER avant d'utiliser ces donnees :
#  - RG=216.57 constant sur les 38 exp -- c'est la valeur qualifiee
#    d'"obsolete" au 330K (cf. T1_recovery_static_330K.py, rga refait ->
#    136.56). Si le rga n'a pas ete relance ici, RG peut etre herite et
#    fausser une comparaison ABSOLUE avec les autres temperatures (la
#    comparaison relative intra-serie reste valide, meme RG partout).
#  - Grille D1 SANS points de fermeture de plateau (400/500s, presents
#    au 298K/330K) -- le script imprime le % de plateau atteint a D1=250s.
# ============================================================

# === CONFIGURATION -- only section to edit ===
# SERIES : "1" (exp200-218) ou "2" (exp219-237), deux passes de la meme nuit
SERIES = "1"

_D1_LABELS = [0.02, 0.05, 0.2, 0.35, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8,
              15, 30, 60, 120, 150, 250]
_NS_BY_D1 = {0.02: 128, 0.05: 128, 0.2: 128, 0.35: 128, 0.5: 128,
             0.7: 32, 1: 32, 1.5: 32, 2: 32, 3: 32, 4: 32, 6: 32, 8: 32,
             15: 32, 30: 32, 60: 32, 120: 32, 150: 32, 250: 32}
_EXP_SERIE_1 = list(range(200, 219))
_EXP_SERIE_2 = list(range(219, 238))
_EXP_LIST = _EXP_SERIE_1 if SERIES == "1" else _EXP_SERIE_2

# D1 (s): (path, NS)
DATASETS = {
    d1: (rf"D:\Postdoc\Datas\LLZO-400-sep26\{exp}", _NS_BY_D1[d1])
    for d1, exp in zip(_D1_LABELS, _EXP_LIST)
}

LB = 10               # line broadening in Hz -- meme valeur que 298K/330K
PH0_MANUAL = 0.0       # irrelevant in magnitude mode
PH1 = 0.0              # irrelevant in magnitude mode
AUTO_PH0 = False       # do not turn on -- unstable on this static line (documented)
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2      # meme referencement que 298K/330K
ZF_FACTOR = 1
# A VERIFIER : fenetre reprise de 330K (raie retrecie) -- 315K est plus
# proche de 298K (raie large) que de 330K, elargir vers (100,-100) si la
# raie deborde de (-10,15) sur le spectre a D1=250s.
PEAK_PPM_WINDOW = (-10, 15)
PPM_OUTLIER_THRESHOLD = 5.0   # ppm from the median -- flags a likely mispicked noise peak
FORCE_INCLUDE_D1 = [0.35, 0.7, 1, 2, 3, 4, 6, 8, 15, 30, 60, 120, 150, 250]
# Union des mispicks des deux series (pas les memes D1 d'une serie a
# l'autre) -- intensite coherente avec la tendance monotone a chaque
# fois, donc vrais points juste mal repere en position (raie large
# statique, argmax instable). Union utilisee pour que la meme liste
# fonctionne quelle que soit SERIES.
OUTPUT_NAME = rf"D:\Postdoc\Figures\T1_global_statique_315K_provisoire_serie{SERIES}"
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
    if rg is not None:
        print(f"  RG = {rg:.2f}")

    aq = dic["acqus"]["TD"] / (2 * dic["acqus"]["SW_h"])

    return delta, spectrum, dic, ph0_deg, aq


def get_peak_intensity_magnitude(delta, spectrum, ppm_window):
    lo, hi = sorted(ppm_window)
    mask = (delta >= lo) & (delta <= hi)
    window = np.abs(spectrum[mask])
    idx = int(np.argmax(window))
    return float(window[idx]), float(delta[mask][idx])


def biexp_recovery(t, M0, f, T1a, T1b):
    return M0 * (1 - f * np.exp(-t / T1a) - (1 - f) * np.exp(-t / T1b))


def triexp_recovery(t, M0, f1, f2, T1a, T1b, T1c):
    return M0 * (1 - f1 * np.exp(-t / T1a) - f2 * np.exp(-t / T1b) - (1 - f1 - f2) * np.exp(-t / T1c))


def get_intensity(path):
    delta, spectrum, _, _, aq = process_1d_spectrum(
        path, LB, PH0_MANUAL, PH1, ZF_FACTOR,
        auto_ph0=AUTO_PH0, read_phase_from_procs=READ_PHASE_FROM_PROCS,
        reference_shift_ppm=REFERENCE_SHIFT_PPM,
    )
    intensity, peak_ppm = get_peak_intensity_magnitude(delta, spectrum, PEAK_PPM_WINDOW)
    return intensity, peak_ppm, aq


if __name__ == "__main__":
    print(f"=== T1 series (static probe, 315K, serie {SERIES}) ===")
    records = []  # (d1, intensity_per_scan, peak_ppm, aq)
    for d1, (path, ns) in sorted(DATASETS.items()):
        intensity, peak_ppm, aq = get_intensity(path)
        intensity_per_scan = intensity / ns
        print(f"  D1={d1:>6.2f}s  NS={ns:>4}  AQ={aq*1000:.1f}ms  peak at {peak_ppm:7.2f} ppm"
              f"  raw={intensity:.4e}  per-scan={intensity_per_scan:.4e}")
        records.append((d1, intensity_per_scan, peak_ppm, aq))

    aq_values = np.array([r[3] for r in records])
    if np.ptp(aq_values) > 1e-6:
        print(f"\n  WARNING: AQ is not constant across the series ({aq_values.min()*1000:.2f}-"
              f"{aq_values.max()*1000:.2f} ms) -- check TD/SW_h per experiment before using a single AQ below.")
    AQ = float(np.mean(aq_values))
    print(f"\n  AQ = {AQ*1000:.2f} ms (from TD/(2*SW_h), constant across the series) -- "
          f"fitting against D1+AQ, not D1 alone.")

    all_ppm = [r[2] for r in records]
    median_ppm = float(np.median(all_ppm))
    is_outlier = lambda r: abs(r[2] - median_ppm) > PPM_OUTLIER_THRESHOLD and r[0] not in FORCE_INCLUDE_D1
    bad = [r for r in records if is_outlier(r)]
    good = [r for r in records if not is_outlier(r)]
    if bad:
        print(f"\n  EXCLUDING {len(bad)} point(s) from the fit -- peak far from the median"
              f" ({median_ppm:.2f} ppm), likely mispicked noise:")
        for d1, _, ppm, _ in bad:
            print(f"    D1={d1}s: peak at {ppm:.2f} ppm ({abs(ppm-median_ppm):.1f} ppm from median) -- EXCLUDED")
    else:
        print(f"\n  All peak positions within {PPM_OUTLIER_THRESHOLD} ppm of the median"
              f" ({median_ppm:.2f} ppm) -- no mispicked-peak red flag.")
    forced = [r for r in records if r[0] in FORCE_INCLUDE_D1 and abs(r[2] - median_ppm) > PPM_OUTLIER_THRESHOLD]
    for d1, _, ppm, _ in forced:
        print(f"    D1={d1}s: peak at {ppm:.2f} ppm ({abs(ppm-median_ppm):.1f} ppm from median)"
              f" -- FORCE-INCLUDED")

    D1 = np.array([r[0] for r in good])
    I = np.array([r[1] for r in good])
    D1_eff = D1 + AQ
    print(f"  Fitting with {len(D1)}/{len(records)} points (D1+AQ used as the recovery time).")

    diffs = np.diff(I)
    if np.any(diffs < 0):
        drops = [(D1[i], D1[i+1]) for i in range(len(diffs)) if diffs[i] < 0]
        print(f"\n  WARNING: intensity decreases somewhere between these consecutive D1 pairs: {drops}"
              f" -- physically shouldn't happen for a recovery curve, worth a second look.")
    else:
        print("\n  Intensity is monotonically non-decreasing with D1 -- consistent with a clean recovery curve.")

    print("\n--- Biexponential fit (sigma=I weighted, t=D1+AQ) ---")
    p0_bi = [1.1 * I.max(), 0.1, 0.05, 20]
    bounds_bi = ([0.5 * I.max(), 0, 0.0001, 1], [5 * I.max(), 1, 5, 500])
    popt_bi, pcov_bi = curve_fit(biexp_recovery, D1_eff, I, p0=p0_bi, sigma=I, maxfev=50000, bounds=bounds_bi)
    perr_bi = np.sqrt(np.diag(pcov_bi))
    M0_bi, f_bi, T1fast_bi, T1slow_bi = popt_bi
    print(f"  T1_slow = {T1slow_bi:.3g} +/- {perr_bi[3]:.2g} s  ({(1-f_bi)*100:.1f}%)")
    print(f"  T1_fast = {T1fast_bi:.3g} +/- {perr_bi[2]:.2g} s  ({f_bi*100:.1f}%)")

    print("\n--- Biexponential fit robustness (leave-one-out / reweighting, t=D1+AQ) ---")
    for label, mask in [("tous", np.ones_like(D1, dtype=bool)),
                         ("sans D1=250", D1 != 250),
                         ("sans D1=150,250", ~np.isin(D1, [150, 250])),
                         ("sans D1=120,150,250", ~np.isin(D1, [120, 150, 250]))]:
        try:
            popt_i, pcov_i = curve_fit(biexp_recovery, D1_eff[mask], I[mask], p0=p0_bi, sigma=I[mask],
                                        maxfev=50000, bounds=bounds_bi)
            print(f"  {label:<20} T1_slow = {popt_i[3]:.3g} +/- {np.sqrt(np.diag(pcov_i))[3]:.2g} s")
        except RuntimeError:
            print(f"  {label:<20} fit failed")
    for label, sigma in [("non pondere", None), ("sigma=sqrt(I)", np.sqrt(I))]:
        try:
            popt_i, pcov_i = curve_fit(biexp_recovery, D1_eff, I, p0=p0_bi, sigma=sigma, maxfev=50000,
                                        bounds=bounds_bi)
            print(f"  {label:<20} T1_slow = {popt_i[3]:.3g} +/- {np.sqrt(np.diag(pcov_i))[3]:.2g} s")
        except RuntimeError:
            print(f"  {label:<20} fit failed")

    print("\n--- Triexponential fit (sigma=I weighted, t=D1+AQ) ---")
    use_tri = False
    try:
        p0_tri = [1.1 * I.max(), 0.05, 0.05, 0.02, 2, 25]
        bounds_tri = ([0.5 * I.max(), 0, 0, 0.0001, 0.01, 1], [5 * I.max(), 0.5, 0.5, 1, 20, 500])
        popt_tri, pcov_tri = curve_fit(triexp_recovery, D1_eff, I, p0=p0_tri, sigma=I, maxfev=50000, bounds=bounds_tri)
        perr_tri = np.sqrt(np.diag(pcov_tri))
        M0_tri, f1_tri, f2_tri, T1a_tri, T1b_tri, T1c_tri = popt_tri
        print(f"  T1a = {T1a_tri:.3g} +/- {perr_tri[3]:.2g} s  ({f1_tri*100:.1f}%)")
        print(f"  T1b = {T1b_tri:.3g} +/- {perr_tri[4]:.2g} s  ({f2_tri*100:.1f}%)")
        print(f"  T1c = {T1c_tri:.3g} +/- {perr_tri[5]:.2g} s  ({(1-f1_tri-f2_tri)*100:.1f}%)")
        use_tri = True
    except RuntimeError as e:
        print(f"  Triexponential fit failed ({e}) -- keep biexponential.")

    if use_tri:
        print("\n--- Triexponential fit robustness (leave-one-out / reweighting, t=D1+AQ) ---")
        for label, mask in [("tous", np.ones_like(D1, dtype=bool)),
                             ("sans D1=250", D1 != 250),
                             ("sans D1=150,250", ~np.isin(D1, [150, 250])),
                             ("sans D1=120,150,250", ~np.isin(D1, [120, 150, 250]))]:
            try:
                popt_i, pcov_i = curve_fit(triexp_recovery, D1_eff[mask], I[mask], p0=p0_tri, sigma=I[mask],
                                            maxfev=50000, bounds=bounds_tri)
                print(f"  {label:<20} T1c = {popt_i[5]:.3g} +/- {np.sqrt(np.diag(pcov_i))[5]:.2g} s")
            except RuntimeError:
                print(f"  {label:<20} fit failed")
        for label, sigma in [("non pondere", None), ("sigma=sqrt(I)", np.sqrt(I))]:
            try:
                popt_i, pcov_i = curve_fit(triexp_recovery, D1_eff, I, p0=p0_tri, sigma=sigma, maxfev=50000,
                                            bounds=bounds_tri)
                print(f"  {label:<20} T1c = {popt_i[5]:.3g} +/- {np.sqrt(np.diag(pcov_i))[5]:.2g} s")
            except RuntimeError:
                print(f"  {label:<20} fit failed")

        resid_tri = 100 * (I - triexp_recovery(D1_eff, *popt_tri)) / I
        print("\nTriexponential (variant A, free) relative residuals (%):", np.round(resid_tri, 2))

    # Variant B: T1_fast contraint a la gamme biexp connue (~0.3-5s) pour
    # forcer la 3e composante a tester une queue LENTE (>30s) plutot
    # qu'une composante ultra-rapide non contrainte par les donnees
    # (cf. le T1a quasi degenere -- incertitude formelle proche de zero,
    # signature classique de parametre fantome -- du variant A ci-dessus).
    print("\n--- Triexponential fit, variant B (T1_fast contraint ~biexp, teste"
          " une vraie queue lente >30s) ---")
    use_tri_b = False
    try:
        p0_tri_b = [1.1 * I.max(), 0.14, 0.02, 1.4, 15, 80]
        bounds_tri_b = ([0.5 * I.max(), 0, 0, 0.3, 5, 30], [5 * I.max(), 0.4, 0.3, 5, 30, 1000])
        popt_tri_b, pcov_tri_b = curve_fit(triexp_recovery, D1_eff, I, p0=p0_tri_b, sigma=I,
                                            maxfev=50000, bounds=bounds_tri_b)
        perr_tri_b = np.sqrt(np.diag(pcov_tri_b))
        M0_b, f1_b, f2_b, T1a_b, T1b_b, T1c_b = popt_tri_b
        print(f"  T1_fast = {T1a_b:.3g} +/- {perr_tri_b[3]:.2g} s  ({f1_b*100:.1f}%)")
        print(f"  T1_mid  = {T1b_b:.3g} +/- {perr_tri_b[4]:.2g} s  ({f2_b*100:.1f}%)")
        print(f"  T1_slow = {T1c_b:.3g} +/- {perr_tri_b[5]:.2g} s  ({(1-f1_b-f2_b)*100:.1f}%)")
        resid_tri_b = 100 * (I - triexp_recovery(D1_eff, *popt_tri_b)) / I
        print("  relative residuals (%):", np.round(resid_tri_b, 2))
        for label, mask in [("tous", np.ones_like(D1, dtype=bool)),
                             ("sans D1=250", D1 != 250),
                             ("sans D1=150,250", ~np.isin(D1, [150, 250]))]:
            try:
                popt_i, pcov_i = curve_fit(triexp_recovery, D1_eff[mask], I[mask], p0=p0_tri_b, sigma=I[mask],
                                            maxfev=50000, bounds=bounds_tri_b)
                print(f"    {label:<18} T1_slow = {popt_i[5]:.3g} +/- {np.sqrt(np.diag(pcov_i))[5]:.2g} s")
            except RuntimeError:
                print(f"    {label:<18} fit failed")
        use_tri_b = True
    except RuntimeError as e:
        print(f"  Variant B fit failed ({e}).")

    USE_TRIEXP_FOR_FIGURE = True   # False -> revient au biexp "officiel"

    resid_bi = 100 * (I - biexp_recovery(D1_eff, *popt_bi)) / I
    print("\nBiexponential relative residuals (%):", np.round(resid_bi, 2))

    # Fraction du plateau de recuperation atteinte a D1max -- si < 95%,
    # des points longs (400/500s) manquent probablement pour fermer le plateau.
    fraction_plateau = 1 - f_bi * np.exp(-(D1.max() + AQ) / T1fast_bi) - (1 - f_bi) * np.exp(-(D1.max() + AQ) / T1slow_bi)
    print(f"\nA D1={D1.max():.0f}s, on est a {fraction_plateau*100:.1f}% du plateau de recuperation "
          f"(biexp) -- {'OK' if fraction_plateau >= 0.95 else 'INSUFFISANT, points longs (400/500s) probablement necessaires'}.")

    df = pd.DataFrame({"D1_s": D1, "AQ_s": AQ, "D1_eff_s": D1_eff, "Intensity_per_scan": I})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    TRIEXP_VARIANT_FOR_FIGURE = "B"   # "A" (libre) ou "B" (T1_fast contraint, teste une queue lente)

    t_fit = np.logspace(np.log10(D1.min() / 2), np.log10(D1.max() * 1.3), 400)
    t_fit_eff = t_fit + AQ
    if USE_TRIEXP_FOR_FIGURE and TRIEXP_VARIANT_FOR_FIGURE == "B" and use_tri_b:
        y_fit = triexp_recovery(t_fit_eff, *popt_tri_b)
        fit_legend = (f"tri-exp fit (var. B): T1slow={T1c_b:.1f}+/-{perr_tri_b[5]:.1f}s "
                      f"({(1 - f1_b - f2_b) * 100:.1f}%), "
                      f"T1mid={T1b_b:.2f}+/-{perr_tri_b[4]:.2f}s ({f2_b*100:.1f}%), "
                      f"T1fast={T1a_b:.3f}+/-{perr_tri_b[3]:.3f}s ({f1_b*100:.1f}%)")
        title_suffix = "triexponential, variant B (T1_fast constrained)"
    elif USE_TRIEXP_FOR_FIGURE and use_tri:
        y_fit = triexp_recovery(t_fit_eff, *popt_tri)
        fit_legend = (f"tri-exp fit (var. A): T1c={T1c_tri:.1f}+/-{perr_tri[5]:.1f}s ({(1-f1_tri-f2_tri)*100:.1f}%), "
                      f"T1b={T1b_tri:.2f}+/-{perr_tri[4]:.2f}s ({f2_tri*100:.1f}%), "
                      f"T1a={T1a_tri*1000:.2f}+/-{perr_tri[3]*1000:.2f}ms ({f1_tri*100:.1f}%)")
        title_suffix = "triexponential, variant A (free)"
    else:
        y_fit = biexp_recovery(t_fit_eff, *popt_bi)
        fit_legend = (f"biexp fit: T1s={T1slow_bi:.1f}+/-{perr_bi[3]:.1f}s ({(1-f_bi)*100:.1f}%), "
                      f"T1f={T1fast_bi*1000:.1f}+/-{perr_bi[2]*1000:.1f}ms ({f_bi*100:.1f}%)")
        title_suffix = "biexponential"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(D1, I, color="blue", s=55, zorder=3, label="data (per-scan, RG=216.57)")
    ax.plot(t_fit, y_fit, color="red", lw=1.5, zorder=2, label=f"{title_suffix} fit")
    ax.set_xscale("log")
    ax.set_xlabel("Recovery delay D1 (s)")
    ax.set_ylabel("Intensity per scan (a.u., magnitude)")
    ax.set_title(rf"$^7$Li T$_1$ recovery — static probe, 315K (01-02/09, serie {SERIES}, RG=216.57)")
    ax.text(0.97, 0.05, fit_legend, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    agr_series = [
        dict(x=D1, y=I, mode="symbol", color="blue", legend="data (per-scan, RG=216.57)"),
        dict(x=t_fit, y=y_fit, mode="line", color="red", legend=fit_legend),
    ]
    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Recovery delay D1 (s)", ylabel="Intensity per scan (a.u.)",
               xlog=True, title="7Li T1 recovery -- static probe, 315K")
