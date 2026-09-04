import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nmrglue as ng
from scipy.optimize import curve_fit

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0
from library_nmr.agr_export import export_agr

# ============================================================
# T1_global_statique_360K -- deuxieme palier VT (27/08)
# Meme pipeline que T1_recovery_static_330K.py (D1+AQ correction, meme
# modele) -- seule la grille DATASETS change.
#
# 14 points, D1=0.02-250s. NS boostee sur les 4 D1 les plus courts
# (128 au lieu de 32) pour eviter le probleme SNR/mispick ppm rencontre
# a 330K sur D1=0.02/0.05/0.2s.
#
# RG=216.57 confirme constant sur toute la serie (rga refait apres
# retuning). p1=2.5us utilise, base sur un spot-check separe non encore
# depouille -- verifier que le max reste bien a 2.5us avant de faire
# confiance a la serie.
#
# Ne couvre que le T1 de ce palier -- T2 (exp619+) traite a part une
# fois complete.
#
# exp635/636 (D1=1s/2s, NS=64) sont des points de verification croises
# des homologues NS=32 deja dans DATASETS (609/610) -- testent
# l'hypothese d'un decalage de calibration/sensibilite entre lots
# d'acquisition a NS different (souleve mais pas tranche au rattrapage
# 330K). Pas ajoutes a DATASETS (doublon de cle D1) -- compares a part
# dans NS_CROSSCHECK, hors fit.
NS_CROSSCHECK = {
    1: (r"D:\Postdoc\Datas\LLZO-400-aug26\635", 64),
    2: (r"D:\Postdoc\Datas\LLZO-400-aug26\636", 64),
}
# ============================================================

# === CONFIGURATION -- only section to edit ===
# D1 (s): (path, NS)
DATASETS = {
    0.02:  (r"D:\Postdoc\Datas\LLZO-400-aug26\605", 128),
    0.05:  (r"D:\Postdoc\Datas\LLZO-400-aug26\606", 128),
    0.2:   (r"D:\Postdoc\Datas\LLZO-400-aug26\607", 128),
    0.5:   (r"D:\Postdoc\Datas\LLZO-400-aug26\608", 128),
    1:     (r"D:\Postdoc\Datas\LLZO-400-aug26\609", 32),
    2:     (r"D:\Postdoc\Datas\LLZO-400-aug26\610", 32),
    4:     (r"D:\Postdoc\Datas\LLZO-400-aug26\611", 32),
    8:     (r"D:\Postdoc\Datas\LLZO-400-aug26\612", 32),
    15:    (r"D:\Postdoc\Datas\LLZO-400-aug26\613", 32),
    30:    (r"D:\Postdoc\Datas\LLZO-400-aug26\614", 32),
    60:    (r"D:\Postdoc\Datas\LLZO-400-aug26\615", 32),
    120:   (r"D:\Postdoc\Datas\LLZO-400-aug26\616", 32),
    150:   (r"D:\Postdoc\Datas\LLZO-400-aug26\617", 32),
    250:   (r"D:\Postdoc\Datas\LLZO-400-aug26\618", 32),
}

LB = 10               # line broadening in Hz
PH0_MANUAL = 0.0       # irrelevant in magnitude mode
PH1 = 0.0              # irrelevant in magnitude mode
AUTO_PH0 = False       # do not turn on -- unstable on this static line (documented)
READ_PHASE_FROM_PROCS = False
REFERENCE_SHIFT_PPM = 2      # inherited from 298K/330K referencing -- check the printed
                              # peak positions below; re-adjust here if they look shifted.
ZF_FACTOR = 1
PEAK_PPM_WINDOW = (-10, 15)   # same fix as 298K/330K
PPM_OUTLIER_THRESHOLD = 5.0   # ppm from the median -- flags a likely mispicked noise peak
FORCE_INCLUDE_D1 = []         # empty at first run -- see header comment; fill in only after
                              # manually verifying a failing point the same way as at 330K.
OUTPUT_NAME = r"D:\Postdoc\Figures\T1_global_statique_360K_provisoire"  # provisoire : T2
                              # de ce palier pas encore complet, garder ce nom tant que la
                              # figure finale du palier n'est pas prete
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
    print("=== T1 series (static probe, 360K palier, 27/08) ===")
    records = []  # (d1, intensity_per_scan, peak_ppm, aq)
    for d1, (path, ns) in sorted(DATASETS.items()):
        intensity, peak_ppm, aq = get_intensity(path)
        intensity_per_scan = intensity / ns
        print(f"  D1={d1:>6.2f}s  NS={ns:>4}  AQ={aq*1000:.1f}ms  peak at {peak_ppm:7.2f} ppm"
              f"  raw={intensity:.4e}  per-scan={intensity_per_scan:.4e}")
        records.append((d1, intensity_per_scan, peak_ppm, aq))

    # NS cross-check (609/610 @ NS=32 vs 635/636 @ NS=64) -- hors fit, voir header.
    print("\n--- NS cross-check (meme D1, NS different) ---")
    main_by_d1 = {r[0]: r for r in records}
    for d1_check, (path, ns_check) in sorted(NS_CROSSCHECK.items()):
        intensity, peak_ppm, aq = get_intensity(path)
        intensity_per_scan = intensity / ns_check
        ref = main_by_d1.get(d1_check)
        print(f"  D1={d1_check}s  NS={ns_check:>4}  peak at {peak_ppm:7.2f} ppm"
              f"  per-scan={intensity_per_scan:.4e}")
        if ref is not None:
            ref_per_scan = ref[1]
            rel_diff = 100 * (intensity_per_scan - ref_per_scan) / ref_per_scan
            print(f"    vs D1={d1_check}s NS=32 deja dans DATASETS "
                  f"(per-scan={ref_per_scan:.4e}) : ecart = {rel_diff:+.1f}%")
        else:
            print(f"    (pas de point NS=32 correspondant trouve dans DATASETS pour D1={d1_check}s)")
    print(
        "  Un ecart de quelques % est attendu (bruit statistique). Un ecart "
        "important et systematique (memes signe sur les deux D1) serait un "
        "signe du decalage de calibration NS deja suspecte au 330K -- sinon, "
        "l'hypothese peut etre ecartee pour ce palier."
    )

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
              f" ({median_ppm:.2f} ppm), likely mispicked noise (very short D1 -> very fast repetition,"
              f" possible receiver/probe settling artifact rather than just low SNR):")
        for d1, _, ppm, _ in bad:
            print(f"    D1={d1}s: peak at {ppm:.2f} ppm ({abs(ppm-median_ppm):.1f} ppm from median) -- EXCLUDED")
    else:
        print(f"\n  All peak positions within {PPM_OUTLIER_THRESHOLD} ppm of the median"
              f" ({median_ppm:.2f} ppm) -- no mispicked-peak red flag.")
    forced = [r for r in records if r[0] in FORCE_INCLUDE_D1 and abs(r[2] - median_ppm) > PPM_OUTLIER_THRESHOLD]
    for d1, _, ppm, _ in forced:
        print(f"    D1={d1}s: peak at {ppm:.2f} ppm ({abs(ppm-median_ppm):.1f} ppm from median)"
              f" -- FORCE-INCLUDED (see FORCE_INCLUDE_D1 comment: intensity checked manually against neighbours)")

    D1 = np.array([r[0] for r in good])
    I = np.array([r[1] for r in good])
    D1_eff = D1 + AQ   # <-- the actual fit variable: effective recycle delay, not D1 alone
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

    USE_TRIEXP_FOR_FIGURE = False  # flip to True by hand after comparing the two fits above

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

    resid_bi = 100 * (I - biexp_recovery(D1_eff, *popt_bi)) / I
    print("\nBiexponential relative residuals (%):", np.round(resid_bi, 2))

    df = pd.DataFrame({"D1_s": D1, "AQ_s": AQ, "D1_eff_s": D1_eff, "Intensity_per_scan": I})
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(f"\nResults exported to {OUTPUT_NAME}.csv")

    t_fit = np.logspace(np.log10(D1.min() / 2), np.log10(D1.max() * 1.3), 400)
    t_fit_eff = t_fit + AQ
    if USE_TRIEXP_FOR_FIGURE and use_tri:
        y_fit = triexp_recovery(t_fit_eff, *popt_tri)
        fit_legend = (f"tri-exp fit: T1c={T1c_tri:.1f}+/-{perr_tri[5]:.1f}s ({(1-f1_tri-f2_tri)*100:.1f}%), "
                      f"T1b={T1b_tri:.2f}+/-{perr_tri[4]:.2f}s ({f2_tri*100:.1f}%), "
                      f"T1a={T1a_tri*1000:.2f}+/-{perr_tri[3]*1000:.2f}ms ({f1_tri*100:.1f}%)")
        title_suffix = "triexponential"
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
    ax.set_title(r"$^7$Li T$_1$ recovery — static probe, 360K (27/08, RG=216.57)")
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
               xlog=True, title="7Li T1 recovery -- static probe, 360K")

    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")
