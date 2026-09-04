import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from library_nmr.agr_export import export_agr

# ============================================================
# T2_comparison_VT -- superposition des courbes T2(decho) a plusieurs T.
#
# Meme principe que T1_comparison_VT.py : lit les CSV deja exportes par
# chaque T2_recovery_static_XXXK.py (pas de retraitement des spectres
# bruts), refit chaque temperature independamment, normalise par M0 pour
# comparer les formes de decroissance malgre des intensites absolues tres
# differentes (RG different a 298K, population de Boltzmann/T).
#
# Modele MONO-exponentiel (pas bi-exponentiel) : le modele bi-exponentiel
# donnait un "T2_slow" fantome -- artefact de plancher de bruit magnitude,
# confirme par cross-check NS au 360K (cf. resume projet). Le fit est donc
# restreint a la region ou la decroissance est du vrai signal (voir
# FIT_DECHO_MAX_US ci-dessous).
#
# ATTENTION -- 330K PAS ENCORE DIAGNOSTIQUE POUR LE PLANCHER DE BRUIT :
# contrairement au 360K, le CSV 330K n'a pas encore ete passe par le meme
# audit (cross-check NS sur les points a decho long). Son T2_fast doit donc
# etre traite comme provisoire. Refaire le meme cross-check NS avant de le
# citer dans l'article.
# ============================================================

# === CONFIGURATION -- only section to edit ===
TEMPERATURES = {
    330: r"D:\Postdoc\Figures\T2_global_statique_330K_provisoire.csv",
    360: r"D:\Postdoc\Figures\T2_global_statique_360K_provisoire.csv",
    298: r"D:\Postdoc\Figures\T2_global_statique_298K_provisoire.csv",
    315: r"D:\Postdoc\Figures\T2_global_statique_315K_provisoire.csv"
}

# Coupure explicite de la region de fit (decho max, en us) pour les paliers
# deja audites pour le plancher de bruit (cf. NS_CROSSCHECK dans le script
# individuel correspondant). 360K : deja restreint a L0<=22 (decho<=220us)
# des l'export du CSV individuel, garde explicite ici par securite si le CSV
# source change de convention.
#
# 330K : coupure a 220us basee sur une observation visuelle (la courbe
# continuait a decroitre jusqu'a ~500-600us sans cette coupure, bien au-dela
# d'ou le 360K s'arrete) -- pas encore un cross-check NS aussi rigoureux que
# celui fait au 360K.
FIT_DECHO_MAX_US = {
    330: 220,
    360: 220,
}

DECHO_COL_CANDIDATES = ["decho_us"]
INTENSITY_COL_CANDIDATES = ["Intensity_per_scan"]

# Couleurs "froid -> chaud", identite fixe dans l'ordre croissant de
# temperature -- ajoute une couleur si tu ajoutes une temperature de plus
COLOR_SEQUENCE = ["#2166ac", "#fdae61", "#d7191c", "#7b3294"]

OUTPUT_NAME = r"D:\Postdoc\Figures\T2_comparaison_VT_provisoire"
# ================================================


def load_series(path):
    df = pd.read_csv(path)
    decho_col = next((c for c in DECHO_COL_CANDIDATES if c in df.columns), None)
    i_col = next((c for c in INTENSITY_COL_CANDIDATES if c in df.columns), None)
    if decho_col is None or i_col is None:
        raise ValueError(
            f"Colonnes attendues introuvables dans {path} (colonnes presentes : "
            f"{list(df.columns)}) -- ajuste DECHO_COL_CANDIDATES/INTENSITY_COL_CANDIDATES "
            f"en tete de script."
        )
    decho_us = df[decho_col].to_numpy(dtype=float)
    I = df[i_col].to_numpy(dtype=float)
    order = np.argsort(decho_us)
    return decho_us[order], I[order]


def monoexp_decay(t, M0, T2):
    return M0 * np.exp(-t / T2)


def fmt_time(t_seconds):
    """Formatte un temps avec l'unite la plus lisible (us/ms/s) -- meme fix
    que T1_comparison_VT.py, evite la notation scientifique moche si un T2
    depasse 1ms."""
    if t_seconds >= 1:
        return f"{t_seconds:.2g}s"
    elif t_seconds >= 1e-3:
        return f"{t_seconds * 1000:.2g}ms"
    else:
        return f"{t_seconds * 1e6:.2g}us"


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 5.5))
    agr_series = []

    for color, (T, path) in zip(COLOR_SEQUENCE, sorted(TEMPERATURES.items())):
        decho_us, I = load_series(path)

        decho_max = FIT_DECHO_MAX_US.get(T)
        if decho_max is not None:
            mask = decho_us <= decho_max
            n_dropped = (~mask).sum()
            decho_us, I = decho_us[mask], I[mask]
            print(f"=== T={T}K ({path}, {len(decho_us)} points apres coupure a "
                  f"decho<={decho_max}us, {n_dropped} point(s) ecarte(s)) ===")
        else:
            print(f"=== T={T}K ({path}, {len(decho_us)} points, PAS de coupure -- "
                  f"palier pas encore audite pour le plancher de bruit) ===")

        decho_s = decho_us * 1e-6

        p0 = [1.1 * I.max(), 1e-4]
        bounds = ([0.5 * I.max(), 1e-6], [5 * I.max(), 1e-2])
        try:
            popt, pcov = curve_fit(monoexp_decay, decho_s, I, p0=p0, sigma=I, maxfev=50000, bounds=bounds)
        except RuntimeError as e:
            print(f"  Fit failed pour T={T}K ({e}) -- verifie le CSV/les colonnes, palier ignore.")
            continue
        perr = np.sqrt(np.diag(pcov))
        M0, T2 = popt
        rel_err = perr[1] / T2 * 100
        print(f"  T2={T2*1e6:.3g}+/-{perr[1]*1e6:.2g}us ({rel_err:.0f}% relatif)  M0={M0:.4e}")
        if rel_err > 30:
            print(f"  ATTENTION : incertitude relative elevee ({rel_err:.0f}%) -- signe possible "
                  f"d'une contamination par le plancher de bruit non encore diagnostiquee/coupee "
                  f"pour ce palier (meme demarche de cross-check NS que faite au 360K a envisager).")

        t_fit = np.logspace(np.log10(decho_s.min() / 2), np.log10(decho_s.max() * 1.3), 400)
        y_fit = monoexp_decay(t_fit, *popt)

        label = f"{T}K: T2={T2*1e6:.1f}+/-{perr[1]*1e6:.1f}us"  # pas fmt_time(T2) : .2g rendait les trois valeurs VT indistinguables ("1e+02 us") et omettait l'incertitude
        ax.scatter(decho_us, I / M0, color=color, s=45, zorder=3)
        ax.plot(t_fit * 1e6, y_fit / M0, color=color, lw=2, zorder=2, label=label)

        agr_series.append(dict(x=decho_us, y=I / M0, mode="symbol", color=color, legend=f"{T}K data"))
        agr_series.append(dict(x=t_fit * 1e6, y=y_fit / M0, mode="line", color=color, legend=label))

    ax.set_xscale("log")
    ax.set_xlabel("Echo delay decho (us)")
    ax.set_ylabel("Normalized intensity (I / M0)")
    ax.set_title(r"$^7$Li T$_2$ decay — static probe, VT comparison")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Echo delay decho (us)", ylabel="Normalized intensity (I/M0)",
               xlog=True, title="7Li T2 decay -- static probe, VT comparison")

    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")