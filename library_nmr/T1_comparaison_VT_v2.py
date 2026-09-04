import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from library_nmr.agr_export import export_agr

# ============================================================
# T1_comparison_VT -- superposition des courbes T1(D1) a plusieurs T.
#
# Ne retraite AUCUN spectre brut : lit directement les CSV deja exportes par
# chaque script T1_recovery_static_XXXK.py (un point par experience,
# D1/intensite par scan), puis refit chaque temperature independamment avec
# le meme modele biexponentiel que les scripts individuels. Chaque courbe
# est normalisee par son propre M0 (plateau ajuste) pour comparer les
# FORMES de recuperation malgre des intensites absolues tres differentes
# d'un palier a l'autre (RG different a 298K, population de Boltzmann qui
# varie avec T -- loi de Curie -- sans lien avec la vitesse de relaxation).
#
# AXE X UNIFIE EN D1+AQ POUR TOUS LES PALIERS, y compris dans le FIT (pas
# seulement l'affichage, contrairement a la version precedente). ATTENTION :
# ajouter AQ avant le fit du 298K a deja detruit une fois la resolution de
# T1_fast (grille D1 concue pour ce point devient quasi degeneree pres
# de 0.5-0.7s des qu'on ajoute AQ, et le fit part sur un T1_fast fantome,
# une valeur non physique proche de la demi-seconde). SI LE T1_FAST DU 298K
# RESSORT TRES DIFFERENT DE CELUI DE LA GRILLE DEDIEE (T1_recovery_static_
# 298K.py), C'EST CET ARTEFACT QUI REAPPARAIT, pas une nouvelle donnee --
# revenir a la version
# precedente (double convention + annotation) dans ce cas ; pas negociable
# par un meilleur p0/bounds, l'information elle-meme est perdue au decalage.
# ============================================================

# === CONFIGURATION -- only section to edit ===
TEMPERATURES = {
    298: r"D:\Postdoc\Figures\T1_global_statique_298K.csv",
    330: r"D:\Postdoc\Figures\T1_global_statique_330K_provisoire_v2.csv",
    360: r"D:\Postdoc\Figures\T1_global_statique_360K_provisoire.csv",
    # 390: r"D:\Postdoc\Figures\T1_global_statique_390K_provisoire.csv",
}

D1_COL_CANDIDATES = ["D1_eff_s", "D1_s"]        # prefere D1_eff_s (deja D1+AQ) si present
INTENSITY_COL_CANDIDATES = ["Intensity_per_scan"]

# AQ (s) a AJOUTER AU FIT pour les temperatures dont le CSV n'a PAS de
# colonne D1_eff_s (ex. 298K, colonne D1_s = D1 brut). Pour les temperatures
# qui ont deja D1_eff_s (330K/360K), rien n'est ajoute ici -- leur propre AQ
# est deja inclus dans la colonne chargee.
AQ_S = {
    298: 0.5,
}

# Couleurs "froid -> chaud", dans l'ordre croissant de temperature (identite
# fixe, pas une colormap recyclee au hasard -- ajoute une couleur si tu
# ajoutes une 5e temperature)
COLOR_SEQUENCE = ["#2166ac", "#4dac26", "#fdae61", "#d7191c", "#7b3294"]

OUTPUT_NAME = r"D:\Postdoc\Figures\T1_comparaison_VT_provisoire_2"
# ================================================


def load_series(path):
    """Charge les donnees TELLES QUELLES (D1 brut ou D1_eff_s selon ce que
    contient le CSV)."""
    df = pd.read_csv(path)
    d1_col = next((c for c in D1_COL_CANDIDATES if c in df.columns), None)
    i_col = next((c for c in INTENSITY_COL_CANDIDATES if c in df.columns), None)
    if d1_col is None or i_col is None:
        raise ValueError(
            f"Colonnes attendues introuvables dans {path} (colonnes presentes : "
            f"{list(df.columns)}) -- ajuste D1_COL_CANDIDATES/INTENSITY_COL_CANDIDATES "
            f"en tete de script."
        )
    d1 = df[d1_col].to_numpy(dtype=float)
    I = df[i_col].to_numpy(dtype=float)
    order = np.argsort(d1)
    return d1[order], I[order], d1_col


def biexp_recovery(t, M0, f, T1a, T1b):
    return M0 * (1 - f * np.exp(-t / T1a) - (1 - f) * np.exp(-t / T1b))


def fmt_time(t_seconds):
    """Formatte un temps avec l'unite la plus lisible (us/ms/s) -- evite la
    notation scientifique moche qui apparaissait pour un T1_fast > 1s affiche
    de force en ms (ex. 2.6e+03ms au lieu de 2.6s)."""
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
        d1, I, d1_col = load_series(path)
        aq = 0.0 if d1_col == "D1_eff_s" else AQ_S.get(T, 0.0)
        if aq:
            d1 = d1 + aq
            print(f"=== T={T}K ({path}, colonne D1='{d1_col}', {len(d1)} points, "
                  f"+{aq}s AQ ajoute AVANT le fit) ===")
        else:
            print(f"=== T={T}K ({path}, colonne D1='{d1_col}', {len(d1)} points) ===")

        # Fit sur d1+AQ pour tous les paliers, de facon homogene -- voir
        # avertissement en tete de fichier.
        p0 = [1.1 * I.max(), 0.15, 0.1, 15]
        bounds = ([0.5 * I.max(), 0, 1e-4, 0.5], [5 * I.max(), 1, 10, 500])
        try:
            popt, pcov = curve_fit(biexp_recovery, d1, I, p0=p0, sigma=I, maxfev=50000, bounds=bounds)
        except RuntimeError as e:
            print(f"  Fit failed pour T={T}K ({e}) -- verifie le CSV/les colonnes, palier ignore.")
            continue
        perr = np.sqrt(np.diag(pcov))
        M0, f, T1fast, T1slow = popt
        print(f"  T1_slow={T1slow:.3g}+/-{perr[3]:.2g}s ({(1-f)*100:.1f}%)  "
              f"T1_fast={T1fast*1000:.3g}+/-{perr[2]*1000:.2g}ms ({f*100:.1f}%)  M0={M0:.4e}")
        if T == 298:
            print(f"  >>> VERIFIE : T1_fast doit rester proche de la valeur de la grille "
                  f"dediee (T1_recovery_static_298K.py) -- si tres different, "
                  f"c'est l'artefact documente qui reapparait (voir en-tete du fichier).")

        t_fit = np.logspace(np.log10(d1.min() / 2), np.log10(d1.max() * 1.3), 400)
        y_fit = biexp_recovery(t_fit, *popt)

        label = (f"{T}K: T1s={fmt_time(T1slow)} ({(1-f)*100:.0f}%), "
                 f"T1f={fmt_time(T1fast)} ({f*100:.0f}%)")
        ax.scatter(d1, I / M0, color=color, s=45, zorder=3)
        ax.plot(t_fit, y_fit / M0, color=color, lw=2, zorder=2, label=label)

        agr_series.append(dict(x=d1, y=I / M0, mode="symbol", color=color, legend=f"{T}K data"))
        agr_series.append(dict(x=t_fit, y=y_fit / M0, mode="line", color=color, legend=label))

    ax.set_xscale("log")
    ax.set_xlabel("Recovery delay D1 + AQ (s)")
    ax.set_ylabel("Normalized intensity (I / M0)")
    ax.set_title(r"$^7$Li T$_1$ recovery — static probe, VT comparison")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Recovery delay D1 + AQ (s)", ylabel="Normalized intensity (I/M0)",
               xlog=True, title="7Li T1 recovery -- static probe, VT comparison")

    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")