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
# A VERIFIER avant le premier run :
# - Chemin du CSV 298K : `T1_global_statique_298K.csv` est une supposition,
#   corrige si le vrai nom differe.
# - Noms de colonnes : le loader ci-dessous essaie plusieurs noms candidats
#   (D1_eff_s en priorite si present -- corrige AQ -- sinon D1_s) et
#   previent clairement s'il ne trouve rien, plutot que d'utiliser
#   silencieusement la mauvaise colonne.
#
# Pour ajouter 390K (ou tout palier futur) une fois son script individuel
# execute et son CSV exporte : ajouter une ligne dans TEMPERATURES
# ci-dessous, rien d'autre a changer.
# ============================================================

# === CONFIGURATION -- only section to edit ===
TEMPERATURES = {
    298: r"D:\Postdoc\Figures\T1_global_statique_298K.csv",
    330: r"D:\Postdoc\Figures\T1_global_statique_330K_provisoire_v2.csv",
    360: r"D:\Postdoc\Figures\T1_global_statique_360K_provisoire.csv",
}

D1_COL_CANDIDATES = ["D1_eff_s", "D1_s"]        # prefere D1_eff_s (corrige AQ) si present
INTENSITY_COL_CANDIDATES = ["Intensity_per_scan"]

# AQ (s) pour les temperatures dont le CSV n'a PAS de colonne D1_eff_s
# (ex. 298K). Applique uniquement a l'AFFICHAGE, jamais au fit : ajouter AQ
# au fit du 298K ecrase la grille D1 fine pres de 0 qui resout T1_fast
# (deja bien contraint par la grille dediee, cf. T1_recovery_static_298K.py)
# et fait diverger le fit sur un T1_fast fantome. Le decalage
# rapproche seulement l'axe visuel de celui de 330K/360K (ou D1+AQ est la
# vraie variable de fit) sans toucher aux valeurs de fit du 298K.
DISPLAY_AQ_OFFSET_S = {
    298: 0.5,
}

# Couleurs "froid -> chaud", dans l'ordre croissant de temperature (identite
# fixe, pas une colormap recyclee au hasard -- ajoute une couleur si tu
# ajoutes une 5e temperature)
COLOR_SEQUENCE = ["#2166ac", "#4dac26", "#fdae61", "#d7191c", "#7b3294"]

OUTPUT_NAME = r"D:\Postdoc\Figures\T1_comparaison_VT_provisoire"
# ================================================


def load_series(path):
    """Charge les donnees TELLES QUELLES (D1 brut ou D1_eff_s selon ce que
    contient le CSV) -- c'est cette valeur, et uniquement celle-la, qui doit
    servir au fit. Le decalage d'affichage (DISPLAY_AQ_OFFSET_S) est applique
    plus tard, separement, jamais ici."""
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
        print(f"=== T={T}K ({path}, colonne D1='{d1_col}', {len(d1)} points) ===")

        # Fit toujours sur d1 tel que charge (D1 brut ou D1_eff_s selon le
        # palier), jamais decale -- garantit T1_fast/T1_slow identiques aux
        # valeurs citees/validees dans le resume projet.
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

        t_fit = np.logspace(np.log10(d1.min() / 2), np.log10(d1.max() * 1.3), 400)
        y_fit = biexp_recovery(t_fit, *popt)

        # Affichage seulement : decalage cosmetique pour les paliers sans
        # D1_eff_s propre (298K), pour un axe x visuellement comparable a
        # 330K/360K. y_fit vient du fit sur d1 brut ; seule la position x
        # du trace est translatee.
        disp_offset = 0.0 if d1_col == "D1_eff_s" else DISPLAY_AQ_OFFSET_S.get(T, 0.0)
        if disp_offset:
            print(f"  (affichage seulement, fit inchange) decalage x de +{disp_offset}s "
                  f"applique pour comparabilite visuelle avec 330K/360K")
        d1_disp = d1 + disp_offset
        t_fit_disp = t_fit + disp_offset

        label = (f"{T}K: T1s={fmt_time(T1slow)} ({(1-f)*100:.0f}%), "
                 f"T1f={fmt_time(T1fast)} ({f*100:.0f}%)")
        ax.scatter(d1_disp, I / M0, color=color, s=45, zorder=3)
        ax.plot(t_fit_disp, y_fit / M0, color=color, lw=2, zorder=2, label=label)

        agr_series.append(dict(x=d1_disp, y=I / M0, mode="symbol", color=color, legend=f"{T}K data"))
        agr_series.append(dict(x=t_fit_disp, y=y_fit / M0, mode="line", color=color, legend=label))

    ax.set_xscale("log")
    ax.set_xlabel("Recovery delay D1 + AQ (s)")
    ax.set_ylabel("Normalized intensity (I / M0)")
    ax.set_title(r"$^7$Li T$_1$ recovery — static probe, VT comparison")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # L'axe est visuellement homogene (D1+AQ partout) mais pas le fit : a
    # 298K le +AQ n'est qu'un decalage d'affichage, le fit reste sur D1 brut
    # (necessaire pour resoudre T1_fast, cf. T1_recovery_static_298K.py). A 330K/360K, D1+AQ est la
    # vraie variable de fit. Precise ici pour eviter toute ambiguite a la
    # relecture (article ou revision future).
    ax.text(
        0.02, 0.98,
        "298K curve display-shifted by +AQ (~0.5s) for visual alignment only\n"
        "(fit performed on raw D1 -- T1_fast unaffected, see the dedicated\n"
        "298K script); D1+AQ is the\n"
        "true fit variable at 330K/360K (T1_fast ~ AQ there)",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        style="italic", color="dimgray",
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_NAME}.pdf")
    plt.show()

    export_agr(f"{OUTPUT_NAME}.agr", agr_series,
               xlabel="Recovery delay D1 (s)", ylabel="Normalized intensity (I/M0)",
               xlog=True, title="7Li T1 recovery -- static probe, VT comparison")

    # export_agr() ne prend pas de parametre pour un texte libre sur le
    # graphe -- la note de convention d'axe n'apparait donc que sur le
    # .pdf/.png matplotlib. Pour l'avoir aussi dans le .agr/Xmgrace : Plot ->
    # Text, cliquer en haut a gauche du graphe, coller le texte ci-dessous.
    print(
        "\nNote (a ajouter manuellement dans Xmgrace si tu veux la meme "
        "annotation sur le .agr) :\n"
        "  '298K curve display-shifted by +AQ (~0.5s) for visual alignment "
        "only (fit performed on raw D1 -- T1_fast unaffected, see the dedicated "
        "298K script); D1+AQ "
        "is the true fit variable at 330K/360K (T1_fast ~ AQ there)'"
    )
    print(f"\nDone. Figure saved as {OUTPUT_NAME}.pdf / .agr")