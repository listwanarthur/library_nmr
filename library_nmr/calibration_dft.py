import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from library_nmr.agr_export import export_agr

# ============================================================
# CALIBRATION DFT/GIPAW — bibliotheque_nmr
# Utilisation : modifier le bloc CONFIGURATION puis lancer
# ============================================================

# === CONFIGURATION ===
FICHIER_MAGRES = "test.magres" # fichier de sortie CASTEP/GIPAW
NOYAU = "Li" # noyau à extraire du fichier magres

# Composés de référence pour la calibration
# delta_exp : déplacements chimiques expérimentaux (ppm)
# sigma_calc : blindages calculés par GIPAW (ppm)
DELTA_EXP_REF = [10.0, 5.0, 0.0, -5.0, -10.0]
SIGMA_CALC_REF = [270.0, 275.0, 280.0, 285.0, 290.0]

# Références depuis fichier CSV ou entrées manuellement
CHARGER_REFS_CSV = False # mettre True pour charger depuis un fichier
FICHIER_REFS = "references.csv" # colonnes attendues : delta_exp, sigma_calc

NOM_SORTIE = "calibration_dft"
# =====================


def lire_magres(fichier, noyau):
    """Lit un fichier .magres et extrait les sigma_iso pour le noyau choisi.
    Comparaison du nom de noyau insensible à la casse ("Li" vs "li" selon les versions)."""
    resultats = []
    with open(fichier, "r") as f:
        for numero_ligne, ligne in enumerate(f, start=1):
            if not ligne.startswith("ms"):
                continue
            parties = ligne.split()
            if len(parties) < 12:
                print(f"  Ligne {numero_ligne} ignorée (format inattendu, "
                      f"{len(parties)} champs au lieu de 12 attendus) : {ligne.strip()!r}")
                continue
            if parties[1].lower() != noyau.lower():
                continue
            try:
                numero = parties[2]
                valeurs = [float(parties[3]), float(parties[7]), float(parties[11])]
                sigma_iso = np.mean(valeurs)
                resultats.append({
                    "atome": noyau,
                    "numero": numero,
                    "sigma_iso": sigma_iso
                })
            except ValueError:
                print(f"  Ligne {numero_ligne} ignorée (valeurs non numériques) : {ligne.strip()!r}")
    return pd.DataFrame(resultats)


def calibration(sigma_calc_ref, delta_exp_ref):
    """Calcule la droite de calibration par régression linéaire."""
    a, b = np.polyfit(sigma_calc_ref, delta_exp_ref, deg=1)
    return a, b


def appliquer_calibration(sigma_iso, a, b):
    """Applique la droite de calibration pour obtenir delta prédit."""
    return a * sigma_iso + b


def validation_croisee_loo(sigma_calc_ref, delta_exp_ref):
    """Validation croisée "leave-one-out" (LOOCV) de la calibration.
    Le R² naïf (sur les points d'ajustement) est toujours optimiste avec peu de
    points ; le LOOCV retire un point, refait la régression, prédit le point
    exclu, et compare — mesure plus honnête de la fiabilité réelle.
    Retourne (delta_predit_loo, rmse_loo, r2_loo)."""
    sigma_calc_ref = np.asarray(sigma_calc_ref, dtype=float)
    delta_exp_ref = np.asarray(delta_exp_ref, dtype=float)
    n = len(sigma_calc_ref)
    delta_predit_loo = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        a_i, b_i = calibration(sigma_calc_ref[mask], delta_exp_ref[mask])
        delta_predit_loo[i] = appliquer_calibration(sigma_calc_ref[i], a_i, b_i)

    residus_loo = delta_exp_ref - delta_predit_loo
    rmse_loo = np.sqrt(np.mean(residus_loo**2))
    ss_res_loo = np.sum(residus_loo**2)
    ss_tot = np.sum((delta_exp_ref - np.mean(delta_exp_ref))**2)
    r2_loo = 1 - ss_res_loo / ss_tot
    return delta_predit_loo, rmse_loo, r2_loo


# === TRAITEMENT ===

# Chargement des références de calibration
if CHARGER_REFS_CSV:
    df_refs = pd.read_csv(FICHIER_REFS)
    for colonne in ("sigma_calc", "delta_exp"):
        if colonne not in df_refs.columns:
            raise ValueError(f"Colonne '{colonne}' absente de {FICHIER_REFS} — "
                              f"colonnes trouvées : {list(df_refs.columns)}")
    sigma_ref = df_refs["sigma_calc"].values
    delta_ref = df_refs["delta_exp"].values
else:
    sigma_ref = np.array(SIGMA_CALC_REF)
    delta_ref = np.array(DELTA_EXP_REF)

if len(sigma_ref) != len(delta_ref):
    raise ValueError(f"sigma_ref ({len(sigma_ref)} valeurs) et delta_ref ({len(delta_ref)} valeurs) "
                      f"n'ont pas la même longueur.")
if len(sigma_ref) < 2:
    raise ValueError(f"Il faut au moins 2 composés de référence pour une régression linéaire "
                      f"(actuellement {len(sigma_ref)}).")
if len(sigma_ref) < 4:
    print(f"ATTENTION : seulement {len(sigma_ref)} composés de référence — la validation croisée "
          f"leave-one-out sera peu informative avec si peu de points (idéalement 5+).")

# Calcul de la droite de calibration et du R² (naïf, sur les points d'ajustement)
a, b = calibration(sigma_ref, delta_ref)
delta_predit_ref = a * sigma_ref + b
ss_res = np.sum((delta_ref - delta_predit_ref)**2)
ss_tot = np.sum((delta_ref - np.mean(delta_ref))**2)
r2 = 1 - ss_res / ss_tot
print(f"Droite de calibration : delta = {a:.4f} * sigma + {b:.2f}")
print(f"R² (naïf, sur les points d'ajustement) = {r2:.4f}")

# Validation croisée leave-one-out — mesure plus honnête de la fiabilité réelle
delta_predit_loo, rmse_loo, r2_loo = validation_croisee_loo(sigma_ref, delta_ref)
print(f"R² (validation croisée leave-one-out) = {r2_loo:.4f}")
print(f"RMSE (validation croisée leave-one-out) = {rmse_loo:.3f} ppm")
if r2 - r2_loo > 0.1:
    print(f"ATTENTION : écart notable entre R² naïf ({r2:.4f}) et R² LOOCV ({r2_loo:.4f}) — "
          f"la calibration est probablement moins fiable hors des points de référence "
          f"que ne le suggère le R² naïf seul.")

# Figure 1 — droite de calibration
sigma_ligne = np.linspace(min(sigma_ref), max(sigma_ref), 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sigma_ref, delta_ref, color="blue", label="références")
ax.plot(sigma_ligne, a * sigma_ligne + b, color="red", label=f"calibration R²={r2:.4f}")
ax.set_xlabel(f"sigma_calc {NOYAU} (ppm)")
ax.set_ylabel("delta_exp (ppm)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
plt.savefig(f"{NOM_SORTIE}_calibration.pdf")
plt.show()

export_agr(
    f"{NOM_SORTIE}_calibration.agr",
    series=[
        dict(x=sigma_ref, y=delta_ref, mode="symbol", color="blue", legend="références"),
        dict(x=sigma_ligne, y=a * sigma_ligne + b, mode="line", color="red",
             legend=f"calibration R²={r2:.4f}"),
    ],
    xlabel=f"sigma_calc {NOYAU} (ppm)", ylabel="delta_exp (ppm)",
)

# Figure 2 — graphe de parité (delta_exp vs delta_prédit), points d'ajustement ET LOOCV
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(delta_predit_ref, delta_ref, color="blue", label="ajustement (optimiste)")
ax.scatter(delta_predit_loo, delta_ref, color="orange", marker="x", label="validation croisée (LOO)")
lim = [min(delta_ref) - 2, max(delta_ref) + 2]
ax.plot(lim, lim, color="black", linestyle="--", label="y=x (idéal)")
ax.set_xlabel("delta prédit (ppm)")
ax.set_ylabel("delta exp (ppm)")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
plt.savefig(f"{NOM_SORTIE}_parite.pdf")
plt.show()

export_agr(
    f"{NOM_SORTIE}_parite.agr",
    series=[
        dict(x=delta_predit_ref, y=delta_ref, mode="symbol", color="blue", legend="ajustement (optimiste)"),
        dict(x=delta_predit_loo, y=delta_ref, mode="symbol", color="orange", legend="validation croisée (LOO)"),
        dict(x=lim, y=lim, mode="line", color="black", legend="y=x (idéal)"),
    ],
    xlabel="delta prédit (ppm)", ylabel="delta exp (ppm)",
)

# Lecture fichier magres et application de la calibration
df = lire_magres(FICHIER_MAGRES, NOYAU)
if df.empty:
    print(f"\nATTENTION : aucun atome '{NOYAU}' trouvé dans {FICHIER_MAGRES}. "
          f"Vérifie le nom du noyau (casse/orthographe) et que le fichier est le bon — "
          f"le CSV de sortie sera vide.")
else:
    df["delta_predit"] = appliquer_calibration(df["sigma_iso"], a, b)
    print(f"\n{len(df)} atome(s) '{NOYAU}' trouvé(s) :")
    print(df)

# Export CSV
df.to_csv(f"{NOM_SORTIE}.csv", index=False)
print(f"Résultats exportés dans {NOM_SORTIE}.csv")

df_calib = pd.DataFrame({
    "sigma_calc_ref": sigma_ref,
    "delta_exp_ref": delta_ref,
    "delta_predit_ref": delta_predit_ref,
    "residus": delta_ref - delta_predit_ref,
    "delta_predit_loo": delta_predit_loo,
    "residus_loo": delta_ref - delta_predit_loo,
})
df_calib.to_csv(f"{NOM_SORTIE}_calibration.csv", index=False)
print(f"Calibration exportée dans {NOM_SORTIE}_calibration.csv")