"""
Analyse XRD - identification de phase (LLZO tetragonal vs cubique vs La2Zr2O7)
================================================================================
Objectif : verifier si le diffractogramme correspond a du LLZO tetragonal pur,
ou s'il y a des signes de contamination par une phase secondaire (La2Zr2O7,
sous-produit de decomposition connu du LLZO a haute T).

Methode :
1) Extraction des positions 2theta / intensite depuis le fichier brut (.x00)
2) Detection des pics (maxima locaux avec un critere de proeminence simple)
3) Calcul des positions 2theta THEORIQUES par la loi de Bragg pour 3 phases
   candidates, a partir de leurs mailles de la litterature :
     - LLZO tetragonal, I41/acd, a=13.134 A, c=12.663 A (Awaka et al. 2009)
     - LLZO cubique,   Ia-3d,   a=12.9682 A (dopee, valeur typique)
     - La2Zr2O7 pyrochlore, Fd-3m, a=10.807 A
   Les regles de selection appliquees sont UNIQUEMENT celles du centrage du
   reseau (I : h+k+l pair : F : h,k,l de meme parite). Ce ne sont PAS les
   regles d'extinction completes du groupe d'espace (glissements, etc.) : donc
   des raies theoriques faibles/interdites en pratique peuvent apparaitre ici.
   -> Cette methode donne des POSITIONS correctes, mais pas les intensites
   relatives reelles (qui dependent des facteurs de structure).
4) Un simple "matching" de positions (pic observe <-> raie theorique la plus
   proche) est fourni a titre indicatif, MAIS ce n'est PAS une preuve de phase :
   les 3 phases ont des listes de raies denses et rapprochees entre 20 et 60
   degres, donc on peut "faire correspondre" presque n'importe quel pic a
   presque n'importe quelle phase par hasard (meme piege que les fits
   multi-exponentiels en RMN : trop de parametres libres = faux positifs).
5) Le test le plus fiable ici est un test QUALITATIF et NEGATIF : est-ce que
   les raies les plus INTENSES et les plus BASSES EN ANGLE de La2Zr2O7
   (111 ~14.2 deg, 002 ~16.4 deg, 022 ~23.3 deg) sont presentes ou absentes
   dans le signal brut ? Si elles sont absentes -> argument fort contre une
   contamination significative (ces raies ne dependent pas des parametres de
   maille affines, donc ce test est robuste).

Pour une identification de phase rigoureuse et quantitative (fractions
massiques, confirmation definitive), il faut un affinement Rietveld ou LeBail
sur le pattern complet (positions + intensites + profil), avec un logiciel
dedie : GSAS-II ou Profex/BGMN (tous deux gratuits, lisent le .xrdml
directement), ou HighScore si le labo y a acces.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1) Chargement du pattern experimental (fichier .x00)
# ---------------------------------------------------------------------------
PATH = "llzo.x00"
FIRST_ANGLE = 5.06000
STEP = 0.09200000000
LAMBDA_KA1 = 1.5405980  # Cu Ka1, en Angstrom

with open(PATH) as f:
    lines = [l.strip() for l in f]

start = lines.index("ScanData") + 1
intens = []
for l in lines[start:]:
    l = l.rstrip(",/")
    if l == "":
        continue
    try:
        intens.append(float(l))
    except ValueError:
        break
intens = np.array(intens)
two_theta = FIRST_ANGLE + STEP * np.arange(len(intens))

# ---------------------------------------------------------------------------
# 2) Detection de pics (maxima locaux + proeminence par rapport au fond local)
# ---------------------------------------------------------------------------
def find_peaks_simple(x, y, min_prominence, min_sep_pts=4):
    peaks = []
    for i in range(2, len(y) - 2):
        if y[i] > y[i-1] and y[i] >= y[i+1] and y[i] > y[i-2] and y[i] >= y[i+2]:
            lo, hi = max(0, i - 25), min(len(y), i + 25)
            local_bg = np.percentile(y[lo:hi], 25)
            prom = y[i] - local_bg
            if prom > min_prominence:
                peaks.append((x[i], y[i], prom))
    peaks.sort(key=lambda p: -p[1])
    kept = []
    for p in peaks:
        if all(abs(p[0] - k[0]) > min_sep_pts * STEP for k in kept):
            kept.append(p)
    kept.sort(key=lambda p: p[0])
    return kept

peaks = find_peaks_simple(two_theta, intens, min_prominence=0.15)

print(f"{len(peaks)} pics detectes (2theta / intensite / proeminence) :")
for tt, I, prom in peaks:
    print(f"  2theta = {tt:7.3f}   I = {I:7.3f}   prom = {prom:6.3f}")

# ---------------------------------------------------------------------------
# 3) Positions theoriques (Bragg) pour les 3 phases candidates
# ---------------------------------------------------------------------------
def bragg_2theta(d, lam=LAMBDA_KA1):
    x = lam / (2 * d)
    return 2 * np.degrees(np.arcsin(x)) if 0 < x <= 1 else None

def tetra_lines(a, c, hmax=6, lmax=6, tt_max=85):
    out = {}
    for h in range(hmax + 1):
        for k in range(hmax + 1):
            for l in range(lmax + 1):
                if h == k == l == 0 or (h + k + l) % 2 != 0:
                    continue
                d = 1 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)
                tt = bragg_2theta(d)
                if tt is not None and tt <= tt_max:
                    out.setdefault(round(tt, 3), (h, k, l, d))
    return out

def cubic_lines(a, hmax=9, tt_max=85, centering="I"):
    out = {}
    for h in range(hmax + 1):
        for k in range(hmax + 1):
            for l in range(hmax + 1):
                if h == k == l == 0:
                    continue
                if centering == "I" and (h + k + l) % 2 != 0:
                    continue
                if centering == "F" and len({h % 2, k % 2, l % 2}) != 1:
                    continue
                d = a / np.sqrt(h**2 + k**2 + l**2)
                tt = bragg_2theta(d)
                if tt is not None and tt <= tt_max:
                    out.setdefault(round(tt, 3), (h, k, l, d))
    return out

tetra_LLZO = tetra_lines(a=13.134, c=12.663)          # Awaka et al. 2009
cubic_LLZO = cubic_lines(a=12.9682, centering="I")     # cubique dopee typique
pyrochlore = cubic_lines(a=10.807, centering="F")      # La2Zr2O7

print("\n--- Raies La2Zr2O7 (pyrochlore) les plus basses en angle ---")
for tt, (h, k, l, d) in sorted(pyrochlore.items())[:5]:
    print(f"  2theta = {tt:7.3f}   hkl=({h}{k}{l})   d = {d:.4f} A")

# ---------------------------------------------------------------------------
# 4) Test qualitatif le plus fiable : les raies (111)/(002)/(022) de La2Zr2O7
#    sont-elles presentes dans le signal brut, meme faiblement ?
# ---------------------------------------------------------------------------
print("\n--- Signal brut autour des raies caracteristiques de La2Zr2O7 ---")
for target in [14.18, 16.39, 23.26]:
    i = np.argmin(np.abs(two_theta - target))
    lo, hi = max(0, i - 5), min(len(two_theta), i + 5)
    window = intens[lo:hi]
    print(f"  autour de {target:.2f} deg : I min={window.min():.3f} max={window.max():.3f} "
          f"(bruit de fond plat -> pas de pic ; un pic donnerait un I nettement "
          f"plus grand que ses voisins)")

# ---------------------------------------------------------------------------
# 5) Matching indicatif position-par-position (a interpreter avec prudence,
#    voir l'avertissement en tete de fichier)
# ---------------------------------------------------------------------------
def best_match(obs_tt, candidates, tol=0.25):
    matches = [(tt, hkl) for tt, hkl in candidates.items() if abs(tt - obs_tt) < tol]
    if not matches:
        return None
    matches.sort(key=lambda m: abs(m[0] - obs_tt))
    return matches[0]

print("\n=== Matching indicatif (tol = 0.25 deg) ===")
header = f"{'2theta obs':>10} {'I':>7} | {'tetra LLZO':>16} | {'cubique LLZO':>16} | {'La2Zr2O7':>16}"
print(header)
for tt_obs, I_obs, prom in peaks:
    row = f"{tt_obs:10.3f} {I_obs:7.2f} |"
    for cand in (tetra_LLZO, cubic_LLZO, pyrochlore):
        m = best_match(tt_obs, cand)
        if m:
            tt_c, (h, k, l, d) = m
            row += f" ({h}{k}{l}) {tt_c:7.3f}  |"
        else:
            row += f" {'--':>16} |"
    print(row)