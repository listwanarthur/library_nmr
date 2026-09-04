# library_nmr

A Python pipeline for processing and analyzing solid-state NMR spectra of
battery solid electrolytes (LLZO, LATP), built on top of
[nmrglue](https://github.com/jjhelmus/nmrglue) for reading Bruker TopSpin
raw data.

Covers the full workflow used in day-to-day acquisition: GRPDLY correction,
automatic phasing, multi-component pseudo-Voigt peak fitting, T1/T2
relaxation extraction (including two-component analysis for overlapping
sites), MQMAS 2D processing, and DFT/GIPAW chemical-shift calibration.

## Why this project

I'm doing a postdoc in solid-state NMR on solid electrolytes for
batteries (LLZO, LATP). A lot of the day-to-day work — phasing a
spectrum, fitting overlapping peaks, extracting a T1 or T2 from a
pseudo-2D series — was stuff I kept redoing by hand in TopSpin, dataset
after dataset. At some point it made more sense to write it once in
Python and reuse it.

I had never really coded before starting this. I learned Python mostly
through this project, one script at a time, so don't expect
textbook-clean software engineering everywhere — but the processing
logic itself (phasing strategy, fitting methodology, drift diagnostics)
comes from actually running these experiments and hitting real problems
with them.

## What it does

| Module | Purpose |
|---|---|
| `pipeline_1d.py` | Main 1D pipeline: GRPDLY correction, auto-phasing, multi-component pseudo-Voigt deconvolution, baseline correction, eta-hypothesis comparison |
| `multi_spectra_comparison.py` | Overlay / stack several 1D spectra with independent processing and normalization (max or area) |
| `check_drift.py` | Diagnoses B0/shim/temperature drift over long pseudo-2D acquisitions, independent of the swept delay |
| `relaxation_T1ir.py` | T1 via inversion recovery, single peak |
| `relaxation_T1ir_twocomponents.py` | T1 via inversion recovery, two overlapping lineshape components (shape fixed on the fully-relaxed row, amplitudes solved per row by linear least squares) |
| `relaxation_T1sr.py` | T1 via saturation recovery, single peak |
| `relaxation_T1sr_twocomponents.py` | T1 via saturation recovery, two components |
| `relaxation_T1_onepulse_series.py` | T1 recovery via a series of independent, individually-phased 1D spectra (D1 series) with biexponential fit — more robust alternative to the pseudo-2D approach when the shared-reference decomposition proved unstable |
| `relaxation_T1_components.py` | T1 per lithium environment: lineshape fixed once on the fully-relaxed (best-S/N) spectrum of the same D1 series, amplitudes solved by NNLS, each component's own biexponential T1 fit — narrow and broad come out with essentially the same T1, unlike T2 (see Notes below) |
| `relaxation_T1_vs_temperature.py` | Variable-temperature T1: runs the same onepulse-D1-series strategy as `relaxation_T1_onepulse_series.py` across a whole temperature series in one pass (biexponential / monoexponential / two-point model selection depending on how many points are usable at each temperature), with an automatic room-temperature sanity check against the corrected reference value |
| `relaxation_T2.py` | T2 via spin-echo, monoexponential fit |
| `relaxation_T2_echo_series.py` | T2 via a series of independent 1D spectra at discrete echo delays (L0 x rotor period): phase frozen once on the reference spectrum, intensity by trapezoidal integration (not argmax) for robustness at low S/N, biexponential fit with relative weighting |
| `relaxation_T2_components.py` | T2 per lithium environment: lineshape fixed on an EXTERNAL onepulse reference spectrum (not an echo spectrum — see Notes below for why), amplitudes solved by NNLS, each component fit independently with its own biexponential decay |
| `mqmas_2d_processing.py` | 2D MQMAS (triple-quantum) processing: States F1 reconstruction, shearing, isotropic/MAS projections |
| `calibration_dft.py` | GIPAW/DFT chemical-shift calibration from reference compounds, with leave-one-out cross-validation |
| `core.py` | Shared low-level building blocks: GRPDLY handling, FID processing, automatic phasing, Bruker delay-list parsing |
| `fitting.py` | Shared pseudo-Voigt lineshape fitting (single and multi-component) |
| `agr_export.py` | Shared: writes matplotlib-style series to a Grace (`.agr`) project file for reopening/polishing in Xmgrace — used by all the figure-producing scripts below |

## T1/T2 relaxation vs. temperature (per-campaign scripts)

One script per temperature/campaign, all built on the shared `core.py` /
`agr_export.py` functions above rather than duplicating that logic. Kept
as separate scripts (not parametrized into one) because each campaign
picked up its own real-world complications (RG changes, lot-mixing,
noise-floor artifacts) that are easier to document and audit per-script
than to hide behind a generic config.

| Module | Purpose |
|---|---|
| `T1_recovery_static_298K.py` | T1 at 298K, static probe, magnitude mode: mixes two RG blocks (archive + plateau-closing addition) via an empirical cross-calibration correction |
| `T1_recovery_static_298K_RG94_only.py` | Same 298K T1 grid restricted to the single self-consistent RG=94.34 block — sidesteps the RG cross-calibration correction entirely |
| `T1_global_statique_298K_peakwindow_test.py` | 298K T1, RG=94.34-only night session, peak-window sensitivity check |
| `T1_recovery_static_315K.py` | T1 at 315K (new sep26 campaign), two independent 19-point series compared for reproducibility |
| `T1_recovery_static_330K.py` | T1 at 330K, grid re-acquired in two consecutive series after a lot-mixing artifact was diagnosed in the first attempt (see `Check_lot_reproducibility.py`) |
| `T1_recovery_330K_v2.py` | 330K T1 redone as one continuous 21-point session after a probe remount, D1 extended to 500s to fully close the recovery plateau |
| `T1_recovery_static_345K.py` | T1 at 345K, same biexp/triexp pipeline as 315K, D1 grid extended to 400/500s from the start (plateau-closing lesson applied up front) |
| `T1_recovery_static_360K.py` | T1 at 360K, NS boosted on the shortest D1 points to fix an SNR/mispick issue first seen at 330K |
| `T1_comparaison_VT.py` / `T1_comparaison_VT_v2.py` | Overlay T1(D1) recovery curves across all temperatures from the already-exported per-temperature CSVs (v2 unifies the x-axis on D1+AQ throughout, including in the fit itself, not just the display) |
| `T2_recovery_static_298K.py` | T2 at 298K, fine echo-delay grid with dedicated noise-floor cross-checks (`PLATEAU_CHECK`, `NS_CROSSCHECK`) |
| `T2_recovery_static_315K.py` | T2 at 315K (sep26 campaign), grid extended further to directly verify the noise-floor threshold established at other temperatures |
| `T2_recovery_static_330K.py` | T2 at 330K, fine grid plus an NS cross-check exposing a magnitude-mode noise-floor signature at long echo delays |
| `T2_recovery_static_345K.py` | T2 at 345K, lighter grid informed by the 315K diagnostic (skips the ambiguous long-echo-delay zone) |
| `T2_recovery_static_360K.py` | T2 at 360K, same pipeline as 330K, with an NS=64 vs NS=220 cross-check at the longest echo delays |
| `T2_comparaison_VT.py` | Overlay T2(echo delay) decay curves across all temperatures; monoexponential rather than biexponential, since the cross-checks showed the apparent "T2_slow" component was a noise-floor fitting artifact, not real signal |

## Cross-checks & diagnostics

Small scripts written to answer one specific question about the data
(is this artifact real? is the phasing stable?) rather than to produce a
final figure — kept in the repo alongside the main scripts because the
answer they gave shaped a methodology decision upstream (e.g. the
monoexponential-only choice in `T2_comparaison_VT.py` above).

| Module | Purpose |
|---|---|
| `Check_lot_reproducibility.py` | Compares intensity-per-scan between two acquisition lots at a shared D1 point — the diagnostic that caught the 330K lot-mixing artifact |
| `Check_T2_298K_NS_crosscheck.py` | NS=64 vs NS=220 intensity comparison at fixed echo delay — isolates a magnitude-mode noise-floor bias from real T2 decay |
| `diagnostic_T1_ph0_scan.py` | Quick PH0 comparison overlay on a single T1-series spectrum |
| `diagnostic_T1_ph1_sidebands_scan.py` | Wide PH1 sweep to re-check phasing against the spinning sidebands |
| `diagnostic_T1_series_overlay.py` | Overlays spectra from several D1 points of a T1 series to sanity-check phasing/lineshape consistency across the series |
| `diagnostic_T2_ph0_scan.py` | PH0 sweep on the best-S/N T2 spectrum |
| `diagnostic_T2_fit_robustness.py` | Refits exported per-component T2 amplitudes under different sigma-weighting choices to check the fit isn't sensitive to that choice |
| `diagnostic_T2_model_comparison.py` | Mono- vs bi-exponential model comparison on exported per-component T2 amplitudes |
| `interactive_phase_slider.py` | Matplotlib slider widget for live PH0/PH1 phasing on a single spectrum |

## Other characterization

| Module | Purpose |
|---|---|
| `diagnostic_satellite_CQ_fit.py` | Exploratory ⁷Li CQ estimate from the static satellite transitions (first-order quadrupolar powder pattern, Monte-Carlo orientation average, I=3/2) |
| `Spectre_statique_298K_satellites_CQfit.py` | Same CQ fit turned into an article-ready figure: full +/-400 ppm static spectrum with the fitted powder pattern overlaid, exported via `agr_export.py` |
| `spectrum_static.py` | Static ⁷Li spectrum export in two views (narrow central transition, wide satellites) |
| `Analyse_XRD.py` | XRD phase-purity check: peak detection + theoretical Bragg positions for tetragonal/cubic LLZO and La2Zr2O7, to screen for high-temperature decomposition contamination (qualitative screen, not a Rietveld refinement — see the module docstring for the method's limits) |

## Example

```python
from library_nmr.pipeline_1d import process_1d_spectrum
from library_nmr.fitting import fit_group

delta, spectrum, dic = process_1d_spectrum(
    "data/7Li_LLZO_example",
    LB=10, ph0_manual=-103.4, ph1=-49.5,
    auto_ph0=True, reference_shift_ppm=2.0,
)

results = fit_group(
    delta, spectrum.real, ppm_min=-23, ppm_max=27,
    p0_list=[
        [3.5e7, 0.6, 5.8, 0.99],   # narrow (mobile Li) component
        [1.0e7, 0.6, 15,  0.5],    # broad (static Li) component
    ],
    eta_fixed_list=[None, 0.0],
    width_bounds_list=[(0, 8), (8, 100)],
    position_bounds_list=[(-4, 5), (-4, 5)],
)

for r in results:
    print(f"position={r['position']:.2f} ppm, width={r['width']:.2f} ppm, "
          f"integral={r['integral']:.3e}")
```

Or, for everyday use, each script is also meant to be run directly:
edit the `CONFIGURATION` block at the top (data path, phasing, fit
windows), then `python pipeline_1d.py`.

### Example result

Applied to ⁷Li in LLZO, this pipeline separates the resonance into two
lithium environments — a narrow, mobile component and a broad, static
one — consistent with the expected mixed-mobility picture in this
material. (Exact population split not reproduced here — part of an
unpublished manuscript; see `examples/library_nmr_demo.ipynb` for a
synthetic worked example with known ground-truth values instead.)

*(Add a plot here, e.g. `results.pdf` from `pipeline_1d.py`, showing the
spectrum with the two-component fit overlaid.)*

## Testing

```bash
pip install pytest
pytest tests/
```

`core.py`, `fitting.py` and `agr_export.py` are covered by unit tests
built on synthetic data or known ground truth (e.g. recovering injected
pseudo-Voigt parameters, a known phase error, or checking the `.agr`
output stays valid Grace format across edge-case axis ranges) — see
`tests/test_core.py`, `tests/test_fitting.py` and
`tests/test_agr_export.py`. Two further tests are regression guards for
bugs found once in this project's actual use and fixed: a `sigma=`
weighting bug in the biexponential relaxation fits
(`tests/test_relaxation_T1_sigma_weighting.py`) and a lineshape-reference
bug in the T2 per-component fit (`tests/test_relaxation_T2_components.py`).

The per-temperature/per-campaign scripts (`T1_recovery_static_*.py`,
`T2_recovery_static_*.py`, the `Check_*`/`diagnostic_*` scripts, etc.)
are not unit tested: they operate on real Bruker acquisitions that
aren't part of this repository, so there's no fixture to test them
against here. Their logic goes through the tested `core.py` /
`fitting.py` / `agr_export.py` functions above.

## Full runnable example (notebook)

See [`examples/library_nmr_demo.ipynb`](examples/library_nmr_demo.ipynb)
for a full, runnable walkthrough: from a synthetic ⁷Li FID to a
two-component pseudo-Voigt deconvolution, using the real `core.py` /
`fitting.py` functions.

## Installation

```bash
git clone https://github.com/listwanarthur/library_nmr.git
cd library_nmr
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy
scipy
matplotlib
pandas
nmrglue
```

## Structure

```
library_nmr/                             (repo root)
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── library_nmr/                         (the package)
│   ├── __init__.py
│   ├── core.py                              # shared: GRPDLY, FID processing, phasing, delay parsing
│   ├── fitting.py                           # shared: pseudo-Voigt fitting
│   ├── agr_export.py                        # shared: matplotlib series -> Grace (.agr) export
│   ├── pipeline_1d.py                       # main 1D processing + fitting
│   ├── multi_spectra_comparison.py
│   ├── check_drift.py
│   ├── relaxation_T1ir.py
│   ├── relaxation_T1ir_twocomponents.py
│   ├── relaxation_T1sr.py
│   ├── relaxation_T1sr_twocomponents.py
│   ├── relaxation_T1_onepulse_series.py
│   ├── relaxation_T1_components.py
│   ├── relaxation_T1_vs_temperature.py
│   ├── relaxation_T2.py
│   ├── relaxation_T2_echo_series.py
│   ├── relaxation_T2_components.py
│   ├── mqmas_2d_processing.py
│   ├── calibration_dft.py
│   ├── spectrum_static.py
│   ├── diagnostic_satellite_CQ_fit.py
│   ├── Spectre_statique_298K_satellites_CQfit.py
│   ├── Analyse_XRD.py
│   ├── interactive_phase_slider.py
│   ├── Check_lot_reproducibility.py
│   ├── Check_T2_298K_NS_crosscheck.py
│   ├── diagnostic_T1_ph0_scan.py
│   ├── diagnostic_T1_ph1_sidebands_scan.py
│   ├── diagnostic_T1_series_overlay.py
│   ├── diagnostic_T2_ph0_scan.py
│   ├── diagnostic_T2_fit_robustness.py
│   ├── diagnostic_T2_model_comparison.py
│   ├── T1_comparaison_VT.py
│   ├── T1_comparaison_VT_v2.py
│   ├── T2_comparaison_VT.py
│   ├── T1_global_statique_298K_peakwindow_test.py
│   ├── T1_recovery_static_298K.py           # + _298K_RG94_only, _315K, _330K, _345K, _360K
│   ├── T1_recovery_330K_v2.py
│   └── T2_recovery_static_298K.py           # + _315K, _330K, _345K, _360K
├── tests/                                (unit tests, pytest)
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_fitting.py
│   ├── test_agr_export.py
│   ├── test_relaxation_T2_components.py
│   └── test_relaxation_T1_sigma_weighting.py
└── examples/                             (runnable notebook demo)
    ├── library_nmr_demo.ipynb
    └── synthetic_7Li_fit.pdf
```

Each processing script (`relaxation_*`, `check_drift.py`,
`mqmas_2d_processing.py`) imports its shared low-level functions from
`core.py` (and `fitting.py` where relevant) rather than redefining them —
a single fix or improvement to, say, the automatic phasing routine
applies everywhere at once.

## Notes on the method

- **GRPDLY correction** is applied before any FFT — without it, Bruker's
  digital-filter delay introduces a frequency-dependent phase error that
  a simple PH0/PH1 correction cannot remove.
- **Two-component relaxation fitting** (`*_twocomponents.py`): rather
  than re-fitting 8 nonlinear lineshape parameters on every row of a
  pseudo-2D series (slow, and poorly conditioned on noisy/low-amplitude
  rows), the lineshape (position, width, eta) of each component is fixed
  once from a full nonlinear fit on the best-conditioned row (fully
  relaxed for T1, shortest echo delay for T2). Every other row then only
  solves for the two component amplitudes via ordinary linear least
  squares against those fixed shapes — fast, well-conditioned, and
  naturally allows the signed amplitudes needed for inversion recovery.
- **Drift diagnostics** (`check_drift.py`): checks peak position/intensity
  against acquisition row index (not the delay itself) to separate real
  T1/T2 physics from instrumental drift over long unlocked acquisitions.
- **T1 via onepulse D1 series** (`relaxation_T1_onepulse_series.py`): an
  alternative to the pseudo-2D T1 scripts above. Instead of a single long
  acquisition split into rows sharing one reference lineshape, each D1
  point is acquired and phased as an independent 1D spectrum. Less
  automated, but proved more robust when the shared-reference
  decomposition was unstable/drift-prone for this particular dataset —
  kept here as a documented alternative, not a replacement.
- **T2 as an independent-spectra series** (`relaxation_T2_echo_series.py`,
  `relaxation_T2_components.py`, `relaxation_T1_components.py`): same
  rationale as the T1 onepulse series above, applied to T2, plus one
  further refinement discovered while cross-checking the two components
  separately.
  - `relaxation_T2_echo_series.py` fits the *total* peak intensity vs
    echo delay with a biexponential. Two lessons learned here: PH0 is
    determined once on the best-S/N spectrum and frozen for the whole
    series (an independent per-spectrum auto-search occasionally locked
    onto noise at low S/N and silently corrupted that point), and
    intensity is extracted by trapezoidal integration over a tight ppm
    window rather than a single-point maximum (far less noise-sensitive
    once the decay approaches the detection floor).
  - `relaxation_T2_components.py` goes further: it fixes each
    component's lineshape once, then solves only the amplitudes per
    spectrum via NNLS — but critically, **the reference spectrum for
    that lineshape fit is deliberately an external onepulse spectrum,
    not an echo spectrum from the T2 series itself**. The shortest echo
    delay achievable under MAS is one full rotor period (~80 µs) — a
    rotor-synchronization constraint, not a choice — and the broad
    component's own T2 turned out to be comparable to that floor. Using
    an echo spectrum as the shape reference therefore meant the "shape"
    itself was already partly decayed, which quietly distorted the fit
    (pinned the narrow component to a pure Lorentzian, and made the
    fitted lineshapes of the two components collinear enough that NNLS
    could no longer tell them apart beyond the first couple of points).
    Swapping in a fully-relaxed onepulse spectrum as the shape reference
    — which only has hardware dead time, not a rotor-period floor —
    fixed this, and also produced each component's own T2 as
    biexponential rather than a single decay time.
  - `relaxation_T1_components.py` runs the same fixed-shape/NNLS strategy
    for T1, where the dead-time problem above doesn't apply (a onepulse
    T1 series can reference itself). Interestingly, narrow and broad come
    back with essentially the *same* T1 despite having clearly different
    T2 — consistent with T1 being driven by a few relaxation sinks
    (e.g. paramagnetic defects) that spin diffusion can average over on
    the (seconds-long) T1 timescale, while T2 reflects the local static
    coupling and stays site-specific on the much faster (microsecond)
    timescale.
- **Weighted fits for wide-dynamic-range curves**: T1 recovery and T2
  decay curves span a wide range of intensities (weak signal near the
  fast-relaxing end, near-full recovery/decay at the other). An
  unweighted least-squares fit is dominated by the large-amplitude points
  and can bias or fail to constrain the fast component — every
  biexponential `curve_fit` call in this package for such a curve is
  weighted with `sigma=` the measured intensities. This was found the
  hard way once (a ~15-18% systematic residual trend at short recovery
  times, misdiagnosed at first as an acquisition issue) and later
  regressed silently in a couple of scripts during refactoring — there's
  now a regression test (`tests/test_relaxation_T1_sigma_weighting.py`)
  that fits synthetic data with and without weighting and asserts
  weighting is required to recover the fast time constant accurately, so
  this can't reappear unnoticed.
- **Variable-temperature automation** (`relaxation_T1_vs_temperature.py`):
  once a D1-series strategy is trusted at one temperature, running it
  across many temperatures is mostly bookkeeping — this script applies
  the same model-selection rule (biexponential / monoexponential /
  two-point estimate, depending on how many points are usable) to every
  temperature in one pass, and flags automatically if the room-temperature
  point drifts too far from the previously validated reference value.

## Context

I'm a postdoctoral researcher in solid-state NMR, working on LLZO/LATP
solid electrolytes for batteries (⁶Li/⁷Li, ²⁷Al, ³¹P, ¹³⁹La). This
library grew out of my own data processing needs, and out of wanting to
actually learn Python rather than just get by with copy-pasted scripts.
I'm not a trained developer — if you spot something that could be done
better, I'd genuinely like to hear about it.

## License

MIT — shared freely, developed in an academic research context.
