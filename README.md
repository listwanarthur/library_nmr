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
| `relaxation_T2.py` | T2 via spin-echo, monoexponential fit |
| `mqmas_2d_processing.py` | 2D MQMAS (triple-quantum) processing: States F1 reconstruction, shearing, isotropic/MAS projections |
| `calibration_dft.py` | GIPAW/DFT chemical-shift calibration from reference compounds, with leave-one-out cross-validation |
| `core.py` | Shared low-level building blocks: GRPDLY handling, FID processing, automatic phasing, Bruker delay-list parsing |
| `fitting.py` | Shared pseudo-Voigt lineshape fitting (single and multi-component) |

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
one — with populations around 78%/22%, consistent with the expected
mixed-mobility picture in this material.

*(Add a plot here, e.g. `results.pdf` from `pipeline_1d.py`, showing the
spectrum with the two-component fit overlaid.)*

## Installation

```bash
git clone https://github.com/<your-username>/library_nmr.git
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
library_nmr/
├── __init__.py
├── core.py                              # shared: GRPDLY, FID processing, phasing, delay parsing
├── fitting.py                           # shared: pseudo-Voigt fitting
├── pipeline_1d.py                       # main 1D processing + fitting
├── multi_spectra_comparison.py
├── check_drift.py
├── relaxation_T1ir.py
├── relaxation_T1ir_twocomponents.py
├── relaxation_T1sr.py
├── relaxation_T1sr_twocomponents.py
├── relaxation_T2.py
├── mqmas_2d_processing.py
└── calibration_dft.py
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

## Context

I'm a postdoctoral researcher in solid-state NMR, working on LLZO/LATP
solid electrolytes for batteries (⁶Li/⁷Li, ²⁷Al, ³¹P, ¹³⁹La). This
library grew out of my own data processing needs, and out of wanting to
actually learn Python rather than just get by with copy-pasted scripts.
I'm not a trained developer — if you spot something that could be done
better, I'd genuinely like to hear about it.

## License

MIT — shared freely, developed in an academic research context.
