import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng

from library_nmr.core import find_grpdly_shift, process_row, find_best_ph0

# ============================================================
# MULTI-SPECTRUM COMPARISON — library_nmr
# Usage: edit the CONFIGURATION block below, then run.
# ============================================================

# === CONFIGURATION — only section to edit ===
SPECTRA = [
    # One dict per spectrum to overlay. Each is processed independently
    # (own GRPDLY correction, own phasing, own referencing) since different
    # experiments can have different acquisition/phase parameters.
    {
        "path": r"D:\Postdoc\Datas\LLZO-400-aug26\215",
        "label": "215 — echo, MAS 12.5kHz, D1 = 150s",
        "color": "blue",
        "LB": 10,
        "PH0_manual": -103.394,
        "PH1": -49.524,
        "auto_ph0": True,
        "reference_shift_ppm": 2.06,
    },
    {
        "path": r"D:\Postdoc\Datas\LLZO-400-aug26\209",
        "label": "209 — one-pulse, MAS 12.5kHz, D1 = 150s",
        "color": "red",
        "LB": 10,
        "PH0_manual": 12.400,
        "PH1": -49.524,
        "auto_ph0": True,
        "reference_shift_ppm": 2.06,
    },
    # Add as many entries as needed, same structure.
]

ZF_FACTOR = 1  # zero-filling: 1=none, 2=double, 4=quadruple — applied to ALL spectra

NORMALIZE = "max"  # None, "max", or "area"
    # None  : raw intensities, as acquired (only meaningful if scan counts/receiver
    #         gain are identical across spectra — otherwise heights aren't comparable)
    # "max" : each spectrum scaled so its peak height = 1 (best for comparing LINESHAPES)
    # "area": each spectrum scaled so its total integral (over ZOOM window if set,
    #         else full axis) = 1 (best for comparing RELATIVE POPULATIONS/areas)

STACK_OFFSET = 0.0  # vertical offset added between successive spectra (0 = full overlay,
                     # >0 = stacked "waterfall" style, easier to read when lines overlap a lot)

ZOOM = (30, -30)  # ppm range to display (and to use for "area" normalization if set)
OUTPUT_NAME = "spectra_comparison"
# ================================================


def process_1d_spectrum(path, LB, ph0_manual, ph1, zf_factor,
                         auto_ph0=True, reference_shift_ppm=0.0):
    """Reads a Bruker file, corrects GRPDLY, apodizes, FFTs, and phases the spectrum.
    Built on the shared core.process_row / core.find_best_ph0 (see pipeline_1d.py
    for the same pattern, with more options such as read_phase_from_procs).

    Returns the ppm axis and the real (absorptive) part of the phased spectrum.
    """
    dic, data = ng.bruker.read(path, read_procs=False)

    grpdly_shift = find_grpdly_shift(dic)
    N = data.shape[0]
    dt = 1 / dic["acqus"]["SW_h"]
    data_zf = np.concatenate([data, np.zeros(zf_factor * N, dtype=complex)])

    ph0_deg = ph0_manual
    if auto_ph0:
        signal_search = process_row(data_zf, dt, LB, 0.0, np.deg2rad(ph1), grpdly_shift=0)
        ph0_deg = find_best_ph0(signal_search, np.deg2rad(ph1))

    spectrum = process_row(data_zf, dt, LB, np.deg2rad(ph0_deg), np.deg2rad(ph1), grpdly_shift)

    f = np.fft.fftshift(np.fft.fftfreq(len(data_zf), dt))
    delta = (dic["acqus"]["O1"] - f) / dic["acqus"]["SFO1"]
    delta = delta + reference_shift_ppm

    return delta, spectrum.real


def normalize_spectrum(delta, signal, method, zoom=None):
    """Normalizes a spectrum in place according to `method` (see NORMALIZE doc above).
    If `zoom` is given, the normalization reference (max or area) is computed only
    within that window — useful to ignore noisy wings when comparing peak shapes.
    """
    if method is None:
        return signal

    if zoom is not None:
        lo, hi = min(zoom), max(zoom)
        mask = (delta >= lo) & (delta <= hi)
    else:
        mask = np.ones_like(delta, dtype=bool)

    if method == "max":
        ref = np.max(np.abs(signal[mask]))
    elif method == "area":
        sort_idx = np.argsort(delta[mask])
        ref = np.abs(np.trapezoid(signal[mask][sort_idx], delta[mask][sort_idx]))
    else:
        raise ValueError(f"Unknown NORMALIZE method: {method!r} (use None, 'max', or 'area')")

    if ref == 0:
        print("Warning: normalization reference is 0 — spectrum left unscaled")
        return signal
    return signal / ref


# === PROCESSING ===
fig, ax = plt.subplots(figsize=(10, 6))

for i, spec_cfg in enumerate(SPECTRA):
    print(f"Processing: {spec_cfg['label']}")
    delta, signal = process_1d_spectrum(
        spec_cfg["path"], spec_cfg["LB"], spec_cfg["PH0_manual"], spec_cfg["PH1"],
        ZF_FACTOR, auto_ph0=spec_cfg.get("auto_ph0", True),
        reference_shift_ppm=spec_cfg.get("reference_shift_ppm", 0.0)
    )
    signal = normalize_spectrum(delta, signal, NORMALIZE, zoom=ZOOM)
    signal = signal + i * STACK_OFFSET  # vertical stacking offset (0 = pure overlay)

    ax.plot(delta, signal, color=spec_cfg.get("color", None), linewidth=1,
            label=spec_cfg["label"])

ax.invert_xaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Chemical shift (ppm)")
ylabel = "Intensity (a.u.)"
if NORMALIZE == "max":
    ylabel = "Normalized intensity (peak = 1)"
elif NORMALIZE == "area":
    ylabel = "Normalized intensity (area = 1)"
ax.set_ylabel(ylabel)
ax.legend()
if ZOOM is not None:
    ax.set_xlim(ZOOM[0], ZOOM[1])

plt.tight_layout()
plt.savefig(f"{OUTPUT_NAME}.pdf")
plt.show()
