"""
agr_export.py — writes a matplotlib-style line/scatter plot directly to a
Grace (.agr) project file, so it can be reopened and polished in Xmgrace
instead of being rebuilt from scratch.

Usage
-----
    from library_nmr.agr_export import export_agr

    export_agr(
        "results.agr",
        series=[
            dict(x=delta, y=signal, mode="line", color="blue", legend="spectrum"),
            dict(x=r["delta_peak"], y=fit_curve, mode="line", color="red",
                 legend="fit 1 (41.3%)"),
        ],
        xlabel="Chemical shift (ppm)",
        ylabel="Intensity (a.u.)",
        invert_x=True,
    )

Each entry in `series` mirrors one matplotlib ax.plot()/ax.scatter() call:
    x, y     : 1D arrays (required)
    mode     : "line" (default) or "symbol" (markers, no connecting line)
    color    : matplotlib-style color name — mapped to a Grace color index
    legend   : legend text (omit for no legend entry)
    linewidth, symsize : optional overrides of the defaults
"""

import unicodedata

import numpy as np

_ASCII_REPLACEMENTS = {
    "°": "deg", "²": "2", "³": "3", "μ": "u", "µ": "u",
    "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"',
}


def _ascii_safe(text):
    """Grace's .agr parser chokes on non-ASCII bytes inside quoted strings
    (accented letters, superscripts, Greek mu, em-dashes, ...) and silently
    desyncs the parser for several following lines. Titles/labels/legends
    are sanitized to plain ASCII here so scripts can keep using proper
    typography (e.g. French labels, "R²") in matplotlib without breaking
    the .agr export."""
    if not text:
        return text
    for src, dst in _ASCII_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


_PALETTE = {
    0: ((255, 255, 255), "white"), 1: ((0, 0, 0), "black"),
    2: ((255, 0, 0), "red"), 3: ((0, 150, 0), "green"),
    4: ((0, 0, 255), "blue"), 5: ((210, 180, 0), "yellow"),
    6: ((139, 69, 19), "brown"), 7: ((120, 120, 120), "grey"),
    8: ((128, 0, 128), "purple"), 9: ((0, 180, 180), "cyan"),
    10: ((200, 0, 200), "magenta"), 11: ((255, 140, 0), "orange"),
    12: ((0, 0, 128), "navy"), 13: ((128, 128, 0), "olive"),
    14: ((255, 105, 180), "pink"), 15: ((0, 128, 128), "teal"),
}
_NAME_TO_INDEX = {
    "white": 0, "black": 1, "red": 2, "green": 3, "blue": 4, "yellow": 5,
    "brown": 6, "grey": 7, "gray": 7, "purple": 8, "cyan": 9, "magenta": 10,
    "orange": 11, "darkorange": 11, "navy": 12, "olive": 13, "pink": 14,
    "teal": 15,
}


def _color_index(name):
    return _NAME_TO_INDEX.get(str(name).lower(), 1)


def _header():
    lines = [
        "# Grace project file", "#", "@version 50122",
        "@page size 1000, 750", "@page scroll 5%", "@page inout 5%",
    ]
    for idx, (rgb, name) in _PALETTE.items():
        lines.append(f'@map color {idx} to ({rgb[0]}, {rgb[1]}, {rgb[2]}), "{name}"')
    lines += [
        "@background color 0", "@default linewidth 1.5",
        "@default char size 1.000000", "@default font 4", "",
    ]
    return "\n".join(lines)


def _axis_range(values, log=False, pad=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if log:
        values = values[values > 0]
    lo, hi = float(np.min(values)), float(np.max(values))
    if log:
        lo_log, hi_log = np.log10(lo), np.log10(hi)
        span = (hi_log - lo_log) or 1.0
        return 10 ** (lo_log - pad * span), 10 ** (hi_log + pad * span)
    span = (hi - lo) or (abs(hi) or 1.0)
    return lo - pad * span, hi + pad * span


def export_agr(path, series, xlabel, ylabel, title="", xlog=False, ylog=False,
                invert_x=False, invert_y=False, legend=True,
                legend_pos=(0.78, 0.85), world=None, xlim=None):
    """Writes a single-panel Grace project file from matplotlib-style series.

    world: optional explicit (xmin, xmax, ymin, ymax) override — computed
    automatically from the data (with a 5% pad) if omitted.
    xlim: optional (xmin, xmax) view-only override (mirrors ax.set_xlim) —
    the y-range is still auto-computed from the full data, not just the
    zoomed window. Ignored if `world` is given.
    """
    all_x = np.concatenate([np.asarray(s["x"], dtype=float) for s in series])
    all_y = np.concatenate([np.asarray(s["y"], dtype=float) for s in series])
    if world is None:
        xmin, xmax = _axis_range(all_x, log=xlog)
        ymin, ymax = _axis_range(all_y, log=ylog)
        if xlim is not None:
            xmin, xmax = min(xlim), max(xlim)
    else:
        xmin, xmax, ymin, ymax = world

    title = _ascii_safe(title)
    xlabel = _ascii_safe(xlabel)
    ylabel = _ascii_safe(ylabel)

    w = ["@g0 on", "@with g0"]
    w.append(f"@    world {xmin:.6g}, {ymin:.6g}, {xmax:.6g}, {ymax:.6g}")
    w.append("@    view 0.150000, 0.150000, 1.150000, 0.850000")
    w.append(f'@    title "{title}"')
    w.append(f"@    xaxes scale {'Logarithmic' if xlog else 'Normal'}")
    w.append(f"@    yaxes scale {'Logarithmic' if ylog else 'Normal'}")
    w.append(f"@    xaxes invert {'on' if invert_x else 'off'}")
    w.append(f"@    yaxes invert {'on' if invert_y else 'off'}")
    w.append(f'@    xaxis  label "{xlabel}"')
    w.append(f'@    yaxis  label "{ylabel}"')
    w.append("@    xaxis  label char size 1.100000")
    w.append("@    yaxis  label char size 1.100000")
    w.append("@    xaxis  ticklabel char size 1.000000")
    w.append("@    yaxis  ticklabel char size 1.000000")
    w.append(f"@    legend {'on' if legend else 'off'}")
    w.append(f"@    legend {legend_pos[0]:.3f}, {legend_pos[1]:.3f}")
    w.append("@    legend box linestyle 0")
    w.append("@    frame linewidth 1.5")

    for i, s in enumerate(series):
        color = _color_index(s.get("color", "black"))
        mode = s.get("mode", "line")
        lw = s.get("linewidth", 2.0)
        symsize = s.get("symsize", 0.6)
        w.append(f"@    s{i} hidden false")
        w.append(f"@    s{i} line color {color}")
        w.append(f"@    s{i} symbol color {color}")
        if mode == "symbol":
            w.append(f"@    s{i} symbol 1")
            w.append(f"@    s{i} symbol size {symsize}")
            w.append(f"@    s{i} symbol linewidth 1.2")
            w.append(f"@    s{i} symbol fill pattern 1")
            w.append(f"@    s{i} symbol fill color {color}")
            w.append(f"@    s{i} line type 0")
        else:
            w.append(f"@    s{i} symbol 0")
            w.append(f"@    s{i} line type 1")
            w.append(f"@    s{i} line linewidth {lw}")
        if s.get("legend"):
            w.append(f'@    s{i} legend "{_ascii_safe(s["legend"])}"')

    out = "\n".join(w) + "\n"
    for i, s in enumerate(series):
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        block = [f"@target G0.S{i}", "@type xy"]
        block += [f"{xi:.8g} {yi:.8g}" for xi, yi in zip(x, y)]
        block.append("&")
        out += "\n".join(block) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(_header())
        f.write(out)
