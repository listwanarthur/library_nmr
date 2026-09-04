"""
Tests for library_nmr.agr_export -- writes matplotlib-style series to a
Grace (.agr) project file.

This module became a shared dependency of nearly every figure-producing
script added in the 315K/330K/345K/360K campaigns (T1/T2 recovery scripts,
VT comparisons, static spectra, satellite CQ fit) -- a silent formatting
bug here would show up as a broken or truncated figure in every one of
them, discovered only when reopening the .agr in Xmgrace. Covered here
with the same synthetic/known-behavior strategy as test_fitting.py, plus
a regression test for the tick-spacing bug documented in _tick_lines's
own docstring (found 26/08, autotick failure on wide/irregular ranges).
"""

import re

import numpy as np
import pytest

from library_nmr.agr_export import (
    _ascii_safe,
    _axis_range,
    _nice_step,
    _tick_lines,
    export_agr,
)


# --- _ascii_safe ------------------------------------------------------

def test_ascii_safe_converts_known_special_characters():
    """Table-mapped characters (degree sign, superscripts, mu, dashes,
    curly quotes) go to their explicit ASCII equivalents, not just
    whatever NFKD decomposition would produce."""
    text = "25°C, R², µs, T1_slow — T1_fast, ‘cf.’ note"
    result = _ascii_safe(text)
    assert result == "25degC, R2, us, T1_slow - T1_fast, 'cf.' note"
    assert result.isascii()


def test_ascii_safe_strips_other_non_ascii_via_nfkd():
    """Anything not in the explicit table (e.g. French accents) falls
    back to NFKD normalization + ASCII-encode-ignore, so it degrades
    gracefully instead of corrupting the .agr parser."""
    result = _ascii_safe("Résumé du modèle")
    assert result.isascii()
    assert result == "Resume du modele"


def test_ascii_safe_empty_string_passthrough():
    assert _ascii_safe("") == ""
    assert _ascii_safe(None) is None


# --- _axis_range --------------------------------------------------------

def test_axis_range_linear_adds_symmetric_padding():
    lo, hi = _axis_range([1, 2, 3, 4, 10], log=False, pad=0.05)
    span = 10 - 1
    assert lo == pytest.approx(1 - 0.05 * span)
    assert hi == pytest.approx(10 + 0.05 * span)


def test_axis_range_log_scale_stays_strictly_positive():
    """A log axis with a negative or zero bound is meaningless for Grace
    (and for np.log10 downstream) -- this must hold even after padding
    is applied, across a realistic multi-decade D1 range like the ones
    used in the T1 recovery scripts (0.02s to 500s)."""
    lo, hi = _axis_range([0.02, 0.05, 1, 10, 100, 500], log=True)
    assert lo > 0
    assert hi > 0
    assert lo < 0.02
    assert hi > 500


def test_axis_range_log_scale_ignores_non_positive_values():
    """Non-positive values can't sit on a log axis and must be filtered
    out rather than crashing np.log10 or silently going through as NaN."""
    lo, hi = _axis_range([-5, 0, 1, 10, 100], log=True)
    assert lo > 0
    assert lo < 1
    assert hi > 100


def test_axis_range_handles_degenerate_zero_span():
    """All-identical values (e.g. a single-point series) must not produce
    a zero-width or NaN range."""
    lo, hi = _axis_range([5, 5, 5], log=False)
    assert lo < 5 < hi
    lo0, hi0 = _axis_range([0, 0, 0], log=False)
    assert lo0 < 0 < hi0


# --- _nice_step -----------------------------------------------------------

@pytest.mark.parametrize("span", [47, 23, 8, 2.3, 0.00047, 913])
def test_nice_step_returns_1_2_5_times_a_power_of_ten(span):
    """Tick spacing must always be a 'nice' 1/2/5 x 10^n value, whatever
    the input span -- otherwise Grace's ticklabels come out ugly
    (e.g. 3.333, 3.333, 3.333 instead of 5, 10, 15)."""
    step = _nice_step(span)
    exp = np.floor(np.log10(step))
    mantissa = round(step / (10 ** exp), 6)
    assert mantissa in (1.0, 2.0, 5.0, 10.0)


def test_nice_step_zero_span_fallback():
    assert _nice_step(0) == 1.0


# --- _tick_lines: regression test for the 26/08 autotick bug --------------

@pytest.mark.parametrize("lo,hi,log", [
    (1, 100000, True),      # 5-decade log span, like a T1 D1 axis
    (0.3, 17.9, False),     # non-round linear span
    (1e-6, 1e-6 * 1.0001, False),  # near-zero span
])
def test_tick_lines_always_explicit_never_relies_on_autotick(lo, hi, log):
    """FIX (26/08, see _tick_lines docstring): Grace's own autotick
    computation can fail ('Invalid major tick spacing' / 'Too many ticks')
    on an unusual range and silently truncate the render. This asserts
    _tick_lines always emits an explicit, well-formed tick major/minor
    spacing regardless of the input range, so that bug can't reappear
    unnoticed."""
    lines = _tick_lines("xaxis", lo, hi, log)
    joined = "\n".join(lines)
    assert "tick on" in joined
    assert re.search(r"tick major [\d.]+", joined)
    assert re.search(r"tick minor ticks \d+", joined)
    major = float(re.search(r"tick major ([\d.]+)", joined).group(1))
    assert major > 0


# --- export_agr: end-to-end file structure ---------------------------------

def _write_and_read(tmp_path, **kwargs):
    path = tmp_path / "out.agr"
    export_agr(str(path), **kwargs)
    return path.read_text(encoding="utf-8")


def test_export_agr_writes_one_target_block_per_series(tmp_path):
    x1, y1 = np.array([1., 2., 3.]), np.array([10., 20., 15.])
    x2, y2 = np.array([1., 3.]), np.array([5., 7.])
    content = _write_and_read(
        tmp_path,
        series=[
            dict(x=x1, y=y1, mode="line", color="blue", legend="fit"),
            dict(x=x2, y=y2, mode="symbol", color="red", legend="data"),
        ],
        xlabel="x", ylabel="y",
    )
    assert "@target G0.S0" in content
    assert "@target G0.S1" in content
    # each block's data lines should match the number of input points
    block0 = content.split("@target G0.S0")[1].split("@target G0.S1")[0]
    data_lines0 = [l for l in block0.splitlines() if l and l not in ("@type xy", "&")]
    assert len(data_lines0) == len(x1)
    block1 = content.split("@target G0.S1")[1]
    data_lines1 = [l for l in block1.splitlines() if l and l not in ("@type xy", "&")]
    assert len(data_lines1) == len(x2)


def test_export_agr_sanitizes_non_ascii_in_title_and_legend(tmp_path):
    """The whole point of _ascii_safe: title/legend text with accents,
    degree signs, or Greek mu must not leak non-ASCII bytes into the
    .agr file, or Grace's parser desyncs on that line and drops
    everything after it."""
    content = _write_and_read(
        tmp_path,
        series=[dict(x=[1, 2], y=[1, 2], legend="fit 1 (25°C, R² = 0.98)")],
        xlabel="D1 (s)", ylabel="I (a.u.)", title="Résumé — 7Li T1",
    )
    assert content.isascii()
    assert "Resume" in content
    assert "25degC" in content
    assert "R2 = 0.98" in content


def test_export_agr_world_override_used_verbatim(tmp_path):
    """world is documented as (xmin, xmax, ymin, ymax); the file itself
    writes the "@world" line as xmin, ymin, xmax, ymax (Grace's own
    convention) -- this checks the override values survive that reorder
    intact rather than getting silently swapped or re-derived from data."""
    content = _write_and_read(
        tmp_path,
        series=[dict(x=[1, 2, 3], y=[1, 2, 3])],
        xlabel="x", ylabel="y", world=(0, 10, -5, 5),  # xmin, xmax, ymin, ymax
    )
    m = re.search(r"@\s+world ([\d.-]+), ([\d.-]+), ([\d.-]+), ([\d.-]+)", content)
    xmin, ymin, xmax, ymax = [float(v) for v in m.groups()]  # file's own order
    assert (xmin, xmax, ymin, ymax) == (0.0, 10.0, -5.0, 5.0)


def test_export_agr_xlim_overrides_x_only(tmp_path):
    """xlim should override the x view without touching the
    auto-computed y range (unlike `world`, which overrides both)."""
    y = [1., 2., 3., 100.]
    content = _write_and_read(
        tmp_path,
        series=[dict(x=[1., 2., 3., 4.], y=y)],
        xlabel="x", ylabel="y", xlim=(1.5, 3.5),
    )
    m = re.search(r"@\s+world ([\d.-]+), ([\d.-]+), ([\d.-]+), ([\d.-]+)", content)
    xmin, ymin, xmax, ymax = [float(v) for v in m.groups()]
    assert xmin == 1.5 and xmax == 3.5
    # y range is still auto-padded from the full data (includes the 100)
    assert ymax > 100


def test_export_agr_symbol_vs_line_mode_directives(tmp_path):
    content = _write_and_read(
        tmp_path,
        series=[
            dict(x=[1, 2], y=[1, 2], mode="line"),
            dict(x=[1, 2], y=[1, 2], mode="symbol"),
        ],
        xlabel="x", ylabel="y",
    )
    s0_lines = "\n".join(l for l in content.splitlines() if l.startswith("@    s0 "))
    s1_lines = "\n".join(l for l in content.splitlines() if l.startswith("@    s1 "))
    assert "line type 1" in s0_lines and "symbol 0" in s0_lines
    assert "symbol 1" in s1_lines and "line type 0" in s1_lines
