#!/usr/bin/env python3
"""Annotate a TITAN gnuplot waveform with TITAN measurement results.

Pipeline:
  1. Parse a TITAN measurement table (the "Measure Statement | Result | ..." block).
  2. Read a gnuplot-format waveform data file (whitespace-separated columns).
  3. Draw vertical bars + labels at every overshoot/undershoot location,
     sized by the measured magnitude relative to a baseline (e.g. 1.5 V).
  4. Export a vector PDF (+ PNG for AsciiDoc/pandoc-docx) and, optionally,
     a PGFPlots/TikZ snippet you can \\input into a LaTeX thesis.

Usage:
  python annotate_waveform.py \
      --waveform vreg_load_transient.dat \
      --measures measures.txt \
      --time-col 0 --signal-col 1 \
      --baseline auto \
      --out reg1v5_load_transient_annotated

Nothing here is TITAN-specific beyond the table layout; adjust the column
indices to match your gnuplot export.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Structured name schema: {cat}_{stim}_{dir}_{lo}_{hi}
#
#   cat   : ref | dv | reg | iq | os | us
#   stim  : load | line
#   dir   : dn | up | -          (- = static / not applicable)
#   lo/hi : level strings, lo <= hi  (e.g. 100n, 10m, 1v8, 2v5)
#
# Backward-compat aliases (old-style names still parse as before; the
# structured fields are simply left None so rendering falls back to
# the name-startswith heuristic).
# --------------------------------------------------------------------------- #
_SCHEMA = re.compile(
    r"^(?P<cat>ref|dv|reg|iq|os|us)"
    r"_(?P<stim>load|line)"
    r"_(?P<dir>dn|up|-)?"
    r"_(?P<lo>[^_]+)"
    r"_(?P<hi>[^_]+)$"
)


@dataclass
class Measurement:
    name: str
    result: float
    trigger: float | None
    target: float | None
    x_at_extreme: float | None = None  # filled from a following "x-value_at_MAX/MIN"
    # Structured name fields (None if old-style name)
    cat:  str | None = None   # ref | dv | reg | iq | os | us
    stim: str | None = None   # load | line
    dir:  str | None = None   # dn | up | -
    lo:   str | None = None   # lower level label
    hi:   str | None = None   # upper level label

    @property
    def is_structured(self) -> bool:
        return self.cat is not None

    @property
    def is_overshoot(self) -> bool:
        if self.is_structured:
            return self.cat == "os"
        return self.name.startswith("overshoot")

    @property
    def is_undershoot(self) -> bool:
        if self.is_structured:
            return self.cat == "us"
        return self.name.startswith("undershoot")

    @property
    def is_ref(self) -> bool:
        if self.is_structured:
            return self.cat == "ref"
        return self.name.startswith("vreg_before")

    @property
    def annotation_label(self) -> str:
        """Human-readable label for the vertical bar (ASCII-safe for .tex)."""
        kind = "OS" if self.is_overshoot else "US"
        if self.is_structured:
            direction = "dn" if self.dir == "dn" else "up"
            return f"{kind} {self.stim} {direction} {self.lo}->{self.hi}"
        return kind


@dataclass
class ParsedMeasures:
    by_name: dict[str, Measurement] = field(default_factory=dict)

    def get(self, name: str) -> Measurement | None:
        return self.by_name.get(name)

    @property
    def events(self) -> list[Measurement]:
        """Overshoot/undershoot rows that carry an x-location to annotate."""
        return [
            m
            for m in self.by_name.values()
            if m.x_at_extreme is not None and (m.is_overshoot or m.is_undershoot)
        ]

    def refs_for(self, stim: str | None, lo: str | None, hi: str | None) -> list[float]:
        """Collect reference (baseline) measurement values for a given context."""
        return [
            m.result
            for m in self.by_name.values()
            if m.is_ref
            and (stim is None or m.stim == stim)
            and (lo is None or m.lo == lo)
            and (hi is None or m.hi == hi)
        ]


_FLOAT = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
# A data row: name | result | trigger | target  (trigger/target optional)
_ROW = re.compile(
    rf"^\s*(?P<name>\S+)\s*\|\s*(?P<result>{_FLOAT})"
    rf"(?:\s*\|\s*(?P<trig>{_FLOAT}))?"
    rf"(?:\s*\|\s*(?P<targ>{_FLOAT}))?\s*$"
)


def parse_measures(text: str) -> ParsedMeasures:
    parsed = ParsedMeasures()
    last_event: Measurement | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or set(line.strip()) <= {"-"}:  # blank or separator rule
            continue
        if "Measure Statement" in line or "Trigger Point" in line:
            continue  # header

        m = _ROW.match(line)
        if not m:
            continue

        name = m.group("name")
        result = float(m.group("result"))
        trig = float(m.group("trig")) if m.group("trig") else None
        targ = float(m.group("targ")) if m.group("targ") else None

        # "x-value_at_MAX" / "x-value_at_MIN" lines attach to the previous event.
        if name.lower().startswith("x-value_at"):
            if last_event is not None:
                last_event.x_at_extreme = result
            continue

        sm = _SCHEMA.match(name)
        meas = Measurement(
            name=name, result=result, trigger=trig, target=targ,
            cat=sm.group("cat")  if sm else None,
            stim=sm.group("stim") if sm else None,
            dir=sm.group("dir")  if sm else None,
            lo=sm.group("lo")    if sm else None,
            hi=sm.group("hi")    if sm else None,
        )
        parsed.by_name[name] = meas
        last_event = meas

    return parsed


# --------------------------------------------------------------------------- #
# Reading a gnuplot data file
# --------------------------------------------------------------------------- #
def read_gnuplot(path: Path, time_col: int, signal_col: int) -> tuple[np.ndarray, np.ndarray]:
    """Read whitespace-separated columns, skipping comments/blank lines.

    Multiple datasets separated by blank lines (gnuplot "index" blocks) are
    concatenated; pass a pre-split file if you need a single block.
    """
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("%"):
            continue
        parts = s.replace(",", " ").split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # header / non-numeric line
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    width = max(len(r) for r in rows)
    if time_col >= width or signal_col >= width:
        raise IndexError(
            f"Column index out of range: file has {width} columns, "
            f"requested time={time_col}, signal={signal_col}"
        )
    arr = np.array([r for r in rows if len(r) == width], dtype=float)
    return arr[:, time_col], arr[:, signal_col]


def resolve_baseline(spec: str, measures: ParsedMeasures, signal: np.ndarray) -> float:
    if spec == "auto":
        # Prefer structured ref_* measurements; fall back to old-style vreg_before_*.
        refs = [m.result for m in measures.by_name.values() if m.is_ref]
        return float(np.mean(refs)) if refs else float(signal[0])
    return float(spec)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def annotate(
    time: np.ndarray,
    signal: np.ndarray,
    measures: ParsedMeasures,
    baseline: float,
    out_stem: Path,
    *,
    use_pgf: bool,
    signal_label: str,
    time_unit: str,
    volt_unit: str,
) -> None:
    if use_pgf:
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(time, signal, lw=1.0, color="tab:blue", label=signal_label)
    ax.axhline(baseline, ls=":", lw=0.8, color="0.4")

    for ev in measures.events:
        x = ev.x_at_extreme
        mag = ev.result
        is_over = ev.is_overshoot
        y_extreme = baseline + mag if is_over else baseline - mag

        # Shade the trigger->target measurement window.
        if ev.trigger is not None and ev.target is not None:
            ax.axvspan(ev.trigger, ev.target, color="0.92", zorder=0)

        # Vertical bar from baseline to the extreme value.
        ax.annotate(
            "",
            xy=(x, y_extreme),
            xytext=(x, baseline),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.1),
        )
        label = f"{ev.annotation_label}\n{mag * 1e3:.1f} m{volt_unit}"
        ax.text(
            x,
            y_extreme + (0.02 * mag if is_over else -0.02 * mag),
            label,
            ha="center",
            va="bottom" if is_over else "top",
            fontsize=7,
            color="crimson",
        )

    ax.set_xlabel(f"time ({time_unit})")
    ax.set_ylabel(f"{signal_label} ({volt_unit})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.4)
    fig.tight_layout()

    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)  # for AsciiDoc/docx
    print(f"Wrote {out_stem.with_suffix('.pdf')} and {out_stem.with_suffix('.png')}")


def emit_pgfplots(
    measures: ParsedMeasures, baseline: float, out_stem: Path, waveform_rel: str
) -> None:
    """Emit a \\input-able PGFPlots snippet (annotations as \\draw/\\node)."""
    lines = [
        "% Auto-generated by annotate_waveform.py -- do not edit by hand.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[xlabel={time (s)}, ylabel={V}, grid=both, width=12cm, height=7cm]",
        f"  \\addplot[blue, thick] table {{{waveform_rel}}};",
        f"  \\draw[dotted] (axis cs:\\pgfkeysvalueof{{/pgfplots/xmin}},{baseline})"
        f" -- (axis cs:\\pgfkeysvalueof{{/pgfplots/xmax}},{baseline});",
    ]
    for ev in measures.events:
        x = ev.x_at_extreme
        is_over = ev.is_overshoot
        y = baseline + ev.result if is_over else baseline - ev.result
        label = f"{ev.annotation_label} {ev.result * 1e3:.1f}\\,mV"
        lines.append(
            f"  \\draw[<->, red, thick] (axis cs:{x},{baseline})"
            f" -- (axis cs:{x},{y}) node[midway, right, font=\\footnotesize]"
            f" {{{label}}};"
        )
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    tex = out_stem.with_suffix(".tex")
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {tex}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--waveform", required=True, type=Path, help="gnuplot .dat file")
    p.add_argument("--measures", required=True, type=Path, help="TITAN measurement table")
    p.add_argument("--time-col", type=int, default=0)
    p.add_argument("--signal-col", type=int, default=1)
    p.add_argument("--baseline", default="auto", help='"auto" or a number, e.g. 1.5')
    p.add_argument("--signal-label", default="vreg")
    p.add_argument("--time-unit", default="s")
    p.add_argument("--volt-unit", default="V")
    p.add_argument("--out", required=True, type=Path, help="output stem (no extension)")
    p.add_argument("--pgf", action="store_true", help="render via LaTeX pgf backend")
    p.add_argument("--emit-pgfplots", action="store_true", help="also write a PGFPlots .tex snippet")
    args = p.parse_args()

    measures = parse_measures(args.measures.read_text())
    time, signal = read_gnuplot(args.waveform, args.time_col, args.signal_col)
    baseline = resolve_baseline(args.baseline, measures, signal)

    annotate(
        time,
        signal,
        measures,
        baseline,
        args.out,
        use_pgf=args.pgf,
        signal_label=args.signal_label,
        time_unit=args.time_unit,
        volt_unit=args.volt_unit,
    )
    if args.emit_pgfplots:
        emit_pgfplots(measures, baseline, args.out, args.waveform.as_posix())


if __name__ == "__main__":
    main()
