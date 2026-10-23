"""Helpers of scripts/build_pdf_report.py: notebook-identical number formatting, LaTeX escaping and the structural LaTeX check."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_pdf_report as bp  # noqa: E402


def test_ci_matches_notebook_format_and_registers():
    bp.CHECKS.clear()
    assert bp.ci(0.0934, 0.0642, 0.1275, 3, 1, True) == "+0.093 [+0.064, +0.128]"
    assert bp.ci(-0.0432, -0.0515, -0.0343, 2, 100, True) == "-4.32 [-5.15, -3.43]"
    assert bp.ci(0.5, float("nan"), float("nan"), 2) == "0.50"
    assert bp.ci(float("nan"), 0, 1) == ""
    bp.ci(0.1, 0.0, 0.2, reg=False)
    assert [s for _, s in bp.CHECKS] == ["+0.093 [+0.064, +0.128]", "-4.32 [-5.15, -3.43]", "0.50"]


def test_tex_escape_is_ascii_and_escapes_specials():
    s = bp.tex_escape("sf_090 ≥ 0.90: 5% & -0.5 [x] {y} M2-M1 ↓")
    assert s.isascii()
    assert r"sf\_090" in s and r"$\geq$" in s and r"5\%" in s and r"\&" in s and r"\textminus{}0.5" in s and r"\{y\}" in s
    assert "M2-M1" in s and r"$\downarrow$" in s


def test_inline_markup_and_references():
    num = {"tab:t2": "4", "fig:f6": "5"}
    h = bp.h_inline("**bold** and *it* with `a_b` see {tab:t2} and {fig:f6}; 3 * 4 stays", num)
    assert "<b>bold</b>" in h and "<i>it</i>" in h and "<code>a_b</code>" in h and "Table&nbsp;4" in h and "Figure&nbsp;5" in h and "3 * 4" in h
    t = bp.t_inline("**bold** see {tab:t2}")
    assert t == r"\textbf{bold} see Table~\ref{tab:t2}"


def test_check_tex_finds_structural_errors(tmp_path):
    (tmp_path / "tables").mkdir()
    doc = bp.Doc()
    doc.p("see {tab:x}")
    doc.table(pd.DataFrame({"a": ["1"], "b": ["2"]}), "cap", "x")
    meta = {"title": "T", "subtitle": "S", "author": "A", "date": "D", "footnote": "F"}
    (tmp_path / "main.tex").write_text(bp.render_latex(doc, meta, tmp_path, str), encoding="ascii")
    assert bp.check_tex(tmp_path) == []
    (tmp_path / "main.tex").write_text(r"\begin{document}{ \ref{nope}\input{tables/missing.tex}", encoding="ascii")
    errs = " ".join(bp.check_tex(tmp_path))
    assert "unclosed brace" in errs and "unclosed environments" in errs and "missing file" in errs and "undefined reference" in errs
