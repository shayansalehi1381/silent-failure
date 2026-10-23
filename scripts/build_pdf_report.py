"""Builds the final PDF report (and its LaTeX source) from saved, committed results only.

python scripts/build_pdf_report.py --config configs/full.yaml

Every number in the text and tables is computed here from the small committed files under outputs/ (and git / configs for design
constants); nothing is typed by hand. One document model is rendered twice: to HTML (printed to PDF with headless Chrome / Edge, since
no LaTeX toolchain is assumed) and to LaTeX (main.tex + tables/ + figures/, shipped as a zip for pdflatex / latexmk / Overleaf).
After building, a consistency check requires every headline number to appear verbatim in the executed notebook's outputs
(silent_failure_report.ipynb) and fails loudly otherwise. Outputs go to report.export_dir (outside the repository, never committed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NB = ROOT / "silent_failure_report.ipynb"

# ============================================================================================ number formatting (as in the notebook)
CHECKS: list[tuple[str, str]] = []             # (what, string that must appear in the executed notebook's outputs)


def ci(e, lo, hi, d=3, scale=1.0, signed=False, reg=True) -> str:
    """'estimate [low, high]' exactly as the notebook formats it; registered for the consistency check unless reg=False."""
    if pd.isna(e):
        return ""
    f = f"{{:{'+' if signed else ''}.{d}f}}"
    s = f.format(scale * e)
    s = s if pd.isna(lo) else f"{s} [{f.format(scale * lo)}, {f.format(scale * hi)}]"
    if reg:
        CHECKS.append(("CI", s))
    return s


def rc(r, d=3, scale=1.0, signed=True, est="estimate", reg=True) -> str:
    return ci(r[est], r["ci_low"], r["ci_high"], d, scale, signed, reg)


def must(s: str, what: str = "phrase") -> str:
    """Register a phrase that the notebook prints verbatim (counts, point estimates in context)."""
    CHECKS.append((what, s))
    return s


def one(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= np.isclose(df[k].astype(float), v) if isinstance(v, float) else (df[k].astype(str) == str(v))
    r = df[m]
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


def rd(rel: str):
    p = OUT / rel
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return p.read_text(encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout.strip()


# ============================================================================================ document model
class Doc:
    """Blocks: ('h', level, text, label), ('p', text), ('ul', items), ('table', spec), ('fig', spec), ('raw', kind), ('break',)."""

    def __init__(self):
        self.blocks: list[tuple] = []

    def h(self, level, text, label=None):
        self.blocks.append(("h", level, text, label))

    def p(self, text):
        self.blocks.append(("p", text))

    def ul(self, items, ordered=False):
        self.blocks.append(("ul", items, ordered))

    def table(self, df: pd.DataFrame, caption: str, label: str, align: str | None = None, size: str = "small", long: bool = False):
        self.blocks.append(("table", dict(df=df.reset_index(drop=True), caption=caption, label=label, align=align, size=size, long=long)))

    def fig(self, path: Path, caption: str, label: str, width: float = 1.0):
        assert path.exists(), path
        self.blocks.append(("fig", dict(path=path, caption=caption, label=label, width=width)))

    def brk(self):
        self.blocks.append(("break",))

    def raw(self, kind):
        self.blocks.append(("raw", kind))

    def numbering(self):
        """Labels -> displayed numbers (sections 1.., appendix A, tables / figures continuous), as LaTeX numbers them."""
        num, sec, sub, tab, fig, app = {}, 0, 0, 0, 0, False
        for b in self.blocks:
            if b[0] == "raw" and b[1] == "appendix":
                app, sec = True, 0
            if b[0] == "h" and b[1] == 1 and b[3] is not None:
                sec += 1; sub = 0
                num["sec:" + b[3]] = chr(64 + sec) if app else str(sec)
            elif b[0] == "h" and b[1] == 2 and b[3] is not None:
                sub += 1
                num["sec:" + b[3]] = f"{chr(64 + sec) if app else sec}.{sub}"
            elif b[0] == "table":
                tab += 1; num["tab:" + b[1]["label"]] = str(tab)
            elif b[0] == "fig":
                fig += 1; num["fig:" + b[1]["label"]] = str(fig)
        return num


REF = re.compile(r"\{(tab|fig|sec):([\w-]+)\}")
TOK = re.compile(r"(\*\*.+?\*\*|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*])|`[^`]+`|\{(?:tab|fig|sec):[\w-]+\})")
REFWORD = {"tab": "Table", "fig": "Figure", "sec": "Section"}


# ---------------------------------------------------------------------------------------------------------------- HTML
def h_inline(s: str, num: dict) -> str:
    out = []
    for t in TOK.split(s):
        if not t:
            continue
        if t.startswith("**"):
            out.append(f"<b>{h_inline(t[2:-2], num)}</b>")
        elif t.startswith("`"):
            out.append(f"<code>{html.escape(t[1:-1])}</code>")
        elif REF.fullmatch(t):
            k, lab = REF.fullmatch(t).groups()
            out.append(f'<a href="#{k}-{lab}">{REFWORD[k]}&nbsp;{num[f"{k}:{lab}"]}</a>')
        elif t.startswith("*") and t.endswith("*") and len(t) > 2:
            out.append(f"<i>{h_inline(t[1:-1], num)}</i>")
        else:
            out.append(html.escape(t, quote=False))
    return "".join(out)


CSS = """
@page { size: A4; margin: 16mm 16mm 17mm 16mm; @bottom-center { content: counter(page); font: 8.5pt 'Cambria', Georgia, serif; color: #555; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Cambria', 'Georgia', serif; font-size: 10.2pt; line-height: 1.36; color: #111; background: #fff; margin: 0; }
h1 { font-size: 14.5pt; margin: 15pt 0 5pt; padding-bottom: 2pt; border-bottom: 0.8pt solid #333; break-after: avoid; }
h2 { font-size: 11.6pt; margin: 11pt 0 3pt; break-after: avoid; }
p { margin: 0 0 5pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 6pt 0; padding-left: 16pt; } li { margin-bottom: 2.5pt; text-align: justify; }
code { font-family: 'Consolas', monospace; font-size: 8.8pt; }
a { color: #0b3d91; text-decoration: none; }
.tbl { margin: 8pt 0 10pt; break-inside: avoid; } .tbl.long { break-inside: auto; }
.cap { font-size: 8.7pt; line-height: 1.3; margin: 0 0 3pt; text-align: left; } .cap b { font-weight: bold; }
table { border-collapse: collapse; width: 100%; line-height: 1.22; }
table.small { font-size: 7.9pt; } table.footnotesize { font-size: 7.3pt; } table.scriptsize { font-size: 6.7pt; }
thead th { text-align: left; font-weight: bold; border-top: 1.1pt solid #000; border-bottom: 0.6pt solid #000; padding: 2.2pt 3pt; vertical-align: bottom; }
tbody td { padding: 1.8pt 3pt; vertical-align: top; } tbody tr:last-child td { border-bottom: 1.1pt solid #000; }
thead { display: table-header-group; } tr { break-inside: avoid; }
figure { margin: 8pt 0 10pt; break-inside: avoid; text-align: center; } figure img { max-width: 100%; }
figcaption { font-size: 8.7pt; line-height: 1.3; text-align: left; margin-top: 3pt; }
.title { height: 245mm; display: flex; flex-direction: column; justify-content: center; text-align: center; break-after: page; }
.title .t { font-size: 22pt; font-weight: bold; line-height: 1.25; margin-bottom: 12pt; }
.title .s { font-size: 12.5pt; font-style: italic; margin: 0 20mm 26pt; line-height: 1.35; }
.title .a { font-size: 13pt; margin-bottom: 5pt; } .title .d { font-size: 11pt; color: #333; }
.title .m { font-size: 8.8pt; color: #444; margin-top: 30pt; line-height: 1.5; }
.pb { break-after: page; height: 0; }
.exec h1 { margin-top: 0; } .exec p, .exec li { font-size: 9.7pt; line-height: 1.32; }
"""


def render_html(doc: Doc, meta: dict, fig_rel: callable) -> str:
    num = doc.numbering()
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(meta['title'])}</title><style>{CSS}</style></head><body>"]
    for b in doc.blocks:
        k = b[0]
        if k == "raw" and b[1] == "title":
            out.append(f"<div class='title'><div class='t'>{html.escape(meta['title'])}</div><div class='s'>{html.escape(meta['subtitle'])}</div>"
                       f"<div class='a'>{html.escape(meta['author'])}</div><div class='d'>{html.escape(meta['date'])}</div>"
                       f"<div class='m'>{h_inline(meta['footnote'], num)}</div></div>")
        elif k == "raw" and b[1] == "exec_begin":
            out.append("<div class='exec'>")
        elif k == "raw" and b[1] == "exec_end":
            out.append("</div><div class='pb'></div>")
        elif k == "raw" and b[1] == "appendix":
            out.append("<div class='pb'></div>")
        elif k == "break":
            out.append("<div class='pb'></div>")
        elif k == "h":
            _, lvl, text, lab = b
            pre = f"{'Appendix ' if (lab or '').startswith('app') and lvl == 1 else ''}{num[f'sec:{lab}']} " if lab else ""
            out.append(f"<h{lvl} id='sec-{lab}'>{pre}{h_inline(text, num)}</h{lvl}>")
        elif k == "p":
            out.append(f"<p>{h_inline(b[1], num)}</p>")
        elif k == "ul":
            tag = "ol" if b[2] else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{h_inline(i, num)}</li>" for i in b[1]) + f"</{tag}>")
        elif k == "table":
            s = b[1]
            n = num[f"tab:{s['label']}"]
            rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in s["df"].astype(str).itertuples(index=False))
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in s["df"].columns)
            out.append(f"<div class='tbl{' long' if s['long'] else ''}' id='tab-{s['label']}'><div class='cap'><b>Table {n}.</b> {h_inline(s['caption'], num)}</div>"
                       f"<table class='{s['size']}'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>")
        elif k == "fig":
            s = b[1]
            n = num[f"fig:{s['label']}"]
            out.append(f"<figure id='fig-{s['label']}'><img src='{fig_rel(s['path'])}' style='width:{100 * s['width']:.0f}%'>"
                       f"<figcaption><b>Figure {n}.</b> {h_inline(s['caption'], num)}</figcaption></figure>")
    out.append("</body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------- LaTeX
UNI = {"≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "→": r"$\rightarrow$", "±": r"$\pm$", "≈": r"$\approx$", "–": "--", "—": "---",
       "↓": r"$\downarrow$", "↑": r"$\uparrow$", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}", "°": r"$^\circ$", "\u00a0": "~", "−": r"\textminus{}"}
TEXSPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "^": r"\textasciicircum{}",
              "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}", "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}


def tex_escape(s: str) -> str:
    s = "".join(TEXSPECIAL.get(c, c) for c in s)
    s = re.sub(r"(?<![A-Za-z0-9])-(?=\d)", r"\\textminus{}", s)          # numeric minus signs
    s = "".join(UNI.get(c, c) for c in s)
    bad = [c for c in s if ord(c) > 127]
    assert not bad, f"unmapped non-ASCII characters for LaTeX: {sorted(set(bad))}"
    return s


def t_inline(s: str) -> str:
    out = []
    for t in TOK.split(s):
        if not t:
            continue
        if t.startswith("**"):
            out.append(r"\textbf{" + t_inline(t[2:-2]) + "}")
        elif t.startswith("`"):
            out.append(r"\texttt{" + tex_escape(t[1:-1]) + "}")
        elif REF.fullmatch(t):
            k, lab = REF.fullmatch(t).groups()
            out.append(f"{REFWORD[k]}~\\ref{{{k}:{lab}}}")
        elif t.startswith("*") and t.endswith("*") and len(t) > 2:
            out.append(r"\emph{" + t_inline(t[1:-1]) + "}")
        else:
            out.append(tex_escape(t))
    return "".join(out)


def tex_table(s: dict) -> str:
    df = s["df"]
    align = s["align"] or ("l" + "Y" * (df.shape[1] - 1))
    assert len(re.findall(r"p\{[^}]*\}|[lrcXY]", align)) == df.shape[1], (s["label"], align, df.shape)
    head = " & ".join(r"\textbf{" + tex_escape(str(c)) + "}" for c in df.columns) + r" \\"
    body = "\n".join(" & ".join(tex_escape(str(v)) for v in r) + r" \\" for r in df.astype(str).itertuples(index=False))
    cap = t_inline(s["caption"])
    if s["long"]:
        return (f"{{\\{s['size']}\n\\begin{{xltabular}}{{\\linewidth}}{{{align}}}\n\\caption{{{cap}}}\\label{{tab:{s['label']}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
                f"\\toprule\n{head}\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot\n{body}\n\\end{{xltabular}}\n}}\n")
    return (f"\\begin{{table}}[htbp]\n\\centering\n\\{s['size']}\n\\caption{{{cap}}}\\label{{tab:{s['label']}}}\n"
            f"\\begin{{tabularx}}{{\\linewidth}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabularx}}\n\\end{{table}}\n")


PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage[margin=17mm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{xltabular}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\renewcommand{\textfraction}{0.1}
\renewcommand{\topfraction}{0.9}
\renewcommand{\floatpagefraction}{0.8}
"""


def render_latex(doc: Doc, meta: dict, tex_dir: Path, fig_name: callable) -> str:
    (tex_dir / "tables").mkdir(parents=True, exist_ok=True)
    out = [PREAMBLE, "\\title{" + tex_escape(meta["title"]) + "}", "\\author{" + tex_escape(meta["author"]) + "}", "\\date{" + tex_escape(meta["date"]) + "}",
           "\\begin{document}"]
    for b in doc.blocks:
        k = b[0]
        if k == "raw" and b[1] == "title":
            out.append("\\begin{titlepage}\n\\centering\n\\vspace*{55mm}\n{\\LARGE\\bfseries " + tex_escape(meta["title"]) + "\\par}\n\\vspace{10mm}\n"
                       "{\\large\\itshape " + tex_escape(meta["subtitle"]) + "\\par}\n\\vspace{18mm}\n{\\large " + tex_escape(meta["author"]) + "\\par}\n"
                       "\\vspace{3mm}\n{" + tex_escape(meta["date"]) + "\\par}\n\\vfill\n{\\small " + t_inline(meta["footnote"]) + "\\par}\n\\end{titlepage}")
        elif k == "raw" and b[1] == "exec_begin":
            out.append("{\\small")
        elif k == "raw" and b[1] == "exec_end":
            out.append("}\n\\clearpage")
        elif k == "raw" and b[1] == "appendix":
            out.append("\\clearpage\n\\appendix")
        elif k == "break":
            out.append("\\clearpage")
        elif k == "h":
            _, lvl, text, lab = b
            cmd = "section" if lvl == 1 else "subsection"
            star = "*" if lab is None else ""
            out.append(f"\\{cmd}{star}{{{t_inline(text)}}}" + (f"\\label{{sec:{lab}}}" if lab else ""))
        elif k == "p":
            out.append(t_inline(b[1]) + "\n")
        elif k == "ul":
            env = "enumerate" if b[2] else "itemize"
            out.append(f"\\begin{{{env}}}\\setlength{{\\itemsep}}{{1pt}}\n" + "\n".join("\\item " + t_inline(i) for i in b[1]) + f"\n\\end{{{env}}}")
        elif k == "table":
            s = b[1]
            (tex_dir / "tables" / f"{s['label']}.tex").write_text(tex_table(s), encoding="ascii")
            out.append(f"\\input{{tables/{s['label']}.tex}}")
        elif k == "fig":
            s = b[1]
            out.append(f"\\begin{{figure}}[htbp]\n\\centering\n\\includegraphics[width={s['width']:.2f}\\linewidth]{{figures/{fig_name(s['path'])}}}\n"
                       f"\\caption{{{t_inline(s['caption'])}}}\\label{{fig:{s['label']}}}\n\\end{{figure}}")
    out.append("\\end{document}\n")
    return "\n".join(out)


def check_tex(tex_dir: Path) -> list[str]:
    """Structural LaTeX check without a TeX engine: ASCII only, balanced braces and environments, every \\input / \\includegraphics
    file exists, every \\ref has a \\label."""
    errs = []
    texts = {p.name: p.read_text(encoding="utf-8") for p in [tex_dir / "main.tex", *sorted((tex_dir / "tables").glob("*.tex"))]}
    allt = "\n".join(texts.values())
    for name, t in texts.items():
        if any(ord(c) > 127 for c in t):
            errs.append(f"{name}: non-ASCII character")
        s = re.sub(r"\\[{}$&#_%]", "", t)
        depth = 0
        for c in s:
            depth += (c == "{") - (c == "}")
            if depth < 0:
                errs.append(f"{name}: unbalanced closing brace"); break
        if depth > 0:
            errs.append(f"{name}: {depth} unclosed brace(s)")
        if len(re.findall(r"(?<!\\)\$", s)) % 2:
            errs.append(f"{name}: odd number of $")
        stack = []
        for m in re.finditer(r"\\(begin|end)\{(\w+\*?)\}", t):
            if m.group(1) == "begin":
                stack.append(m.group(2))
            elif not stack or stack.pop() != m.group(2):
                errs.append(f"{name}: mismatched \\end{{{m.group(2)}}}"); break
        if stack:
            errs.append(f"{name}: unclosed environments {stack}")
    for f in re.findall(r"\\input\{([^}]+)\}", allt) + re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", allt):
        if not (tex_dir / f).exists():
            errs.append(f"missing file {f}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", allt))
    errs += [f"undefined reference {r}" for r in set(re.findall(r"\\ref\{([^}]+)\}", allt)) - labels]
    return errs


# ============================================================================================ content
def build(cfg: dict, doc: Doc, figdir: Path) -> dict:
    t2 = rd("stage5/table2.csv"); t3 = rd("stage7/table3_referral.csv"); t10 = rd("stage10/e1_ensemble_baseline.csv")
    c6 = rd("stage6/contrasts.csv"); c6a = rd("stage6/a1/contrasts.csv")
    far6 = rd("stage6/far_table.csv"); far6a = rd("stage6/a1/far_table.csv")
    tt = rd("stage7/threshold_transfer.csv")
    G = {g: rd(f"stage8/tables/ablation_{g}.csv") for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")}
    EX = rd("stage8/tables/exploratory_targeted.csv")
    ftau = rd("stage4/frozen_tau.csv"); ao = rd("stage11/accepted_only.csv")
    tab1 = rd("stage1/table1.csv"); fs = rd("stage3/family_summary.csv"); ss = rd("stage3/shortcut_states.csv")
    cm = rd("stage4/cue_mass.csv"); cc = rd("stage11/cam_cases.csv"); il = rd("stage5/incremental_logit.csv")
    inv = rd("stage5/inversion_diagnosis.json"); tiers = rd("stage7/tiers.csv"); sens5 = rd("stage5/sensitivity.csv")

    def t2r(H, q, m): return one(t2, heldout=H, quantity=q, metric=m)
    def t3r(H, q, m, b): return one(t3, heldout=H, quantity=q, metric=m, budget=float(b))
    def g8(H, q, b, m="dresid_sf", g="G2"):
        d = G[g]; return one(d[(d.kind == "contrast") & (d.level_col.isna())], heldout=H, quantity=q, metric=m, budget=float(b))
    def ex(H, q, b, m="dresid_sf"): return one(EX, heldout=H, quantity=q, metric=m, budget=float(b))
    def c6r(tab, H, a, b, kind="standard"):
        return one(tab[(tab["set"] == "primary") & (tab["size"] == 100) & (tab["mask"] == "silent_pooled")], H=H, kind=kind, a=a, b=b)
    def rc6(r): return ci(r["diff"], r["ci_low"], r["ci_high"], 3, 1, True)
    def e10(H, q, m, b=None, kind="level"):
        d = t10[t10.kind == kind]
        return one(d, heldout=H, quantity=q, metric=m) if b is None else one(d, heldout=H, quantity=q, metric=m, budget=float(b))
    def e20(H, q): return rc(one(t10, kind="contrast", heldout=H, quantity=q, metric="dresid_sf", budget=0.2), 2, 100)
    def pt(r): return rc(r, 2, 100)
    MARK = {"reduces": "↓", "improves": "better", "increases": "↑", "worse": "worse", "no practical difference": "=", "inconclusive": "?"}

    nfar_pre = int(far6[far6["size"] == 100].flag_outside_3_7pct.sum()); nfar_a1 = int(far6a[far6a["size"] == 100].flag_outside_3_7pct.sum())
    ncell = int((far6["size"] == 100).sum())
    n_tt = int(tt[tt.metric == "realised_workload"].workload_flag.astype(str).eq("True").sum()); N_tt = int((tt.metric == "realised_workload").sum())
    n_degen = int(ftau.degenerate.notna().sum())
    must(f"{nfar_pre}/{ncell} cells outside 3-7% (after A1: {nfar_a1}/{ncell})"); must(f"{n_tt}/{N_tt} cells off by")
    must(f"for {n_degen} of {len(ftau)} classifiers")

    # ---- frozen plans: commit that ADDED each plan file (read from git, never typed)
    plans = [("docs/analysis_plan_frozen.md", "sample-level detection (Phase 5)"), ("docs/batch_plan_frozen.md", "batch monitoring (Phase 6)"),
             ("docs/batch_plan_amendment_1.md", "amendment A1 (Phase 6)"), ("docs/referral_plan_frozen.md", "referral policy (Phase 7)"),
             ("docs/ablation_plan_frozen.md", "ablations (Phase 8)"), ("docs/extension_plan_frozen.md", "extensions E1/E2 (Phase 10, post-hoc)")]
    PL = {}
    for f, what in plans:
        h, date = git("log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", f).splitlines()[-1].split()
        PL[f] = (h, date, what)
    hP5, hP6, hA1, hP7, hP8, hP10 = (PL[f][0] for f, _ in plans)
    for h in (hP5, hP6, hA1, hP7, hP8, hP10):
        must(h, "plan commit")

    # ---- design constants (configs/full.yaml)
    seeds = len(cfg["train"]["seeds"]); B5 = cfg["stage5"]["bootstrap_B"]; budgets = cfg["stage7"]["budgets"]
    bud = " / ".join(f"{int(100 * b)}" for b in budgets) + "%"
    win = cfg["stage6"]["size"]; refn = cfg["stage4"]["ref_n"]; gate = cfg["stage8"]["gate"]
    npool = {H: int(one(t2, heldout=H, quantity="M0", metric="AUROC").n_per_seed) for H in "ABCX"}
    prev = {H: one(t2, heldout=H, quantity="M0", metric="AUROC").sf_rate for H in "ABCX"}
    pm_tr = one(tab1, dataset="pathmnist", split="train"); cam_tr = one(tab1, dataset="camelyon17", split="train")
    must(f"PathMNIST train has {int(pm_tr.n):,} images; Camelyon train {int(cam_tr.n):,} patches from only {int(cam_tr.n_groups)} patients")

    # ================================================================================ title + executive summary
    doc.raw("title")
    doc.raw("exec_begin")
    doc.h(1, "Executive summary")
    doc.p("**Question.** Do *representation drift* (Mahalanobis distance of an image's features to the training data) and *Grad-CAM explanation drift* "
          "detect **silent failures** - wrong predictions made with calibrated confidence ≥ 0.90 - beyond the model's own calibrated uncertainty, on a kind "
          "of distribution shift the detector was not tuned on? And do they lower the residual silent-failure rate when a fixed share of cases can be sent "
          f"to a human? Setting: ResNet-18 ({seeds} seeds) on PathMNIST and Camelyon17; shift families A (11 image corruptions × 5 severities), B (two unseen "
          "hospitals) and C (a learned colour-cue shortcut, removed or reversed); nested detectors M0 (uncertainty) < M1 (+ Mahalanobis) < M2 (+ CAM drift); "
          f"every analysis pre-registered; paired two-level cluster bootstrap ({B5:,} resamples), 95% CIs in brackets.")
    doc.p("**Three main findings.**")
    doc.ul([
        f"**Representation drift helps for natural shifts, hurts on a shortcut, and its gain is mostly absorbed by an ensemble.** AUROC gain M1 - M0 = "
        f"{rc(t2r('A','M1-M0','dAUROC'))} (A), {rc(t2r('B','M1-M0','dAUROC'))} (B), {rc(t2r('C','M1-M0','dAUROC'))} (C). Against a 4-seed deep ensemble "
        f"(post-hoc extension) M1 leaves more silent failures at a 20% budget on B ({e20('B','M1-E_fuse')} pt) and C ({e20('C','M1-E_fuse')} pt) and is inconclusive on A "
        f"({e20('A','M1-E_fuse')} pt).",
        f"**Explanation drift adds nothing as a pre-registered (monotone) alarm, but helps with a free sign.** M2 - M1 dAUROC = {rc(t2r('A','M2-M1','dAUROC'),4)} (A), "
        f"{rc(t2r('B','M2-M1','dAUROC'),4)} (B), {rc(t2r('C','M2-M1','dAUROC'),4)} (C). Allowing a negative CAM weight (pre-registered ablation P2) lowers the residual "
        f"silent-failure rate at a 10% budget on A ({pt(g8('A','sf_M2-sf_M1',0.1))} pt) and C ({pt(g8('C','sf_M2-sf_M1',0.1))} pt): confident errors look *typical*.",
        f"**Uncertainty-first referral cannot reach silent failures at small budgets.** At a 5% review budget it leaves *more* silent failures than random review "
        f"(M0 - random: A {pt(t3r('A','M0-random','dresid_sf',0.05))} pt, B {pt(t3r('B','M0-random','dresid_sf',0.05))} pt); adding M1 helps only at 20% "
        f"(A {pt(t3r('A','M1-M0','dresid_sf',0.2))} pt, B {pt(t3r('B','M1-M0','dresid_sf',0.2))} pt) and hurts on C ({pt(t3r('C','M1-M0','dresid_sf',0.2))} pt).",
    ])
    shown = pd.DataFrame([
        ("M1 beats single-model uncertainty on corruptions (A) and new hospitals (B) (pre-registered)", "M1 beats a deep-ensemble uncertainty baseline"),
        ("Monotone CAM drift adds nothing; CIs exclude all but very small gains (pre-registered)", "Any detector that works on the strong shortcut (C)"),
        ("Sign-free CAM drift lowers residual silent failures on A and C (pre-registered ablation)", "The mechanism behind the negative CAM weight (hypothesis only)"),
        (f"Uncertainty-first referral is worse than random at a 5% budget", "That any detector helps at a 5% budget"),
        (f"Thresholds do not transfer: batch FAR check failed ({nfar_pre}/{ncell}), referral workload missed ({n_tt}/{N_tt})", "Generalisation to other architectures or attribution methods"),
    ], columns=["shown (95% CI excludes 0)", "not shown"])
    doc.table(shown, "What the study shows and does not show.", "shown", align="YY", size="small")
    doc.p("**Main limitation.** One architecture family (ResNet-18) and one explanation method (Grad-CAM) on two pathology datasets, with few Camelyon patients; "
          "the representation gain is largely what any better uncertainty estimate (a deep ensemble) provides, so the practical case for drift monitors rests on "
          "the sign-free explanation result and on budgets of 20% or more.")
    doc.raw("exec_end")

    # ================================================================================ 1 introduction
    doc.h(1, "Introduction", "intro")
    doc.p("Image classifiers in pathology are typically trained on a few centres and deployed on images that differ in staining, scanner, artefacts or "
          "spurious cues. Most errors under such shift come with low confidence and can be routed to a pathologist. The dangerous errors are **silent**: "
          "the model is wrong *and* confident after calibration. Calibrated uncertainty cannot flag them by construction, so other signals are needed. Two "
          "candidates are popular in the monitoring literature: *representation drift* (is the image's feature vector far from the training data?) and "
          "*explanation drift* (does the model attend to an unusual image region?). This study tests both in a pre-registered design in which every "
          "detector is tuned on two kinds of shift and evaluated on a third, so that all results are out-of-family.")
    doc.p("**Research questions.**")
    doc.ul([
        "**RQ1.** Does representation drift (Mahalanobis distance, detector M1) detect silent failures beyond calibrated uncertainty (M0), out-of-family?",
        "**RQ2.** Does Grad-CAM explanation drift (M2) add beyond representation drift?",
        "**RQ3.** Do batch (window) monitors detect silently failed windows beyond mean uncertainty?",
        "**RQ4.** Do the detectors lower the residual silent-failure rate among automatically accepted cases under a fixed human-review budget?",
    ])
    doc.p(f"Methods are in {{sec:methods}}, results in {{sec:results}}, the answers with a strength-of-evidence rating in {{sec:answers}}, limitations in "
          f"{{sec:limits}} and reproducibility in {{sec:repro}}. Every number in this report is generated by `scripts/build_pdf_report.py` from the saved result "
          "tables of the repository and checked against the executed final notebook `silent_failure_report.ipynb`.")

    # ================================================================================ 2 methods
    doc.h(1, "Methods", "methods")
    doc.h(2, "Data and shift families", "data")
    cam = tab1[tab1.dataset == "camelyon17"].set_index("split"); pmn = tab1[tab1.dataset == "pathmnist"].set_index("split")
    doc.p(f"**PathMNIST** (MedMNIST v2, 128 × 128 px, 9 colorectal tissue classes): the official train + validation pool is split by near-duplicate "
          f"clusters into train ({int(pmn.loc['train','n']):,}), val_es (early stopping, {int(pmn.loc['val_es','n']):,}), val_cal (calibration and references, "
          f"{int(pmn.loc['val_cal','n']):,}) and id_test ({int(pmn.loc['id_test','n']):,}); the official test set (CRC-VAL-HE-7K, another centre) is a supplementary family X. "
          f"**Camelyon17-WILDS v1.0** (96 × 96 px, tumour vs normal): patient-disjoint source splits from hospitals 0, 3, 4 (train {int(cam.loc['train','n']):,} patches from "
          f"{int(cam.loc['train','n_groups'])} patients); hospitals 1 and 2 ({int(cam.loc['ood_c1','n']):,} and {int(cam.loc['ood_c2','n']):,} patches) are held out. "
          f"Shift families (pool rows per seed): **A** = MedMNIST-C v0.1.0, 11 corruptions × severities 1-5 applied at 224 px and resized, plus clean images, on one fixed "
          f"3,000-image id_test sample ({npool['A']:,} rows); **B** = Camelyon id_test + the two held-out hospitals ({npool['B']:,}); **C** = a synthetic shortcut: a class-coded "
          f"colour cue in the image corner present on 100% of training images (v2, primary; v1 with 95% is a supplement), at test aligned, removed or reversed ({npool['C']:,}); "
          f"**X** = PathMNIST id_test + CRC-VAL-HE-7K ({npool['X']:,}). Silent-failure prevalence (sf_090) in the pools: A {prev['A']:.1%}, B {prev['B']:.1%}, C {prev['C']:.1%}, X {prev['X']:.1%}.")
    doc.h(2, "Classifiers and calibration", "models")
    doc.p(f"ResNet-18 initialised from ImageNet, SGD (lr {cfg['train']['optimizer']['lr']}, cosine schedule), at most {cfg['train']['epochs']} epochs with early stopping on "
          f"val_es loss (patience {cfg['train']['early_stopping']['patience']}), augmentation by flips and 90° rotations only (no colour or blur augmentation, which would "
          f"pre-train robustness to family A). {seeds} seeds for each of four model kinds (PathMNIST, Camelyon, shortcut v1, shortcut v2) = {4 * seeds} frozen classifiers. "
          "Temperature scaling is fitted on val_cal. Calibrated confidence is the maximum softmax probability of z/T.")
    doc.h(2, "Silent-failure labels", "labels")
    doc.p(f"Primary label **sf_090**: prediction wrong AND calibrated confidence ≥ 0.90. Sensitivity labels: sf_095 (≥ 0.95) and sf_099 (99%-precision rule). "
          f"The originally planned rule sf_tau (smallest threshold with ≥ 95% precision on val_cal) is **degenerate**: for {n_degen} of {len(ftau)} classifiers "
          "the val_cal accuracy already exceeds 95%, so tau falls to the lowest observed confidence and sf_tau labels nearly every error. The fixed 0.90 threshold was "
          f"therefore made primary before any detection result was computed (plan {hP5}); sf_tau remains a sensitivity analysis.")
    doc.h(2, "Signals, fusion and held-out-family rotation", "signals")
    doc.p("Per-image signals (higher = more suspicious): **M0** uncertainty, log(1 - confidence) and log entropy; **M1** adds the Mahalanobis distance of the "
          f"penultimate features to the predicted class (class means and one shared Ledoit-Wolf covariance fitted on {refn['pathmnist']:,} PathMNIST / {refn['camelyon17']:,} "
          "Camelyon *training* images only, frozen with checksums before any shifted file was read); **M2** adds layer4 Grad-CAM drift, the Jensen-Shannon divergence "
          "between the image's normalised CAM and the predicted class's mean training CAM, divided by its typical variability under benign transforms. kNN distance, "
          "layer3/layer2 CAMs, raw JS and a PCA-embedding CAM distance are sensitivity variants. Each signal is standardised to its percentile among val_cal values "
          "(no shifted data). **Fusion** = logistic regression with *non-negative* weights (the pre-registered monotone alarm); a *sign-free* variant was a "
          "pre-registered Phase 8 ablation. **Held-out-family rotation:** weights are fitted per seed on two of A/B/C and evaluated on the third.")
    doc.h(2, "Endpoints and inference", "inference")
    doc.p(f"Sample level: AUROC, AUPRC and AURC (area under the risk-coverage curve, lower is better). Referral: the top B% of the held-out pool by score is "
          f"referred (exact budget B = {bud}, fractional ties), and the endpoint is the residual silent-failure rate among accepted cases, with random referral "
          f"and an oracle as references; verdict rule: CI excludes 0 and |difference| ≥ {100 * cfg['stage7']['min_effect']:.1f} pt. Batch level: windows of {win} images "
          f"with a shifted fraction of 0-100%, alarm at the 95th percentile of {cfg['stage6']['n_valcal_windows']:,} source windows (nominal 5% false-alarm rate, FAR), "
          f"actual FAR measured on {cfg['stage6']['n_far_windows']:,} held-out source windows. **Inference:** paired two-level cluster bootstrap with {B5:,} resamples "
          "(seeds, then near-duplicate clusters for PathMNIST or patients for Camelyon; the same draw for all methods), 95% percentile CIs, leave-one-seed-out "
          "jackknife; declared primaries unadjusted, Benjamini-Hochberg FDR inside each secondary family.")
    doc.h(2, "Pre-registration and amendment", "prereg")
    plan_tab = pd.DataFrame([(f, w, h, d) for f, (h, d, w) in PL.items()], columns=["frozen document", "analysis", "commit", "date"])
    doc.table(plan_tab, "Frozen analysis plans: each was committed before any of its numbers were computed (commit that added the file, read from git).",
              "plans", align="lYll", size="small")
    doc.p(f"**Amendment A1** ({hA1}): the pre-registered batch false-alarm check failed (thresholds set on val_cal gave actual FARs outside 3-7% on id_test windows in "
          f"{nfar_pre} of {ncell} cells). A1, committed before computing it, draws thresholds and evaluation windows from two disjoint halves of the same source pool; "
          "pre-registered and A1 results are both reported. Analyses added after the pre-registered phases (ensemble baseline, accepted-rows-only sensitivity, "
          "evidence map, excluded-benefit bounds, Grad-CAM case study, confidence-gated ranking) are labelled *post-hoc*.")

    # ================================================================================ 3 results
    doc.h(1, "Results", "results")
    doc.h(2, "Classifier performance and behaviour under shift", "behaviour")
    mps = pd.concat([rd("stage2/metrics_per_seed_pathmnist.csv"), rd("stage2/metrics_per_seed_camelyon17.csv")], ignore_index=True)
    for c in ("acc", "ece"):
        mps[c] = mps[c].round(4)
    summ = mps[mps.variant == "T_standard"].groupby(["dataset", "split"]).agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
                                                                               ece_mean=("ece", "mean"), ece_sd=("ece", "std")).round(4).reset_index()
    pmid = one(summ, dataset="pathmnist", split="id_test"); cmid = one(summ, dataset="camelyon17", split="id_test"); c2 = one(summ, dataset="camelyon17", split="ood_c2")
    a5 = one(fs, family="A", severity=5); r2 = one(ss, model_kind="shortcutv2", condition="reversed")
    pm_txt = must(f"id_test accuracy {pmid.acc_mean:.4f} +- {pmid.acc_sd:.4f}, ECE {pmid.ece_mean:.4f}").replace("+-", "±")
    cm_txt = must(f"id_test {cmid.acc_mean:.3f} +- {cmid.acc_sd:.3f}").replace("+-", "±")
    doc.p(f"PathMNIST is almost solved in distribution ({pm_txt}); Camelyon is harder "
          f"({cm_txt}) and degrades on hospital 2 ({must(f'accuracy {c2.acc_mean:.3f}, ECE {c2.ece_mean:.3f}')}); "
          "see {tab:t1} and the reliability diagrams in {fig:relp} and {fig:relc}. Under shift, confidence falls much more slowly than accuracy, so silent failures grow with severity "
          f"({must(f'severity 5: accuracy {a5.acc:.3f}, confidence {a5.conf:.3f}, SF rate {a5.sf_rate:.3f}')}; {{fig:behA}}); under a reversed v2 cue the model is almost "
          f"always wrong yet confident ({must(f'accuracy {r2.acc:.3f}, SF rate {r2.sf_rate:.3f}')}).")
    t1 = summ.assign(**{"accuracy (mean ± SD)": [f"{a:.4f} ± {s:.4f}" for a, s in zip(summ.acc_mean, summ.acc_sd)],
                        "ECE (mean ± SD)": [f"{a:.4f} ± {s:.4f}" for a, s in zip(summ.ece_mean, summ.ece_sd)]})[["dataset", "split", "accuracy (mean ± SD)", "ECE (mean ± SD)"]]
    doc.table(t1, f"Classifier accuracy and expected calibration error after temperature scaling, mean ± SD over {seeds} seeds (outputs/stage2/metrics_per_seed_*.csv).",
              "t1", align="llYY", size="small")
    doc.fig(OUT / "stage3/figures/behaviour_family_A.png", "Family A (MedMNIST-C), all corruptions pooled: accuracy, mean calibrated confidence, ECE and silent-failure "
            "rate by severity (0 = clean), mean over seeds with seed SD (outputs/stage3/figures). Confidence stays high while accuracy drops.", "behA", 1.0)

    doc.h(2, "Sample-level detection (pre-registered primary)", "detection")
    rows = []
    for H in "ABCX":
        for m in ("AUROC", "AUPRC", "AURC"):
            r = {"family": H, "metric": m}
            for q in ("M0", "M1", "M2"):
                d = one(t2, heldout=H, quantity=q, metric=m); r[q] = ci(d.estimate, d.ci_low, d.ci_high, 3, 1, False)
            for q, lab in (("M1-M0", "M1 - M0"), ("M2-M1", "M2 - M1")):
                d = one(t2, heldout=H, quantity=q, metric="d" + m); r[lab] = ci(d.estimate, d.ci_low, d.ci_high, 4, 1, True)
            rows.append(r)
    m21B = must(f"{t2r('B', 'M2-M1', 'dAUROC').estimate:+.4f}")
    cam_med = must("median " + ", ".join(f"{il[il.heldout == H].coef_cam.median():.2f} {H}" for H in "ABC"))
    doc.table(pd.DataFrame(rows), f"Sample-level detection of silent failures (sf_090) on the held-out family, mean over {seeds} seeds [95% paired CI]. "
              "AURC: lower is better. X is a supplement (weights fitted on A + B + C).", "t2", align="llYYYYY", size="footnotesize")
    doc.p(f"Adding representation drift improves detection on corruptions and new hospitals and lowers it on the shortcut (M1 - M0 dAUROC "
          f"A {rc(t2r('A','M1-M0','dAUROC'))}, B {rc(t2r('B','M1-M0','dAUROC'))}, C {rc(t2r('C','M1-M0','dAUROC'))}; {{tab:t2}}, {{fig:f4}}). Adding CAM drift "
          f"changes nothing: the non-negative fusion assigns the CAM signal a zero weight whenever A or C is held out, so M2 - M1 is {t2r('A','M2-M1','dAUROC').estimate:+.1e} (A), "
          f"{m21B} (B) and {t2r('C','M2-M1','dAUROC').estimate:+.1e} (C) - formal 'worse' verdicts with CIs of width ~1e-6 are no practical effect. Given M0 and "
          f"Mahalanobis, the unconstrained CAM coefficient is negative in every held-out family ({cam_med}): "
          "confident errors have *typical-looking* explanations, which a monotone alarm cannot use.")
    doc.fig(OUT / "stage5/figures/fig4_increments.png", "Incremental value of representation drift (M1 - M0) and explanation drift (M2 - M1) on the held-out family: "
            "dAUROC, dAUPRC, dAURC with 95% paired CIs (pre-registered primary; outputs/stage5/figures; panel titles use the notebook's figure numbers).", "f4", 1.0)
    f3 = rd("stage5/fig3_data.csv")
    doc.p(f"Single signals ({{fig:f3}}) show the same pattern: on A, Mahalanobis alone reaches AUROC {rc(one(f3, family='A', signal='r_maha_pred', metric='AUROC'), signed=False)} "
          f"vs {rc(one(f3, family='A', signal='u_log1msp', metric='AUROC'), signed=False)} for log(1 - confidence), while CAM signals stay near 0.5-0.6. On C the "
          f"pooled AUROCs hide an inversion: within the reversed cue state every signal is far below 0.5 (log(1 - confidence) {must(format(inv['auroc_sf090']['reversed']['u_log1msp'], '.3f'))}, "
          f"Mahalanobis {must(format(inv['auroc_sf090']['reversed']['r_maha_pred'], '.3f'))}): the few correct predictions are the atypical cases, the confident errors look normal.")
    doc.fig(OUT / "stage5/figures/fig3_single_signals.png", "Single-signal detection of silent failures per family: AUROC (left) and AUPRC (right), descriptive "
            "(outputs/stage5/figures; panel titles use the notebook's figure numbers).", "f3", 0.9)

    doc.h(2, "Population sensitivity: accepted rows only (post-hoc)", "accepted")
    g2l = G["G2"]; g2l = g2l[g2l.level_col.isna()]
    def lvl_all(H, q, m):
        return one(t2, heldout=H, quantity=q, metric=m.upper()) if q in ("M0", "M1", "M2") else one(g2l[g2l.kind == "level"], heldout=H, quantity=q, metric=m)
    def con_all(H, q, m):
        return one(t2, heldout=H, quantity=q, metric="d" + m.upper()) if q in ("M1-M0", "M2-M1") else one(g2l[g2l.kind == "contrast"], heldout=H, quantity=q, metric="d" + m)
    def lvl_acc(H, q, m): return one(ao[ao.kind == "level"], heldout=H, quantity=q, metric=m)
    def con_acc(H, q, m): return one(ao[ao.kind == "contrast"], heldout=H, quantity=q, metric="d" + m)
    poprows = {m: [] for m in ("auroc", "auprc")}
    for H in "ABC":
        for m in ("auroc", "auprc"):
            for pop, lv_, cn_ in (("all rows (primary)", lvl_all, con_all), ("accepted only", lvl_acc, con_acc)):
                r = {"family": H, "population": pop, "SF prev.": f"{float(lv_(H, 'M0', m)['sf_rate' if pop.startswith('all') else 'prevalence']):.4f}"}
                for q in ("M0", "M1"):
                    d = lv_(H, q, m); r[q] = ci(d.estimate, d.ci_low, d.ci_high, 3, 1, False)
                for q, lab in (("M1-M0", "M1 - M0"), ("M2-M1", "M2 - M1"), ("sf_M2-sf_M1", "sign-free M2 - M1")):
                    d = cn_(H, q, m); r[lab] = ci(d.estimate, d.ci_low, d.ci_high, 3, 1, True)
                poprows[m].append(r)
    sh = ao[["heldout", "accepted_share"]].drop_duplicates().set_index("heldout").accepted_share
    acc_sh = must(f"accepted rows are {100 * sh['A']:.0f}% (A), {100 * sh['B']:.0f}% (B) and {100 * sh['C']:.0f}% (C)")
    doc.p(f"The primary population is all deployed rows. As a post-hoc sensitivity analysis the same detectors and weights were evaluated on accepted rows only "
          f"(confidence ≥ {cfg['stage11']['accepted_only']['accept_conf']:.2f}; {acc_sh} of the pools), "
          "where silent failures are the only errors ({tab:acc}). The direction of M1 - M0 is unchanged (A and B positive, C negative), but the sign-free CAM benefit "
          f"is not uniform across populations: on accepted rows its dAUROC is {rc(con_acc('A','sf_M2-sf_M1','auroc'))} on A and {rc(con_acc('C','sf_M2-sf_M1','auroc'))} on C.")
    doc.table(pd.DataFrame(poprows["auroc"]), "AUROC of silent-failure detection on all rows (primary) vs accepted rows only (post-hoc sensitivity), held-out family, "
              "[95% CI] (outputs/stage11/accepted_only.csv; AUPRC in {tab:accprc}).", "acc", align="llrYYYYY", size="footnotesize")

    doc.h(2, "Batch-level drift detection and the failed false-alarm check", "batch")
    cmp = []
    for H, k in (("A", "standard"), ("B", "standard"), ("C", "shortcutv2"), ("X", "standard")):
        for a_, b_ in (("comb", "u_mean"), ("r_mmd", "u_mean"), ("e_cam_l4", "r_mmd")):
            r0, r1 = c6r(c6, H, a_, b_, k), c6r(c6a, H, a_, b_, k)
            cmp.append({"family": H, "contrast": f"{a_} - {b_}", "pre-registered thresholds (unequal FAR)": rc6(r0) + f" {r0.verdict}",
                        "A1, FAR-matched": rc6(r1) + f" {r1.verdict}", "silent windows": int(r1.n_windows)})
    doc.p(f"The pre-registered check of the batch monitors **failed**: with thresholds from val_cal windows, {nfar_pre} of {ncell} (statistic × family) cells had an "
          f"actual FAR on id_test windows outside 3-7% (e.g. mean uncertainty on A: {must(format(one(far6, H='A', kind='standard', size=100, stat='u_mean').far, '.3f'))}). "
          "Fresh val_cal windows alarm at the nominal rate, so the pipeline is unbiased, but val_cal and id_test are not exchangeable at window level: a small row-level "
          f"difference becomes large for a mean over {win} images. Under amendment A1 {nfar_a1} of {ncell} cells remain outside the tolerance; statistics are compared only at "
          f"matched actual FAR. Then the combined monitor never beats mean uncertainty on silently failed windows (A {rc6(c6r(c6a,'A','comb','u_mean'))}), MMD beats mean "
          f"uncertainty only on the new hospitals (B {rc6(c6r(c6a,'B','r_mmd','u_mean'))}), and window CAM drift beats MMD only on the shortcut, where MMD is blind "
          f"(C {rc6(c6r(c6a,'C','e_cam_l4','r_mmd','shortcutv2'))}; {{tab:batch}}, {{fig:f5}}).")
    doc.table(pd.DataFrame(cmp), f"Declared primary batch contrasts: difference in power on silently failed windows (window size {win}) [95% CI] and verdict, "
              "pre-registered thresholds vs FAR-matched amendment A1 (outputs/stage6/contrasts.csv, outputs/stage6/a1/contrasts.csv). Only the A1 column is a valid comparison.",
              "batch", align="llYYr", size="footnotesize")
    doc.fig(OUT / "stage6/a1/figures_fig5_power.png", "Batch monitoring under amendment A1: alarm rate vs shifted fraction of the window for all shifted windows (top) "
            "and silently failed windows (bottom), one representative cell per family (A severity 3, B hospital 2, C reversed v2, X) (outputs/stage6/a1).", "f5", 1.0)

    doc.h(2, "Referral at fixed review budgets (pre-registered primary)", "referral")
    rows = []
    for H in "ABCX":
        for b in budgets:
            sub = t3[(t3.heldout == H) & np.isclose(t3.budget, b) & (t3.metric == "resid_sf")]
            r = {"family": H, "budget": f"{int(100 * b)}%", "prev. (%)": f"{100 * sub.prevalence.iloc[0]:.2f}"}
            for q in ("M0", "M1", "M2", "oracle"):
                d = sub[sub.quantity == q].iloc[0]; r[q + " (%)"] = ci(d.estimate, d.ci_low, d.ci_high, 2, 100, False)
            for q in ("M1-M0", "M2-M1"):
                d = one(t3, heldout=H, quantity=q, metric="dresid_sf", budget=float(b))
                r[q.replace("-", " - ") + " (pt)"] = ci(d.estimate, d.ci_low, d.ci_high, 2, 100, True) + f" {MARK.get(d.verdict, '')}"
            rows.append(r)
    capt = must(f"M0 captures {t3r('A', 'M0', 'capture', 0.05).estimate:.1%} of silent failures at 5% on A (oracle {t3r('A', 'oracle', 'capture', 0.05).estimate:.1%})")
    rerr = must(f"{t3r('A', 'M0', 'resid_err', 0.2).estimate:.1%} vs {t3r('A', 'random', 'resid_err', 0.2).estimate:.1%} random at 20%")
    doc.p("Random referral leaves the prevalence as residual rate (column prev.); the oracle refers silent failures first. Ranking by uncertainty sends the *unsure* "
          "cases to review first, but silent failures are by definition *sure*: at small budgets the review capacity is spent on visible errors "
          f"({{fig:f6}}, {{tab:t3}}). At 5%, M0 leaves more silent failures than random (A {pt(t3r('A','M0-random','dresid_sf',0.05))} pt). "
          f"{capt} while it does lower the residual *error* rate ({rerr}). "
          f"M1 reduces the residual silent-failure rate only at 20% (A {pt(t3r('A','M1-M0','dresid_sf',0.2))} pt, B {pt(t3r('B','M1-M0','dresid_sf',0.2))} pt) and raises "
          f"it on C ({pt(t3r('C','M1-M0','dresid_sf',0.2))} pt); M2 - M1 is zero on A and C.")
    doc.table(pd.DataFrame(rows), "Residual silent-failure rate among accepted cases at exact review budgets, % [95% CI], and paired contrasts in percentage points "
              "(↓ reduces, ↑ increases, = no practical difference, ? inconclusive) (outputs/stage7/table3_referral.csv).", "t3", align="lllYYYYYY", size="scriptsize")
    doc.fig(OUT / "stage7/figures/fig6_residual_vs_budget.png", "Residual silent-failure rate among accepted cases vs review budget, with random and oracle references "
            "(pre-registered primary; outputs/stage7/figures).", "f6", 1.0)
    ttw = tt[tt.metric == "realised_workload"].copy()
    tw = ttw.assign(v=[ci(e, l, h, 1, 100, False, reg=False) + (" *" if str(f) == "True" else "") for e, l, h, f in zip(ttw.estimate, ttw.ci_low, ttw.ci_high, ttw.workload_flag)],
                    B=(100 * ttw.budget).round(0).astype(int).astype(str) + "%")
    tw = tw.pivot_table(index=["heldout", "quantity"], columns="B", values="v", aggfunc="first").reset_index()
    tw = tw[["heldout", "quantity"] + sorted(tw.columns[2:], key=lambda c: int(c.rstrip("%")))]
    tw.columns = ["family", "method"] + [f"nominal {c}" for c in tw.columns[2:]]
    doc.p(f"**Threshold transfer (secondary).** The Blueprint-literal protocol tunes score thresholds on the two tuning families and transfers them. "
          f"{n_tt} of {N_tt} cells miss the intended workload by more than 1 point ({{tab:tt}}): score scales do not transfer between families - the same lesson as the "
          "batch FAR failure - which is why the primary protocol ranks within the deployment pool.")
    doc.table(tw, "Threshold transfer: realised review workload on the held-out family, % [95% CI]; * = more than 1 point from the nominal budget "
              "(outputs/stage7/threshold_transfer.csv).", "tt", align="llYYY", size="small")

    doc.h(2, "Explanation drift: monotone vs sign-free fusion (Phase 8)", "signfree")
    G2c = ab_compact(G["G2"], ["sf_M0-M0", "sf_M1-M1", "sf_M2-M2", "sf_M1-sf_M0", "sf_M2-sf_M1"], budgets=(0.1, 0.2), MARK=MARK)
    doc.p(f"Phase 8 declared three primary contrasts on the residual silent-failure rate: P1 sign-free M1 - sign-free M0, P2 sign-free M2 - sign-free M1, P3 sign-free M1 - "
          f"non-negative M1. P3 is 0.00 pt everywhere (A 20% {pt(g8('A','sf_M1-M1',0.2))}): uncertainty and Mahalanobis weights are already non-negative, so P1 equals "
          f"the referral result. **P2 reduces residual silent failures** on A ({pt(g8('A','sf_M2-sf_M1',0.05))} / {pt(g8('A','sf_M2-sf_M1',0.1))} / {pt(g8('A','sf_M2-sf_M1',0.2))} pt "
          f"at 5/10/20%) and C ({pt(g8('C','sf_M2-sf_M1',0.05))} / {pt(g8('C','sf_M2-sf_M1',0.1))} / {pt(g8('C','sf_M2-sf_M1',0.2))} pt), not on B "
          f"({pt(g8('B','sf_M2-sf_M1',0.2))} pt at 20%), with a negative fitted CAM weight ({{tab:g2}}, {{fig:f8a}}). A non-negative detector is adequate for uncertainty "
          "and representation drift but not for explanation drift. Under sign-free fusion layer3 and layer2 CAMs also help on C, so the sign matters more than the layer "
          "(other ablations in the appendix).")
    doc.table(G2c, "Fusion-constraint ablation G2 (contains the pre-registered primaries P1-P3): difference in residual silent-failure rate, pt [95% CI], at 10% and 20% "
              "budgets (outputs/stage8/tables/ablation_G2.csv).", "g2", align="l" + "Y" * (G2c.shape[1] - 1), size="footnotesize")
    doc.fig(OUT / "stage8/figures/fig8a_incremental_deltas.png", "Incremental effect of each ablation (variant - reference) on the held-out family: residual "
            "silent-failure rate at a 20% budget (top, pt) and AUROC (bottom), 95% paired CIs (outputs/stage8/figures).", "f8a", 1.0)
    doc.p(f"*Exploratory, post-hoc:* ranking the confident cases (confidence ≥ {gate:.2f}) first, by a refitted score, cuts residual silent failures strongly on A and B "
          f"(gated M1 - M1 at 20%: A {pt(ex('A','G_M1-M1',0.2))} pt, B {pt(ex('B','G_M1-M1',0.2))} pt) at the price of more visible errors among accepted cases; the gate "
          "reuses the label threshold, so part of this advantage is by construction (appendix {fig:f8d}).")

    doc.h(2, "Deep-ensemble baseline (post-hoc extension)", "ensemble")
    cx = t10[(t10.kind == "contrast")]
    def cls(v):
        return "better" if v in ("reduces", "improves") else "worse" if v in ("increases", "worse") else "no difference" if v == "no practical difference" else "inconclusive"
    rr = []
    for H in "ABC":
        for m, bs in (("dresid_sf", budgets), ("dauroc", (None,))):
            for b in bs:
                def g(q):
                    d = cx[(cx.heldout == H) & (cx.quantity == q) & (cx.metric == m)]
                    return (d[np.isclose(d.budget, b)] if b is not None else d).iloc[0]
                dd, sc_ = (2, 100) if m == "dresid_sf" else (3, 1)
                a_, e_, f_ = g("M1-M0"), g("M1-E_fuse"), g("E_M1-E_fuse")
                rr.append({"family": H, "endpoint": "resid. SF (pt)" if m == "dresid_sf" else "dAUROC", "budget": "" if b is None else f"{int(100 * b)}%",
                           "M1 vs M0": f"{rc(a_, dd, sc_)} {MARK.get(a_.verdict, '')}", "M1 vs ensemble": f"{rc(e_, dd, sc_)} {MARK.get(e_.verdict, '')}",
                           "class": "same" if cls(a_.verdict) == cls(e_.verdict) else "CHANGES", "Mahalanobis added to ensemble": f"{rc(f_, dd, sc_)} {MARK.get(f_.verdict, '')}"})
    rr = pd.DataFrame(rr)
    nchg = int((rr["class"] == "CHANGES").sum())
    must(f"the verdict class changes in {nchg} of {len(rr)}")
    ens5 = must(f"{100 * e10('A', 'E_fuse', 'resid_sf', 0.05).estimate:.1f}% residual silent failures at 5% on A (random {100 * e10('A', 'random', 'resid_sf', 0.05).estimate:.1f}%)")
    doc.p(f"Per-seed probabilities of the other four seeds were reconstructed from saved features and averaged into a leave-seed-out ensemble (E-fuse = non-negative "
          f"fusion of ensemble uncertainties; E-M1 = E-fuse + Mahalanobis). The ensemble recovers most of M1's benefit over a single model: at a 20% budget E-fuse - M0 = "
          f"{e20('A','E_fuse-M0')} pt (A) and {e20('B','E_fuse-M0')} pt (B), and the verdict class of 'M1 vs baseline' changes in {nchg} of {len(rr)} cells ({{tab:ens}}, {{fig:f10}}). "
          f"Mahalanobis on top of the ensemble still helps on A ({e20('A','E_M1-E_fuse')} pt) and B ({e20('B','E_M1-E_fuse')} pt) but hurts on C ({e20('C','E_M1-E_fuse')} pt). "
          f"Even the ensemble leaves {ens5}.")
    doc.table(rr, "Does the deep-ensemble baseline change a conclusion? M1 vs single-model M0, M1 vs the leave-seed-out ensemble E-fuse, and Mahalanobis added to the "
              "ensemble, [95% CI] (post-hoc extension; outputs/stage10/e1_ensemble_baseline.csv).", "ens", align="lllYYlY", size="scriptsize")
    doc.fig(OUT / "stage10/figures/fig10_e1_referral.png", "Extension E1: residual silent-failure rate vs review budget for single-model M0/M1, the ensemble E-fuse "
            "and E-M1, with random and oracle references (outputs/stage10/figures).", "f10", 1.0)

    doc.h(2, "Qualitative Grad-CAM case study (post-hoc)", "cam")
    sf_m = cc[cc.kind == "silent failure"].cam_mass_top_left_2x2_cells_l4.mean(); ok_m = cc[cc.kind == "correct"].cam_mass_top_left_2x2_cells_l4.mean()
    cmc = cfg["stage11"]["cam_cases"]
    sf_txt, ok_txt = must(f"mean {sf_m:.3f}; uniform"), must(f"({ok_m:.3f})")
    cue_al = must(f"aligned {one(cm, model_kind='shortcutv2', condition='aligned').cue_mass_l3_mean:.3f}")
    doc.p(f"{{fig:f7b}} shows {cmc['n_silent']} silent failures and {cmc['n_correct']} correct predictions of the shortcut-v2 model (seed {cmc['seed']}) under the reversed cue, "
          "drawn by a fixed random rule. In the silent failures the model predicts the class coded by the cue, not the tissue, with confidence near 1, and the saved "
          f"layer4 CAM puts more mass on the 2 × 2 cells around the cue ({sf_txt} would be 0.0625) than for the correct cases "
          f"{ok_txt}. Layer3 CAM mass on the cue moves with the cue state (v2: {cue_al}, "
          f"removed {one(cm, model_kind='shortcutv2', condition='removed').cue_mass_l3_mean:.3f}): the explanation *sees* the cue, yet cue-driven errors are confident and "
          "typical. Twelve cases from one seed illustrate; they do not test.")
    doc.fig(OUT / "stage11/figures/fig7b_cam_cases.png", "Qualitative case study (post-hoc): silent failures and correct predictions (labelled in each panel title) of the "
            "shortcut-v2 model under the reversed cue; each pair shows the image with the pasted cue (dashed square) and its saved layer4 Grad-CAM (outputs/stage11/figures).", "f7b", 0.82)

    # ================================================================================ 4 answers
    doc.h(1, "Answers to the research questions", "answers")
    _a = one(ao[ao.kind == "contrast"], heldout="A", quantity="sf_M2-sf_M1", metric="dauroc")
    _c = one(ao[ao.kind == "contrast"], heldout="C", quantity="sf_M2-sf_M1", metric="dauroc")
    RQ = [
        ("RQ1 representation drift beyond uncertainty",
         f"vs single-model M0: dAUROC A {rc(t2r('A','M1-M0','dAUROC'))}, B {rc(t2r('B','M1-M0','dAUROC'))}, C {rc(t2r('C','M1-M0','dAUROC'))}; "
         f"vs the 4-seed ensemble (20% budget, residual SF): B {e20('B','M1-E_fuse')} pt, C {e20('C','M1-E_fuse')} pt (worse), A {e20('A','M1-E_fuse')} pt",
         "moderate (helps vs a single model on A and B, not vs an ensemble; harmful on C)"),
        ("RQ2 explanation drift beyond representation",
         f"monotone M2 - M1 dAUROC: A {rc(t2r('A','M2-M1','dAUROC'),4)}, B {rc(t2r('B','M2-M1','dAUROC'),4)}, C {rc(t2r('C','M2-M1','dAUROC'),4)}; "
         f"sign-free P2 (residual SF, 10%): A {must(format(g8('A','sf_M2-sf_M1',0.1).estimate*100, '+.2f'))} pt, C {must(format(g8('C','sf_M2-sf_M1',0.1).estimate*100, '+.2f'))} pt, "
         f"B {must(format(g8('B','sf_M2-sf_M1',0.1).estimate*100, '+.2f'))} pt",
         "none as a monotone alarm (strong negative result); moderate for the sign-free variant (pre-registered ablation, mechanism untested)"),
        ("RQ3 batch monitors beyond mean uncertainty",
         f"FAR check failed ({nfar_pre}/{ncell} cells outside 3-7%); at matched FAR (A1): MMD - u_mean on B {rc6(c6r(c6a,'B','r_mmd','u_mean'))}, "
         f"on A {rc6(c6r(c6a,'A','r_mmd','u_mean'))}; CAM - MMD on C {rc6(c6r(c6a,'C','e_cam_l4','r_mmd','shortcutv2'))}; combined - u_mean on A {rc6(c6r(c6a,'A','comb','u_mean'))}",
         "weak (only MMD on new hospitals beats mean uncertainty; thresholds do not transfer; A1 is post-freeze)"),
        ("RQ4 residual silent failures at a fixed budget",
         f"M1 - M0 residual SF at 20%: A {pt(t3r('A','M1-M0','dresid_sf',0.2))} pt, B {pt(t3r('B','M1-M0','dresid_sf',0.2))} pt, C {pt(t3r('C','M1-M0','dresid_sf',0.2))} pt; "
         f"at 5%: A {pt(t3r('A','M1-M0','dresid_sf',0.05))} pt (no effect); M0 - random at 5% on A {pt(t3r('A','M0-random','dresid_sf',0.05))} pt",
         "weak (only at a 20% budget; no effect at 5%; harmful on the shortcut; vanishes vs an ensemble)"),
    ]
    for _, s, _ in RQ:
        must(s, "RQ headline")
    doc.p("{tab:rq} gives each research question's headline result and a strength-of-evidence rating (strong = pre-registered primary, consistent across families "
          "and sensitivities; moderate = pre-registered but family-dependent or absorbed by a stronger baseline; weak = only in a subset of conditions or post-freeze).")
    doc.table(pd.DataFrame(RQ, columns=["research question", "headline result [95% CI]", "strength of evidence"]),
              "Research questions, headline results and strength of evidence.", "rq", align="p{2.6cm}YY", size="footnotesize")
    doc.p(f"**RQ1.** Relative to single-model uncertainty, Mahalanobis drift improves detection on corruptions and new hospitals and lowers it on the strong shortcut; "
          "the gain grows with severity. Against an ensemble the advantage disappears on B and C and is inconclusive on A. **RQ2.** As a monotone alarm explanation "
          "drift adds nothing; with a free sign it helps on A and C, because confident errors have *less* explanation drift than correct predictions. Among accepted "
          f"rows only, the sign-free AUROC change is {rc(_a)} on A and {rc(_c)} on C, so this benefit depends on the population. **RQ3.** The pre-registered FAR check "
          "failed; at matched FAR only MMD on new hospitals beats mean uncertainty. **RQ4.** No detector helps at a 5% budget; M1 helps at 20% on A and B; a "
          "confidence-gated ranking (exploratory) helps substantially on A and B.")

    # evidence map (same rule and code path as notebook Section 10c)
    EV = evidence_map(t2, t3, c6, G, t10, c6r)
    cnt = EV.groupby(["component", "source", "status"])["class"].value_counts().unstack(fill_value=0).reindex(columns=["improves", "no detectable effect", "worsens"], fill_value=0)
    cnt["cells"] = cnt.sum(axis=1); cnt["below margin"] = EV.groupby(["component", "source", "status"]).negligible.sum()
    n_imp, n_wor = int((EV["class"] == "improves").sum()), int((EV["class"] == "worsens").sum())
    ei, ew = int(((EV.component == "explanation") & (EV["class"] == "improves")).sum()), int(((EV.component == "explanation") & (EV["class"] == "worsens")).sum())
    ri, rw = int(((EV.component == "representation") & (EV["class"] == "improves")).sum()), int(((EV.component == "representation") & (EV["class"] == "worsens")).sum())
    must(f"of {len(EV)} cells, {n_imp} improve and {n_wor} worsen"); must(f"explanation drift: {ei} improve / {ew} worsen"); must(f"representation: {ri} improve / {rw} worsen")
    doc.h(2, "Evidence map (post-hoc, descriptive)", "evmap")
    doc.p(f"Every paired-bootstrap contrast of the two components in the saved tables was classified by one rule: the 95% CI excludes 0 in the favourable direction "
          f"(*improves*), in the unfavourable direction (*worsens*), or not at all. Of **{len(EV)} cells, {n_imp} improve, {n_wor} worsen and {len(EV) - n_imp - n_wor} show "
          f"no detectable effect** ({{tab:evmap}}, {{fig:evmap}}). Explanation drift: {ei} improve / {ew} worsen, almost all favourable cells coming from the sign-free "
          f"ablation and layer/form variants; representation drift: {ri} improve / {rw} worsen. {int(EV.negligible.sum())} of the {n_imp + n_wor} classified cells are "
          "below a practical margin (0.1 pt residual rate or dAURC, 0.005 AUROC/AUPRC, 1 pt power).")
    ct = cnt.reset_index()
    ct.columns = ["component", "source", "status", "improves", "no effect", "worsens", "cells", "below margin"]
    doc.table(ct, "Evidence map: endpoint cells by paired-CI outcome, per component and source.", "evmap", align="lYlrrrrr", size="footnotesize")
    evidence_figure(EV, cnt, figdir / "evidence_map.png")
    doc.fig(figdir / "evidence_map.png", "Evidence map (post-hoc, descriptive): endpoint cells that improve, show no detectable effect or worsen, per component "
            "and source (generated from the saved tables by the report build script).", "evmap", 0.95)
    doc.h(2, "Excluded-benefit bounds (post-hoc, descriptive)", "bounds")
    nd = EV[EV["class"] == "no detectable effect"].copy()
    nd["endpoint"] = nd.metric.map({"dauroc": "dAUROC", "dauprc": "dAUPRC", "daurc": "dAURC (reduction)", "dresid_sf": "residual SF (reduction, pt)", "power": "power (pt)"})
    sb = nd.groupby(["component", "source", "endpoint"])["benefit bound"].agg(cells="size", median="median", maximum="max").reset_index()
    sb["median"] = sb["median"].map(lambda v: f"{v:.4f}"); sb["maximum"] = sb["maximum"].map(lambda v: f"{v:.4f}")
    sb.columns = ["component", "source", "endpoint", "cells", "median bound", "max bound"]
    m2 = nd[(nd.component == "explanation") & nd.source.str.startswith("Phase 5")]; r2b = nd[(nd.component == "explanation") & nd.source.str.startswith("Phase 7")]
    bmax5 = must(f"larger than {m2[m2.metric == 'dauroc']['benefit bound'].max():.4f}")
    bmax7 = must(f"larger than {r2b['benefit bound'].max():.2f} pt")
    doc.p(f"For cells without a detectable effect, the favourable end of the CI is the largest benefit the data do not exclude ({{tab:bounds}}). For the pre-registered "
          f"monotone explanation detector the data exclude any AUROC gain {bmax5} and any residual-rate reduction {bmax7} at the 5-20% budgets.")
    doc.table(sb, "Largest benefit not excluded by the 95% CI over the no-detectable-effect cells (AUROC/AUPRC/AURC units; rates and power in pt).",
              "bounds", align="lYYrrr", size="footnotesize")

    # ================================================================================ 5 limitations
    doc.h(1, "Limitations and threats to validity", "limits")
    v5 = sens5[sens5.quantity.str.contains("-") & (sens5.metric == "dAUROC")]
    doc.ul([
        "**One architecture family and one attribution method.** All classifiers are ResNet-18 fine-tuned from ImageNet and all explanations are Grad-CAM; a second "
        "backbone, MC-dropout, Grad-CAM++/ScoreCAM or Integrated Gradients were not run. Results may not transfer.",
        "**Coarse explanation summaries (PCA cap).** CAMs are 8 × 8 maps; the PCA-embedding variant keeps only the components explaining 95% of the variance of the "
        "reference CAMs, so fine spatial differences are discarded. The primary JS-divergence signal is equally coarse.",
        f"**Reference bank (\"lite\") vs full training set.** References use {refn['pathmnist']:,} PathMNIST and {refn['camelyon17']:,} Camelyon training images, not the full "
        f"PathMNIST train set ({int(pm_tr.n):,}). Shrinking the bank matters little on A and somewhat on B (M1 with 500 vs the full bank at 20%: "
        f"A {pt(g8('A','M1_N500-M1_Nfull',0.2,g='G5'))} pt, B {pt(g8('B','M1_N500-M1_Nfull',0.2,g='G5'))} pt), but a bank larger than the one used was not tested.",
        f"**Accepted threshold 0.90.** The label threshold is a convention. sf_095 keeps the A result (M1 - M0 dAUROC {rc(one(v5, variant='label sf_095', heldout='A', quantity='M1-M0'))}); "
        f"the degenerate sf_tau shrinks it ({rc(one(v5, variant='label sf_tau', heldout='A', quantity='M1-M0'))}). The exploratory gate reuses the same threshold.",
        "**Post-hoc items.** The ensemble baseline, accepted-rows-only sensitivity, evidence map, excluded-benefit bounds, gated ranking and the case study were added "
        "after the pre-registered analyses; they are labelled and never replace M0/M1/M2.",
        "**Few Camelyon patients and a tumour-prior gap.** Source splits contain few patients, so Camelyon CIs are wide; train is balanced while the other splits are not.",
        "**Shortcut strength is a tuned dial**, and absolute residual rates are not comparable across families (different prevalences).",
        "**Thresholds do not transfer** (batch FAR, referral workload): any deployment would need local calibration. The negative-CAM-weight mechanism is a hypothesis.",
    ])

    # ================================================================================ 6 reproducibility
    doc.h(1, "Reproducibility", "repro")
    desc = {"phase-0": "repository, environment, configs, benchmarks", "phase-1": "data, frozen splits, leakage audit", "phase-2": "classifiers, temperature scaling",
            "phase-3": "shift families A, B, C, X", "phase-4": "references and per-image signals", "phase-5": "sample-level detection",
            "phase-6": "batch monitoring (+ amendment A1)", "phase-7": "referral policy", "phase-8": "ablations", "phase-9": "final notebook",
            "phase-9b": "notebook fixes, known gaps", "phase-10": "extensions E1/E2", "phase-final": "presentation polish",
            "phase-11a": "RQ answers, evidence map, bounds, case study", "phase-11c": "source appendix, how to reproduce"}
    tags = [t for t in git("tag", "--sort=creatordate").split() if t.startswith("phase-")]
    tg = pd.DataFrame([(t, git("rev-list", "-n1", "--abbrev-commit", t), desc.get(t, "")) for t in tags], columns=["tag", "commit", "content"])
    ntest = count_tests()
    nfiles = len([f for f in git("ls-files").splitlines() if f.startswith(("sfd/", "scripts/", "tests/"))])
    doc.p(f"Code, configuration and the small result files are in a private git repository (`sfd/` package, `scripts/` one resumable script per stage, `configs/full.yaml` "
          f"for every path and hyperparameter, `tests/`; {nfiles} tracked code files). The test suite has {ntest} tests. The final notebook `silent_failure_report.ipynb` "
          "reads only committed files and re-executes from a fresh clone in well under a minute; this PDF is generated by `scripts/build_pdf_report.py`, which cross-checks "
          "every headline number against that notebook. Data, checkpoints and per-image arrays are git-ignored (tens of GB). Phase tags are listed in {tab:tags}; "
          "this report's build script is tagged `phase-11b`.")
    doc.p("**How to rerun.** Quick check: `pip install -r requirements.txt`, then `jupyter nbconvert --to notebook --execute --inplace silent_failure_report.ipynb` and "
          "`python scripts/build_pdf_report.py --config configs/full.yaml`. Full pipeline (CUDA GPU, raw data): `python scripts/run_all.py --config configs/full.yaml`, then "
          "`run_stage3.py` ... `run_stage11.py` in order (each `--config configs/full.yaml`, resumable with `--resume`), then rebuild the notebook. A smoke test runs in minutes: "
          "`python scripts/run_all.py --config configs/smoke.yaml`.")
    doc.table(tg, "Phase tags of the repository (read from git).", "tags", align="llY", size="footnotesize")

    # ================================================================================ appendix
    doc.raw("appendix")
    doc.h(1, "Additional tables and figures", "app")
    doc.p("Full versions of results summarised in the main text; every table here is generated from the same saved files.")
    doc.table(pd.DataFrame(poprows["auprc"]), "AUPRC of silent-failure detection on all rows vs accepted rows only (post-hoc sensitivity), [95% CI] "
              "(outputs/stage11/accepted_only.csv).", "accprc", align="llrYYYYY", size="footnotesize")
    t1s = tab1[["dataset", "split", "role", "n", "n_groups"]].rename(columns={"n": "images", "n_groups": "groups (clusters / patients)"})
    t1s["images"] = t1s["images"].map(lambda v: f"{int(v):,}"); t1s["groups (clusters / patients)"] = t1s["groups (clusters / patients)"].map(lambda v: f"{int(v):,}")
    doc.table(t1s, "Datasets and splits (outputs/stage1/table1.csv).", "s1", align="llYrr", size="footnotesize")

    def far_pivot(t):
        d = t[t["size"] == 100].copy(); d["cell"] = d.H + " / " + d.kind
        p = d.pivot(index="cell", columns="stat", values="far").round(3)
        f = d.pivot(index="cell", columns="stat", values="flag_outside_3_7pct")
        return (p.astype(str) + np.where(f.astype(str) == "True", " *", "")).reset_index()
    fp = far_pivot(far6).merge(far_pivot(far6a), on="cell", suffixes=(" pre", " A1"))
    fpc = far_pivot(far6)
    doc.table(fpc, f"Actual FAR on held-out id_test windows with the pre-registered val_cal thresholds (window size {win}); * = outside 3-7% "
              "(outputs/stage6/far_table.csv).", "farpre", size="footnotesize")
    doc.table(far_pivot(far6a), "Actual FAR under amendment A1; * = outside 3-7% (outputs/stage6/a1/far_table.csv).", "fara1", size="footnotesize")
    for g, qs, ttl in (("G1", ["M1-M0", "M0c-M0", "M2-M1", "s_r_maha_pred-s_u_log1msp", "s_r_knn-s_u_log1msp", "s_e_js_norm_l4-s_u_log1msp"], "signal groups"),
                       ("G3", [f"{pre}M1+{c}-{pre}M1" for pre in ("", "sf_") for c in ("e_js_norm_l3", "e_js_norm_l2", "e_js_raw_l4", "e_pca_l4")], "CAM layer / form"),
                       ("G4", ["M1knn-M1", "M2knn-M2", "sf_M1knn-sf_M1", "sf_M2knn-sf_M2"], "Mahalanobis vs kNN"),
                       ("G5", [f"{pre}M1_N{n}-{pre}M1_Nfull" for pre in ("", "sf_") for n in (500, 1000, 2000)], "reference size"),
                       ("G6", ["M0_T1-M0", "M1_T1-M1"], "temperature scaling")):
        d = ab_compact(G[g], qs, budgets=(0.2,), MARK=MARK)
        doc.table(d, f"Ablation {g} ({ttl}): difference in residual silent-failure rate at a 20% budget, pt [95% CI] (outputs/stage8/tables/ablation_{g}.csv).",
                  f"ab{g}", align="l" + "Y" * (d.shape[1] - 1), size="footnotesize")
    exl = EX[(EX.kind == "level") & (EX.metric == "resid_sf") & EX.quantity.isin(["random", "M0", "M1", "G_M0", "G_M1", "oracle"])].copy()
    exl["val"] = [ci(e, l, h, 2, 100, False) for e, l, h in zip(exl.estimate, exl.ci_low, exl.ci_high)]; exl["B"] = (100 * exl.budget).round(0).astype(int)
    exp_ = exl.pivot_table(index=["heldout", "B"], columns="quantity", values="val", aggfunc="first").reset_index()[["heldout", "B", "random", "M0", "M1", "G_M0", "G_M1", "oracle"]]
    exp_["B"] = exp_["B"].astype(str) + "%"
    exp_.columns = ["family", "budget", "random", "M0", "M1", "gated M0", "gated M1", "oracle"]
    doc.table(exp_, "EXPLORATORY, post-hoc: residual silent-failure rate of plain vs confidence-gated rankings, % [95% CI] (outputs/stage8/tables/exploratory_targeted.csv).",
              "gated", align="llYYYYYY", size="scriptsize")
    figs = [("stage2/figures/reliability_pathmnist.png", "Reliability diagrams, PathMNIST (outputs/stage2/figures).", "relp", 0.75),
            ("stage2/figures/reliability_camelyon17.png", "Reliability diagrams, Camelyon17 (outputs/stage2/figures).", "relc", 0.75),
            ("stage3/figures/examples.png", "Example images per shift family (outputs/stage3/figures).", "examples", 1.0),
            ("stage3/figures/behaviour_families_BCX.png", "Behaviour under shift for families B, C and X (outputs/stage3/figures).", "behBCX", 1.0),
            ("stage6/a1/figures_fig5_delay.png", "Detection delay after an abrupt shift under amendment A1 (outputs/stage6/a1).", "f5d", 0.75),
            ("stage8/figures/fig8b_severity_A.png", "Ablation G7: contrasts by severity on family A (outputs/stage8/figures).", "f8b", 0.9),
            ("stage8/figures/fig8c_shortcut_states.png", "Ablation G8: contrasts by cue state on family C, v2 (outputs/stage8/figures).", "f8c", 0.9),
            ("stage8/figures/fig8d_exploratory_gated.png", "EXPLORATORY, post-hoc: residual silent-failure rate of plain vs confidence-gated rankings (outputs/stage8/figures).", "f8d", 1.0),
            ("stage4/figures/dose_response_A.png", "Dose response of the per-image signals over severity on family A (outputs/stage4/figures).", "dose", 0.6)]
    for rel, cap, lab, w in figs:
        doc.fig(OUT / rel, cap, lab, w)
    return {"tests": ntest, "head": git("rev-parse", "--short", "HEAD")}


def ab_compact(d, quantities, budgets=(0.2,), MARK=None, metric="dresid_sf"):
    """Wide table as in the notebook: one row per contrast, one column per family (and budget); cell = 'difference in pt [95% CI] mark'."""
    d = d[(d.kind == "contrast") & (d.metric == metric) & d.quantity.isin(quantities) & d.level_col.isna()]
    out = {}
    for _, r in d.iterrows():
        for b in budgets:
            if np.isclose(r.budget, b):
                col = f"{r.heldout} @{int(100 * b)}%" if len(budgets) > 1 else f"{r.heldout}"
                out.setdefault(r.quantity, {})[col] = f"{ci(r.estimate, r.ci_low, r.ci_high, 2, 100, True)} {MARK.get(r.verdict, '')}".strip()
    df = pd.DataFrame.from_dict(out, orient="index")
    df = df.loc[[q for q in quantities if q in df.index], sorted(df.columns)]
    return df.reset_index().rename(columns={"index": "contrast"})


def evidence_map(t2, t3, c6, G, t10, c6r) -> pd.DataFrame:
    """Same rule and inputs as the notebook's Section 10c."""
    LOWER = {"daurc", "dresid_sf"}
    MARGIN = {"daurc": 0.001, "dresid_sf": 0.001, "dauroc": 0.005, "dauprc": 0.005, "power": 0.01}
    ev = []
    def add(df, component, source, status):
        for _, r in df.iterrows():
            ev.append({"component": component, "source": source, "status": status, "family": r["heldout"], "metric": str(r["metric"]).lower(),
                       "budget": r["budget"] if "budget" in df.columns else np.nan, "est": r["estimate"], "lo": r["ci_low"], "hi": r["ci_high"]})
    for q, comp in {"M1-M0": "representation", "M2-M1": "explanation"}.items():
        add(t2[(t2.quantity == q) & t2.metric.isin(["dAUROC", "dAUPRC", "dAURC"])], comp, "Phase 5 sample level (dAUROC, dAUPRC, dAURC)", "pre-registered primary")
        add(t3[(t3.quantity == q) & (t3.metric == "dresid_sf")], comp, "Phase 7 referral (residual SF at 5/10/20%)", "pre-registered primary")
    for H, k in (("A", "standard"), ("B", "standard"), ("C", "shortcutv2"), ("X", "standard")):
        for (a_, b_), comp in ((("r_mmd", "u_mean"), "representation"), (("e_cam_l4", "r_mmd"), "explanation")):
            r = c6r(c6, H, a_, b_, k)
            ev.append({"component": comp, "source": "Phase 6 batch power on silent windows (pre-registered thresholds)", "status": "pre-registered primary", "family": H,
                       "metric": "power", "budget": np.nan, "est": r["diff"], "lo": r["ci_low"], "hi": r["ci_high"]})
    g2 = G["G2"]; g2 = g2[(g2.kind == "contrast") & g2.level_col.isna() & (g2.quantity == "sf_M2-sf_M1") & g2.metric.isin(["dresid_sf", "dauroc", "dauprc", "daurc"])]
    add(g2, "explanation", "Phase 8 sign-free M2 - M1 (P2)", "pre-registered primary")
    g3 = G["G3"]; g3 = g3[(g3.kind == "contrast") & g3.level_col.isna() & (g3.metric == "dresid_sf") & g3.quantity.str.contains(r"M1\+") & g3.quantity.str.endswith("M1")]
    add(g3, "explanation", "Phase 8 CAM layer/form variants (residual SF)", "secondary")
    x10 = t10[(t10.kind == "contrast") & (t10.quantity == "E_M1-E_fuse") & t10.metric.isin(["dresid_sf", "dauroc", "dauprc", "daurc"])]
    add(x10, "representation", "Phase 10 Mahalanobis added to the ensemble", "extension (post-hoc)")
    EV = pd.DataFrame(ev)
    low = EV.metric.isin(LOWER)
    better = np.where(low, EV.hi < 0, EV.lo > 0); worse = np.where(low, EV.lo > 0, EV.hi < 0)
    EV["class"] = np.where(better, "improves", np.where(worse, "worsens", "no detectable effect"))
    EV["negligible"] = (EV["class"] != "no detectable effect") & (EV.est.abs() < EV.metric.map(MARGIN))
    sc_ = np.where(EV.metric.isin(["dresid_sf", "power"]), 100.0, 1.0)
    EV["benefit bound"] = np.where(low, -EV.lo, EV.hi) * sc_
    return EV


def evidence_figure(EV, cnt, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    c_ = cnt.reset_index()
    lab = [f"{r.component}: {r.source}" for r in c_.itertuples()]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    left = np.zeros(len(c_))
    for k, col in (("improves", "#009E73"), ("no detectable effect", "#999999"), ("worsens", "#D55E00")):
        ax.barh(range(len(c_)), c_[k], left=left, color=col, label=k); left += c_[k].to_numpy()
    ax.set_yticks(range(len(c_))); ax.set_yticklabels(lab, fontsize=7); ax.invert_yaxis(); ax.set_xlabel("endpoint cells")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def count_tests() -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    assert m, r.stdout[-500:]
    return int(m.group(1))


# ============================================================================================ checks, printing, packaging
def notebook_text(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if "text" in o:
                parts.append("".join(o["text"]) if isinstance(o["text"], list) else o["text"])
            for mime in ("text/html", "text/markdown", "text/plain"):
                v = o.get("data", {}).get(mime)
                if v:
                    parts.append("".join(v) if isinstance(v, list) else v)
    assert parts, "the notebook has no outputs: execute it first"
    return html.unescape("\n".join(parts))


def consistency_check(nb_path: Path) -> tuple[int, list]:
    txt = notebook_text(nb_path)
    missing = sorted({(w, s) for w, s in CHECKS if s not in txt})
    return len({s for _, s in CHECKS}), missing


def find_browser(cands: list[str]) -> str:
    for c in cands:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    raise SystemExit("no Chrome / Edge found: set report.browsers in the config")


def print_pdf(browser: str, html_path: Path, pdf_path: Path, profile: Path) -> None:
    pdf_path.unlink(missing_ok=True)
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--no-pdf-header-footer",
           f"--user-data-dir={profile}", "--allow-file-access-from-files", "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, capture_output=True, timeout=300)
    assert pdf_path.exists() and pdf_path.stat().st_size > 10000, "PDF printing failed"


def pdf_pages(pdf: Path) -> int:
    b = pdf.read_bytes()
    n = len(re.findall(rb"/Type\s*/Page(?![s\w])", b))
    counts = [int(x) for x in re.findall(rb"/Count\s+(\d+)", b)]
    return max([n] + counts)


MAKEFILE = """# Compile the report: `make` (latexmk) or `make pdflatex` (two pdflatex passes for references).
main.pdf: main.tex tables/*.tex figures/*
\tlatexmk -pdf -interaction=nonstopmode main.tex

pdflatex:
\tpdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

clean:
\tlatexmk -C
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/full.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text(encoding="utf-8"))
    rcfg = cfg["report"]
    export = (ROOT / rcfg["export_dir"]).resolve()
    assert ROOT not in export.parents and export != ROOT, "the export directory must be outside the repository"
    export.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tex_dir = tmp / "latex"; (tex_dir / "figures").mkdir(parents=True)
        doc = Doc()
        info = build(cfg, doc, tex_dir / "figures")
        fig_names = {}
        for b in doc.blocks:
            if b[0] == "fig":
                p = b[1]["path"]
                name = p.name if p.parent == tex_dir / "figures" else p.relative_to(OUT).as_posix().replace("/figures/", "_").replace("/", "_")
                fig_names[p] = name
                if p.parent != tex_dir / "figures":
                    shutil.copy2(p, tex_dir / "figures" / name)
        today = dt.date.today()
        meta = {"title": "Silent failure detection of medical image classifiers under distribution shift",
                "subtitle": "Do representation drift and Grad-CAM explanation drift catch confident errors beyond calibrated uncertainty? "
                            "A pre-registered study on PathMNIST and Camelyon17",
                "author": rcfg["author"], "date": f"{today.day} {today.strftime('%B %Y')}",
                "footnote": f"Generated by `scripts/build_pdf_report.py` from the saved results of the repository at commit `{info['head']}`; every number is read "
                            "from the committed result tables and cross-checked against the executed notebook `silent_failure_report.ipynb`."}
        (tex_dir / "main.tex").write_text(render_latex(doc, meta, tex_dir, lambda p: fig_names[p]), encoding="ascii")
        (tex_dir / "Makefile").write_text(MAKEFILE, encoding="utf-8")
        (tex_dir / "README.txt").write_text("Compile with pdflatex/latexmk (make, or: latexmk -pdf main.tex) or upload this folder to Overleaf.\n"
                                            "Needs: lmodern, booktabs, tabularx, xltabular, caption, microtype, hyperref (all in TeX Live / MiKTeX).\n"
                                            "Generated by scripts/build_pdf_report.py; do not edit by hand.\n", encoding="utf-8")
        errs = check_tex(tex_dir)
        if errs:
            raise SystemExit("LaTeX structural check FAILED:\n" + "\n".join(errs))
        html_path = tex_dir / "report.html"
        html_path.write_text(render_html(doc, meta, lambda p: "figures/" + fig_names[p]), encoding="utf-8")
        pdf = export / rcfg["pdf_name"]
        print_pdf(find_browser(rcfg["browsers"]), html_path, pdf, tmp / "profile")
        html_path.unlink()
        zpath = export / rcfg["latex_zip"]
        zpath.unlink(missing_ok=True)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tex_dir.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tex_dir).as_posix())
    n, missing = consistency_check(NB)
    print(f"PDF: {pdf} ({pdf.stat().st_size / 1e6:.2f} MB, {pdf_pages(pdf)} pages); LaTeX zip: {zpath} ({zpath.stat().st_size / 1e6:.2f} MB)")
    print(f"LaTeX structural check: passed; tests collected: {info['tests']}")
    if missing:
        raise SystemExit(f"CONSISTENCY CHECK FAILED: {len(missing)} of {n} headline numbers not found in the notebook outputs:\n" +
                         "\n".join(f"  [{w}] {s}" for w, s in missing))
    print(f"Consistency check: passed ({n} distinct headline numbers / phrases found verbatim in the executed notebook)")


if __name__ == "__main__":
    main()
