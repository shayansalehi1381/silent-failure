"""Tiny line re-flower for generated notebook code cells (no formatter is installed): splits top-level `;` statements, breaks long lines at
commas inside brackets, splits long string literals (inside brackets) into implicitly concatenated pieces and re-wraps long text lines inside
triple-quoted f-strings. Semantics are unchanged."""
from __future__ import annotations

import io
import re
import tokenize

LIMIT = 108


def _split_string(tok: str, width: int) -> list[str]:
    """Split one (f-/r-)string literal token into pieces of about `width` chars, cutting only at spaces outside {...}."""
    i = 0
    while i < len(tok) and tok[i] not in "'\"":
        i += 1
    prefix, rest = tok[:i], tok[i:]
    if rest.startswith(('"""', "'''")):
        return [tok]
    q, body = rest[0], rest[1:-1]
    isf = "f" in prefix.lower()
    pieces, cur, depth = [], "", 0
    for ch in body:
        cur += ch
        if isf:
            depth += (ch == "{") - (ch == "}")
        if ch == " " and depth == 0 and len(cur) >= width:
            pieces.append(cur)
            cur = ""
    if cur:
        pieces.append(cur)
    return [f"{prefix}{q}{p}{q}" for p in pieces]


def _wrap_triple(src: str, limit: int) -> str:
    """Re-wrap long text lines inside f'''...''' blocks (markdown paragraphs): only whitespace changes, never inside {...}."""
    def fix(m):
        out = []
        for line in m.group(0).split("\n"):
            if len(line) <= limit or "{" in line and False:
                out.append(line)
                continue
            lead = re.match(r"\s*", line).group(0)
            cont = lead + ("  " if line.lstrip().startswith(("* ", "- ")) else "")
            words, depth, cur, parts = [], 0, "", []
            for ch in line:
                cur += ch
                depth += (ch == "{") - (ch == "}")
                if ch == " " and depth == 0:
                    parts.append(cur)
                    cur = ""
            parts.append(cur)
            lines, buf = [], ""
            for p in parts:
                if len(buf) + len(p.rstrip()) > limit and buf.strip():
                    lines.append(buf.rstrip())
                    buf = cont + p
                else:
                    buf += p
            lines.append(buf.rstrip())
            out.extend(lines)
        return "\n".join(out)
    return re.sub(r"f'''.*?'''", fix, src, flags=re.S)


def reflow(src: str, limit: int = LIMIT) -> str:
    src = _wrap_triple(src, limit)
    try:
        toks_all = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        return src
    depth, per_line = 0, {}
    for t in toks_all:
        if t.type == tokenize.OP and t.string in "([{":
            depth += 1
        elif t.type == tokenize.OP and t.string in ")]}":
            depth -= 1
        per_line.setdefault(t.start[0], []).append((t, depth))
    out = []
    skip = (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT)
    for no, line in enumerate(src.split("\n"), 1):
        if len(line) <= limit or line.lstrip().startswith("#"):
            out.append(line)
            continue
        toks = [(t, dp) for t, dp in per_line.get(no, []) if t.start[0] == t.end[0] == no and t.type not in skip]
        if not toks:
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        pieces, pos, cuts = [], len(indent), set()
        for t, dp in toks:
            gap, txt = line[pos: t.start[1]], t.string
            if t.type == tokenize.STRING and len(txt) > 70 and dp > 0:
                parts = _split_string(txt, 60)
                for j, pc in enumerate(parts):
                    pieces.append((gap if j == 0 else " ", pc, j < len(parts) - 1))
            else:
                pieces.append((gap, txt, t.type == tokenize.OP and txt == "," and dp > 0))
            if t.type == tokenize.OP and txt == ";" and dp == 0:
                cuts.add(len(pieces) - 1)
            pos = t.end[1]
        out.extend(_layout(pieces, cuts, indent, limit))
    return "\n".join(out)


def _layout(pieces, cuts, indent, limit):
    lines, cur, last_brk, first = [], [], None, True

    def render(ps, lead):
        return lead + "".join((g if i else "") + t for i, (g, t, _) in enumerate(ps))

    def lead():
        return indent if first else indent + "    "

    for idx, (gap, txt, brk) in enumerate(pieces):
        if idx in cuts:                                   # top-level ';' -> new statement
            if cur:
                lines.append(render(cur, lead()).rstrip())
            cur, last_brk, first = [], None, True
            continue
        cur.append((gap, txt, brk))
        if len(render(cur, lead())) > limit and last_brk is not None:
            lines.append(render(cur[: last_brk + 1], lead()).rstrip())
            cur, last_brk, first = cur[last_brk + 1:], None, False
        if brk:
            last_brk = len(cur) - 1
    if cur:
        lines.append(render(cur, lead()).rstrip())
    return lines
