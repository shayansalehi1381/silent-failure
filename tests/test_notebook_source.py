"""The 'Complete source code' appendix of silent_failure_report.ipynb must never go stale: every tracked source file (sfd/, scripts/, tests/,
configs/, requirements.txt) is embedded in a hidden markdown cell and must be identical to the file in the repo (line endings normalised to LF,
because git may check files out with CRLF on Windows). If this test fails, rebuild: python scripts/build_report_notebook.py and re-execute."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "silent_failure_report.ipynb"


def embedded_files() -> list[str]:
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True).stdout.split("\n")
    except Exception:                                        # not a git checkout: fall back to the file system
        tracked = [str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*") if p.is_file()]
    keep = []
    for f in tracked:
        if f.startswith(("sfd/", "scripts/", "tests/")) and f.endswith(".py") or f.startswith("configs/") or f == "requirements.txt":
            if (ROOT / f).exists():
                keep.append(f)
    return sorted(keep)


def _extract(cell: dict) -> str:
    src = "".join(cell["source"])
    lines = src.split("\n")
    assert lines[0].startswith("`") and lines[-1].startswith("`"), "embedded cell is not a fenced block"
    body = "\n".join(lines[1:-1])
    if cell["metadata"].get("eof_newline", True):
        body += "\n"
    return body


@pytest.fixture(scope="module")
def cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return {c["metadata"]["embedded_file"]: c for c in nb["cells"] if c["cell_type"] == "markdown" and "embedded_file" in c.get("metadata", {})}


def test_every_source_file_is_embedded_and_identical(cells):
    files = embedded_files()
    assert len(files) > 50, "file enumeration failed"
    missing = [f for f in files if f not in cells]
    assert not missing, f"not embedded in the notebook appendix: {missing}"
    stale = []
    for f in files:
        text = (ROOT / f).read_bytes().decode("utf-8").replace("\r\n", "\n")
        if _extract(cells[f]) != text:
            stale.append(f)
    assert not stale, f"embedded source differs from the repo (rebuild the notebook): {stale}"


def test_no_extra_files_embedded_and_cells_are_collapsed(cells):
    assert set(cells) == set(embedded_files())
    assert all(c["metadata"].get("jupyter", {}).get("source_hidden") for c in cells.values())
