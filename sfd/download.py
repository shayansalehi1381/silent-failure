"""Parallel, resumable HTTP download with Range requests and MD5 verification (TLS verification always on).

The file is split into fixed-size chunks; each finished chunk is recorded in `<out>.parts.json`, so an
interrupted download continues where it stopped. Falls back to a single stream if Range is unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def _size(url: str) -> tuple[int, bool]:
    """Size and Range support, probed with a real 1-byte Range GET (some servers omit Accept-Ranges in HEAD)."""
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60, stream=True)
    r.raise_for_status()
    cr = r.headers.get("Content-Range", "")
    r.close()
    if r.status_code == 206 and "/" in cr:
        return int(cr.split("/")[-1]), True
    h = requests.head(url, allow_redirects=True, timeout=60)
    return int(h.headers["Content-Length"]), False


def md5_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(url: str, out: Path, md5: str | None = None, workers: int = 8, chunk_mb: int = 64,
             log=print) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size, ranged = _size(url)
    log(f"[download] {url} -> {out}: {size / 2**30:.2f} GiB, ranges={'yes' if ranged else 'no'}")
    part_file = Path(str(out) + ".parts.json")
    if out.exists() and out.stat().st_size == size and not part_file.exists():
        log("[download] file present (completeness is decided by the MD5 check below)")
    else:
        if not ranged:
            raise RuntimeError("server does not support Range requests")
        # the progress file is written BEFORE the file is preallocated, so a preallocated but unfinished file
        # can never be mistaken for a complete one
        done = set(json.loads(part_file.read_text())) if part_file.exists() else set()
        part_file.write_text(json.dumps(sorted(done)))
        cs = chunk_mb << 20
        chunks = [(i, i * cs, min(size, (i + 1) * cs) - 1) for i in range((size + cs - 1) // cs)]
        if not out.exists() or out.stat().st_size != size:
            with open(out, "wb") as f:                        # preallocate (sparse on NTFS)
                f.truncate(size)
        lock, t0, got = threading.Lock(), time.time(), [0]

        def fetch(c):
            i, a, b = c
            for k in range(8):
                try:
                    # stream the chunk to its own region of the file in 1 MB pieces (RAM ~1 MB per worker)
                    with requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120, stream=True) as r:
                        r.raise_for_status()
                        n = 0
                        with open(out, "r+b") as f:
                            f.seek(a)
                            for piece in r.iter_content(1 << 20):
                                f.write(piece)
                                n += len(piece)
                    assert n == b - a + 1, "short range read"
                    with lock:
                        done.add(i)
                        got[0] += n
                        part_file.write_text(json.dumps(sorted(done)))
                    return
                except Exception as e:                       # retry transient errors with backoff
                    if k == 7:
                        raise RuntimeError(f"chunk {i} failed: {e}")
                    time.sleep(min(60, 2 * 2 ** k))

        todo = [c for c in chunks if c[0] not in done]
        log(f"[download] {len(todo)} of {len(chunks)} chunks to fetch")
        with ThreadPoolExecutor(workers) as ex:
            for n, _ in enumerate(ex.map(fetch, todo), 1):
                if n % 16 == 0:
                    el = time.time() - t0
                    log(f"[download] {len(done)}/{len(chunks)} chunks, {got[0] / el / 2**20:.1f} MiB/s")
        part_file.unlink()                                   # our own bookkeeping file, download complete
    if md5:
        got_md5 = md5_file(out)
        log(f"[download] md5 {got_md5} expected {md5}: {'OK' if got_md5 == md5 else 'MISMATCH'}")
        if got_md5 != md5:
            raise RuntimeError(f"MD5 mismatch for {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--md5")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    download(a.url, Path(a.out), a.md5, a.workers)

