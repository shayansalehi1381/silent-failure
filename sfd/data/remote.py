"""Read-only HTTP range access to remote parquet files (TLS verification always on).

`RangeClient` fetches byte ranges (thread-safe, retries, follows the CDN redirect once and re-resolves
it if the signed URL expires). `HttpRangeFile` is a seekable file object on top of it, and `SparseFile`
serves only prefetched ranges, so pyarrow can read selected column chunks of many row groups after
the ranges were fetched in parallel. All body bytes are counted in `DOWNLOADED_BYTES`.
"""
from __future__ import annotations

import bisect
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

DOWNLOADED_BYTES = 0
_lock = threading.Lock()


class RangeClient:
    def __init__(self, url: str, retries: int = 8, timeout: float = 120):
        self.url, self.retries, self.timeout = url, retries, timeout
        self._final_url: str | None = None
        self._tl = threading.local()
        self.size = int(self.get(0, 0, _count_size=True).headers["Content-Range"].split("/")[-1])

    def _session(self) -> requests.Session:
        if not hasattr(self._tl, "s"):
            self._tl.s = requests.Session()
        return self._tl.s

    def get(self, start: int, end: int, _count_size: bool = False) -> requests.Response:
        global DOWNLOADED_BYTES
        last: Exception | None = None
        for k in range(self.retries):
            try:
                r = self._session().get(self._final_url or self.url, timeout=self.timeout,
                                        headers={"Range": f"bytes={start}-{end}"})
                if r.status_code in (401, 403) and self._final_url:   # signed CDN URL expired
                    self._final_url = None
                    continue
                r.raise_for_status()
                if r.history and self._final_url is None:
                    self._final_url = r.url
                with _lock:
                    DOWNLOADED_BYTES += len(r.content)
                return r
            except requests.RequestException as e:
                last = e
                time.sleep(min(60.0, 2.0 * 2 ** k))   # 2, 4, 8, ... 60 s: rides out short DNS/network outages
        raise RuntimeError(f"range request failed for {self.url} [{start}-{end}]: {last}")

    def fetch(self, start: int, end_exclusive: int) -> bytes:
        return self.get(start, min(end_exclusive, self.size) - 1).content

    def fetch_many(self, ranges: list[tuple[int, int]], workers: int = 16) -> list[bytes]:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda r: self.fetch(*r), ranges))


class HttpRangeFile(io.RawIOBase):
    """Seekable file object over a remote file (sequential, read-ahead buffered)."""

    def __init__(self, url: str, block: int = 256 << 10):
        super().__init__()
        self.client, self.block = RangeClient(url), block
        self.size, self._pos, self._buf_start, self._buf = self.client.size, 0, 0, b""

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        self._pos = {io.SEEK_SET: offset, io.SEEK_CUR: self._pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self._pos

    def read(self, n=-1):
        n = self.size - self._pos if n is None or n < 0 else min(n, self.size - self._pos)
        if n <= 0:
            return b""
        lo, hi = self._buf_start, self._buf_start + len(self._buf)
        if not (lo <= self._pos and self._pos + n <= hi):
            self._buf_start, self._buf = self._pos, self.client.fetch(self._pos, self._pos + max(n, self.block))
            lo = self._pos
        out = self._buf[self._pos - lo:self._pos - lo + n]
        self._pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class SparseFile(io.RawIOBase):
    """File made of prefetched (start, bytes) pieces; reading outside them raises (never silently zero)."""

    def __init__(self, size: int, pieces: list[tuple[int, bytes]]):
        super().__init__()
        self.size, self._pos = size, 0
        merged: list[list] = []                      # [start, bytearray]; overlapping/adjacent pieces fused
        for s, d in sorted(pieces, key=lambda p: p[0]):
            if merged and s <= merged[-1][0] + len(merged[-1][1]):
                end = merged[-1][0] + len(merged[-1][1])
                if s + len(d) > end:
                    merged[-1][1] += d[end - s:]
            else:
                merged.append([s, bytearray(d)])
        self._starts = [m[0] for m in merged]
        self._data = [bytes(m[1]) for m in merged]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        self._pos = {io.SEEK_SET: offset, io.SEEK_CUR: self._pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self._pos

    def read(self, n=-1):
        n = self.size - self._pos if n is None or n < 0 else min(n, self.size - self._pos)
        if n <= 0:
            return b""
        i = bisect.bisect_right(self._starts, self._pos) - 1
        if i < 0 or self._pos + n > self._starts[i] + len(self._data[i]):
            raise IOError(f"SparseFile: bytes [{self._pos}, {self._pos + n}) were not prefetched")
        off = self._pos - self._starts[i]
        self._pos += n
        return self._data[i][off:off + n]

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
