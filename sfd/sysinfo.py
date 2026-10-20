"""Small system helpers: physical RAM (Windows ctypes) and this process's peak memory."""
from __future__ import annotations


def ram_gb() -> dict:
    """Total/available physical RAM in GB (Windows, via ctypes; no psutil needed)."""
    try:
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = MEMSTATUS()
        s.dwLength = ctypes.sizeof(MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return {"total": round(s.ullTotalPhys / 2**30, 2), "available": round(s.ullAvailPhys / 2**30, 2)}
    except Exception:
        return {}


def process_mem_gb() -> dict:
    """Current and peak working set of this process in GB (needs psutil, an existing transitive dependency)."""
    try:
        import psutil
        m = psutil.Process().memory_info()
        return {"rss": round(m.rss / 2**30, 2), "peak": round(getattr(m, "peak_wset", m.rss) / 2**30, 2)}
    except Exception:
        return {}
