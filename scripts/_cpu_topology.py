#!/usr/bin/env python3
"""
Shared Linux sysfs CPU & cache topology helpers for llama-modchip.
Safely enumerates online non-SMT physical cores, excludes minority L3-less SoC-tile
LP-E cores (e.g. Intel Meteor Lake / Arrow Lake-H) without dropping 50/50 designs
(e.g. Lunar Lake), and estimates static/dynamic effective compute cores on hybrid CPUs.
"""

import glob
import os
import re
from typing import Dict, List, Optional, Set, Tuple


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def parse_cpu_list(spec: str) -> Set[int]:
    out: Set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            if lo_s.isdigit() and hi_s.isdigit():
                out.update(range(int(lo_s), int(hi_s) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def format_cpu_list(cpus: List[int]) -> str:
    """Format sorted CPU IDs into compact Linux taskset range string (e.g. '0,2,4,6,8-15')."""
    if not cpus:
        return ""
    xs = sorted(set(cpus))
    parts: List[str] = []
    start = prev = xs[0]
    for c in xs[1:]:
        if c == prev + 1:
            prev = c
        else:
            parts.append(f"{start}-{prev}" if prev - start >= 2 else (f"{start},{prev}" if prev != start else f"{start}"))
            start = prev = c
    parts.append(f"{start}-{prev}" if prev - start >= 2 else (f"{start},{prev}" if prev != start else f"{start}"))
    return ",".join(parts)


def discover_l3_domains() -> Dict[str, int]:
    """Return {shared_cpu_list: size_bytes} for all Data/Unified L3 caches in sysfs."""
    domains: Dict[str, int] = {}
    for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cache/index[0-9]*")):
        lvl = _read(os.path.join(d, "level"))
        typ = _read(os.path.join(d, "type"))
        if lvl != "3" or typ not in ("Data", "Unified"):
            continue
        s_list = _read(os.path.join(d, "shared_cpu_list"))
        sz_str = (_read(os.path.join(d, "size")) or "").upper()
        if not s_list or not sz_str:
            continue
        try:
            if sz_str.endswith("K"):
                sz_bytes = int(sz_str[:-1]) * 1024
            elif sz_str.endswith("M"):
                sz_bytes = int(sz_str[:-1]) * 1024 * 1024
            elif sz_str.endswith("G"):
                sz_bytes = int(sz_str[:-1]) * 1024 * 1024 * 1024
            else:
                sz_bytes = int(sz_str)
            domains[s_list] = sz_bytes
        except ValueError:
            pass
    return domains


def enumerate_online_physical_cores(ignore_affinity: bool = False) -> List[int]:
    """Return sorted list of 1 logical CPU ID per online physical core (intersected with sched_getaffinity unless ignore_affinity=True)."""
    online_str = _read("/sys/devices/system/cpu/online")
    online_set = parse_cpu_list(online_str) if online_str else set()
    aff: Set[int] = set()
    if not ignore_affinity:
        try:
            aff = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            aff = set()

    core_map: Dict[Tuple[int, int, int], int] = {}
    for cpu_dir in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*")):
        m = re.search(r"cpu(\d+)$", cpu_dir)
        if not m:
            continue
        cid_num = int(m.group(1))
        if online_set and cid_num not in online_set:
            continue
        if aff and cid_num not in aff:
            continue
        pkg_s = _read(os.path.join(cpu_dir, "topology/physical_package_id"))
        die_s = _read(os.path.join(cpu_dir, "topology/die_id")) or "0"
        core_s = _read(os.path.join(cpu_dir, "topology/core_id"))
        if pkg_s is not None and core_s is not None:
            try:
                key = (int(pkg_s), int(die_s) if die_s.lstrip("-").isdigit() else 0, int(core_s))
                if key not in core_map or cid_num < core_map[key]:
                    core_map[key] = cid_num
            except ValueError:
                pass
    if core_map:
        return sorted(core_map.values())
    if aff:
        return sorted(aff)
    if online_set:
        return sorted(online_set)
    return list(range(os.cpu_count() or 8))


def _cache_levels(cid: int) -> Optional[Set[int]]:
    """Data/Unified cache levels listed for cpu<cid>; None if no usable cacheinfo."""
    dirs = glob.glob(f"/sys/devices/system/cpu/cpu{cid}/cache/index[0-9]*")
    levels: Set[int] = set()
    for d in dirs:
        lvl = _read(os.path.join(d, "level"))
        typ = _read(os.path.join(d, "type"))
        if lvl and lvl.isdigit() and typ in ("Data", "Unified"):
            levels.add(int(lvl))
    return levels or None


def _virtualized() -> bool:
    cpuinfo = _read("/proc/cpuinfo") or ""
    return bool(re.search(r"\bhypervisor\b", cpuinfo))


def drop_l3less_cores(
    phys_cpus: List[int], log=None, respect_env: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Exclude off-ring, L3-less cores (e.g. Intel SoC-tile LP-E on Meteor/Arrow Lake-H).
    Returns (kept_cpus, dropped_cpus).
    """
    if (
        (respect_env and os.environ.get("MODCHIP_KEEP_NO_L3") == "1")
        or (respect_env and os.environ.get("TASKSET_MASK", "").strip() != "")
        or _virtualized()
        or len(phys_cpus) < 5
    ):
        return phys_cpus, []
    info = {c: _cache_levels(c) for c in phys_cpus}
    if any(v is None for v in info.values()):
        return phys_cpus, []  # incomplete cacheinfo: do not guess
    with_l3 = [c for c in phys_cpus if 3 in info[c]]  # type: ignore[operator]
    no_l3 = [c for c in phys_cpus if 3 not in info[c] and 2 in info[c]]  # type: ignore[operator]
    if (
        len(with_l3) >= 4
        and no_l3
        and len(with_l3) + len(no_l3) == len(phys_cpus)
        and 2 * len(no_l3) <= len(with_l3)  # minority rule: never drop half the CPU (e.g. Lunar Lake)
    ):
        if log is not None:
            log(
                f"[topology] excluding L3-less cores {no_l3} (SoC-tile LP-E); "
                f"set MODCHIP_KEEP_NO_L3=1 to keep them"
            )
        return with_l3, no_l3
    return phys_cpus, []


def estimate_hybrid_effective_cores(phys_cpus: List[int]) -> Tuple[float, float, float, bool]:
    """
    Inspect per-core cpuinfo_max_freq on Linux to detect heterogeneous P+E topologies.
    Weights each core by (freq / max_freq) * vector_width_factor (0.60 on Intel E-cores
    whose 128-bit/lower-throughput SIMD pipelines split 256-bit AVX2 ops vs P-cores;
    1.0 on uniform-IPC designs such as AMD Zen5 + Zen5c).
    Returns (static_bound_cores, dynamic_bound_cores, blended_cores, is_hybrid).
    """
    n = float(len(phys_cpus) or 1)
    freqs: List[int] = []
    for c in phys_cpus:
        s = _read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq")
        if s and s.isdigit() and int(s) > 0:
            freqs.append(int(s))
    if len(freqs) != len(phys_cpus) or not freqs:
        return n, n, n, False
    max_f = float(max(freqs))
    min_f = float(min(freqs))
    if min_f >= 0.88 * max_f:
        return n, n, n, False
    is_intel = "GenuineIntel" in (_read("/proc/cpuinfo") or "")
    weights: List[float] = []
    for f in freqs:
        clk = f / max_f
        vec = 0.60 if (is_intel and clk < 0.88) else 1.0
        weights.append(clk * vec)
    dyn_cores = sum(weights)
    stat_cores = len(weights) * min(weights)
    blended = 0.50 * stat_cores + 0.50 * dyn_cores
    return stat_cores, dyn_cores, blended, True
