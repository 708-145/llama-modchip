#!/usr/bin/env python3
"""
llama-modchip v2.1 — Zero-Dependency Interactive Terminal Chat & Shareable Benchmark Suite
Connects to local Gateway (8787) or llama-server (8789/8080/8001), streams tokens live,
and reports exact Prefill tok/s, Decode tok/s, TTFT, and Stockfish-PVS (M=7) draft acceptance.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"


def detect_endpoint(explicit_url: Optional[str] = None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    env_url = (
        os.environ.get("MODCHIP_URL")
        or os.environ.get("LLAMA_API_URL")
    )
    if env_url:
        return env_url.rstrip("/")
    gw_host = os.environ.get("GATEWAY_HOST", "").strip()
    if not gw_host:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gh_file = os.path.join(repo_root, "workspace", "run", "gateway_host")
        if os.path.isfile(gh_file):
            try:
                with open(gh_file, "r", encoding="utf-8") as f:
                    gw_host = f.readline().strip()
            except Exception:
                pass
    gw_port = int(os.environ.get("GATEWAY_PORT", "8787"))
    srv_port = int(os.environ.get("SERVER_PORT", "8789"))
    hosts = []
    if gw_host and gw_host not in ("127.0.0.1", "0.0.0.0"):
        hosts.append(gw_host)
    hosts.append("127.0.0.1")
    candidates = []
    for h in hosts:
        for p in (gw_port, srv_port, 8080, 8001):
            u = f"http://{h}:{p}"
            if u not in candidates:
                candidates.append(u)
    cfg_path = os.path.expanduser("~/.config/modchip/remote_url")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_url = f.readline().strip().rstrip("/")
                if cfg_url:
                    candidates.append(cfg_url)
        except Exception:
            pass
    for url in candidates:
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            try:
                req = urllib.request.Request(f"{url}/v1/models", method="GET")
                with urllib.request.urlopen(req, timeout=0.6) as resp:
                    if resp.status == 200:
                        return url
            except Exception:
                continue
    return "http://127.0.0.1:8787"


def detect_active_model(base_url: str) -> Tuple[str, str, str]:
    """Return (raw_model_id, display_description, prompt_label)."""
    raw_id = "nemotron-30b"
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            models = data.get("data") or data.get("models") or []
            if models and isinstance(models[0], dict):
                raw_id = models[0].get("id") or models[0].get("name") or raw_id
    except Exception:
        pass

    # If llama-server was started with --alias, /v1/models returns the alias rather than the GGUF filename;
    # inspect /props (or local backend :8789/props) to recover the actual GGUF checkpoint path.
    if not raw_id.lower().endswith(".gguf"):
        prop_urls = [f"{base_url}/props"]
        if ":8787" in base_url:
            prop_urls.append(base_url.replace(":8787", ":8789") + "/props")
        for purl in prop_urls:
            try:
                req = urllib.request.Request(purl, method="GET")
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    pdata = json.loads(resp.read().decode("utf-8", errors="replace"))
                    mpath = pdata.get("model_path") or ""
                    if mpath and mpath.lower().endswith(".gguf"):
                        raw_id = mpath
                        break
            except Exception:
                pass

    base_name = os.path.basename(raw_id)
    base_clean = re.sub(r"-0000\d-of-0000\d\.gguf$|\.gguf$", "", base_name, flags=re.IGNORECASE)
    low = base_clean.lower()
    if "deepseek" in low or "dsv4" in low:
        if "fractal" in low:
            display = f"{base_clean} (284B MoE / 13.2B Active · SwiGLU-H2 + Fractal-64 Q2_0)"
        else:
            display = f"{base_clean} (DeepSeek-V4-Flash Hybrid MLA + MoE)"
        return raw_id, display, "DeepSeek-V4-Flash"
    if "nemotron" in low:
        if "fractal" in low:
            display = f"{base_clean} (30B MoE / 3B Active · Fractal-64 Q2_0)"
        else:
            display = f"{base_clean} (30B Mamba-2 + MoE Hybrid)"
        return raw_id, display, "Nemotron-30B"
    return raw_id, base_clean or raw_id, base_clean or "Assistant"


def get_host_summary() -> Tuple[str, str, str]:
    sys_name = platform.system()
    if sys_name == "Darwin":
        cpu = "Apple Silicon"
        ram_gib = 0.0
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if brand:
                cpu = brand
        except Exception:
            pass
        try:
            mem_bytes = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
                ).strip()
            )
            ram_gib = mem_bytes / (1024.0 ** 3)
        except Exception:
            pass
        return cpu, f"{ram_gib:.0f} GB", "Unified Metal"

    if sys_name == "Windows":
        cpu = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "Windows CPU"
        ram_gib = 0.0
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                ram_gib = stat.ullTotalPhys / (1024.0 ** 3)
        except Exception:
            pass
        return cpu, f"{ram_gib:.0f} GB", "Windows VirtualAlloc"

    cpu = "Generic CPU"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    ram_gib = 0.0
    hp_gib = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_gib = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("AnonHugePages:"):
                    hp_gib = int(line.split()[1]) / (1024 * 1024)
        if ram_gib > 65536.0:
            import glob
            numa_kb = 0
            for npath in glob.glob("/sys/devices/system/node/node*/meminfo"):
                try:
                    with open(npath, "r", encoding="utf-8") as nf:
                        for nline in nf:
                            if "MemTotal:" in nline:
                                numa_kb += int(nline.split()[-2])
                except Exception:
                    pass
            if 0 < numa_kb < (65536 * 1024 * 1024):
                ram_gib = numa_kb / (1024.0 * 1024.0)
            else:
                st = os.statvfs("/dev/shm")
                shm_gib = (st.f_blocks * st.f_frsize) / (1024.0 ** 3)
                if shm_gib > 0.5:
                    ram_gib = shm_gib if shm_gib >= 24.0 else shm_gib * 2.0
    except Exception:
        pass
    return cpu, f"{ram_gib:.0f} GB", f"2MB THP: {hp_gib:.2f} GiB"


def _parse_cpu_list_fallback(spec: str):
    out = set()
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


def _format_cpu_list_fallback(cpus) -> str:
    if not cpus:
        return ""
    xs = sorted(set(cpus))
    parts = []
    start = prev = xs[0]
    for c in xs[1:]:
        if c == prev + 1:
            prev = c
        else:
            parts.append(f"{start}-{prev}" if prev - start >= 2 else (f"{start},{prev}" if prev != start else f"{start}"))
            start = prev = c
    parts.append(f"{start}-{prev}" if prev - start >= 2 else (f"{start},{prev}" if prev != start else f"{start}"))
    return ",".join(parts)


def _read_proc_argv(pid: str) -> List[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [a.decode("utf-8", errors="replace") for a in f.read().split(b"\x00") if a]
    except OSError:
        return []


def _read_proc_environ(pid: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for item in f.read().split(b"\x00"):
                if b"=" in item:
                    k, v = item.split(b"=", 1)
                    env[k.decode("utf-8", errors="replace")] = v.decode("utf-8", errors="replace")
    except OSError:
        pass
    return env


def _arg_value(argv: List[str], *names: str) -> str:
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return ""


def _is_local_url(base_url: str) -> bool:
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").strip().lower()
    if not host:
        return False
    if host in ("localhost", "0.0.0.0", "::"):
        return True
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
    except ValueError:
        pass
    # Also recognize a local LAN/Tailscale GATEWAY_HOST started on this machine
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gh_file = os.path.join(repo_root, "workspace", "run", "gateway_host")
    gw_pid_file = os.path.join(repo_root, "workspace", "run", "gateway.pid")
    try:
        if os.path.isfile(gh_file) and os.path.isfile(gw_pid_file):
            with open(gh_file, "r", encoding="utf-8") as f:
                local_gh = f.readline().strip().lower()
            with open(gw_pid_file, "r", encoding="utf-8") as f:
                gw_pid = f.read().strip()
            if local_gh and host == local_gh and gw_pid.isdigit():
                gw_argv = _read_proc_argv(gw_pid)
                if any("sse_gateway.py" in a for a in gw_argv):
                    return True
    except OSError:
        pass
    return False


def get_server_affinity_summary(base_url: str) -> str:
    """Return active local llama-server CPU affinity across all threads + -t/-tb & LP-E status on Linux."""
    if platform.system() != "Linux" or not _is_local_url(base_url):
        return ""
    import glob
    from urllib.parse import urlparse

    want_port = urlparse(base_url).port
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # If base_url points at a local sse_gateway.py instance, resolve its LLAMA_BACKEND_URL
    gw_candidates: List[str] = []
    gw_pid_file = os.path.join(repo_root, "workspace", "run", "gateway.pid")
    try:
        if os.path.isfile(gw_pid_file):
            with open(gw_pid_file, "r", encoding="utf-8") as f:
                gp = f.read().strip()
                if gp.isdigit():
                    gw_candidates.append(gp)
    except OSError:
        pass
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "sse_gateway.py"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            p = line.strip()
            if p.isdigit() and p not in gw_candidates:
                gw_candidates.append(p)
    except Exception:
        pass

    for gp in gw_candidates:
        gw_argv = _read_proc_argv(gp)
        if not any("sse_gateway.py" in a for a in gw_argv):
            continue
        gw_env = _read_proc_environ(gp)
        gw_port_s = gw_argv[-1] if gw_argv and gw_argv[-1].isdigit() else gw_env.get("GATEWAY_PORT", "8787")
        if want_port and gw_port_s.isdigit() and int(gw_port_s) == want_port:
            backend_url = gw_env.get("LLAMA_BACKEND_URL", "http://127.0.0.1:8789")
            if not _is_local_url(backend_url):
                return ""
            want_port = urlparse(backend_url).port or 8789
            break

    parse_fn = _parse_cpu_list_fallback
    format_fn = _format_cpu_list_fallback
    dropped: List[int] = []
    try:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from _cpu_topology import (
            drop_l3less_cores,
            enumerate_online_physical_cores,
            format_cpu_list,
            parse_cpu_list,
        )

        parse_fn = parse_cpu_list
        format_fn = format_cpu_list
        _, dropped = drop_l3less_cores(
            enumerate_online_physical_cores(ignore_affinity=True), respect_env=False
        )
    except Exception:
        pass

    pid_file = os.path.join(repo_root, "workspace", "run", "server.pid")
    pids: List[str] = []
    try:
        if os.path.isfile(pid_file):
            with open(pid_file, "r", encoding="utf-8") as f:
                p = f.read().strip()
                if p.isdigit():
                    pids.append(p)
    except OSError:
        pass
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "llama-server"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            p = line.strip()
            if p.isdigit() and p not in pids:
                pids.append(p)
    except Exception:
        pass

    matched_pid = ""
    matched_argv: List[str] = []
    matched_env: Dict[str, str] = {}
    for pid in pids:
        argv = _read_proc_argv(pid)
        if not argv or not os.path.basename(argv[0]).startswith("llama-server"):
            continue
        env_map = _read_proc_environ(pid)
        srv_port = _arg_value(argv, "--port") or env_map.get("LLAMA_ARG_PORT", "") or "8080"
        if want_port and srv_port.isdigit() and int(srv_port) != want_port:
            continue
        matched_pid = pid
        matched_argv = argv
        matched_env = env_map
        break

    if not matched_pid:
        return ""

    t_dec = _arg_value(matched_argv, "-t", "--threads") or matched_env.get("LLAMA_ARG_THREADS", "")
    t_batch = _arg_value(matched_argv, "-tb", "--threads-batch") or matched_env.get("LLAMA_ARG_THREADS_BATCH", "")

    per_task_sets = []
    live_cpus = set()
    try:
        # Union Cpus_allowed_list across all threads (/proc/<pid>/task/*/status):
        # /proc/<pid>/status alone only reflects thread 0, which OMP_PROC_BIND=close pins to cpu0.
        task_paths = glob.glob(f"/proc/{matched_pid}/task/*/status") or [f"/proc/{matched_pid}/status"]
        for spath in task_paths:
            try:
                with open(spath, "r", encoding="utf-8") as sf:
                    for sline in sf:
                        if sline.startswith("Cpus_allowed_list:"):
                            tset = parse_fn(sline.split(":", 1)[1].strip())
                            if tset:
                                per_task_sets.append(tset)
                                live_cpus.update(tset)
                            break
            except (OSError, ValueError):
                pass
    except OSError:
        pass

    if not live_cpus:
        return ""

    # If llama-server was started without `taskset`, idle HTTP threads retain the full host CPU mask
    # while OpenMP compute threads bind to narrower core subsets once initialized.
    max_t = max(
        int(t_dec) if t_dec.isdigit() else 0,
        int(t_batch) if t_batch.isdigit() else 0,
    )
    effective_cpus = live_cpus
    if max_t > 1 and len(live_cpus) > max_t:
        narrow_union = set()
        for tset in per_task_sets:
            if 0 < len(tset) < len(live_cpus):
                narrow_union.update(tset)
        if len(narrow_union) >= max_t:
            effective_cpus = narrow_union

    mask_str = format_fn(sorted(effective_cpus))
    if not mask_str:
        return ""

    meta_parts: List[str] = []
    if t_dec and t_batch and t_batch != t_dec:
        meta_parts.append(f"-t {t_dec}/-tb {t_batch}")
    elif t_dec:
        meta_parts.append(f"-t {t_dec}")
    elif t_batch:
        meta_parts.append(f"-tb {t_batch}")

    if dropped:
        active_lpe = sorted(set(dropped) & effective_cpus)
        if active_lpe:
            if matched_env.get("MODCHIP_KEEP_NO_L3") == "1":
                meta_parts.append(f"LP-E {active_lpe} kept via MODCHIP_KEEP_NO_L3=1")
            elif matched_env.get("TASKSET_MASK", "").strip() and (
                set(active_lpe) & parse_fn(matched_env["TASKSET_MASK"])
            ):
                meta_parts.append(f"LP-E {active_lpe} included via TASKSET_MASK")
            else:
                meta_parts.append(f"LP-E {active_lpe} active — run './modchip restart' to exclude")
        else:
            meta_parts.append(f"excluded L3-less LP-E: {dropped}")

    meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
    return f"mask={mask_str}{meta_suffix}"


def stream_chat_turn(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    think_budget: int = 0,
    quiet: bool = False,
) -> Dict[str, float]:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "nemotron-30b",
        "messages": messages,
        "stream": True,
        "temperature": 0.0,
        "repeat_penalty": 1.05,
        "repeat_last_n": 64,
        "max_tokens": max_tokens,
        "reasoning_budget": think_budget,
        "chat_template_kwargs": {"enable_thinking": think_budget != 0},
        "timings_per_token": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    chunks = 0
    full_content: List[str] = []
    in_think = False
    final_timings: Dict[str, float] = {}

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                ev = json.loads(data_str)
            except Exception:
                continue

            if "timings" in ev and isinstance(ev["timings"], dict):
                final_timings = ev["timings"]

            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            rc = delta.get("reasoning_content") or ""
            cc = delta.get("content") or ""

            if rc or cc:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                chunks += 1
                if rc:
                    if not in_think and not quiet:
                        sys.stdout.write(f"{DIM}{PURPLE}[Thinking] ")
                        in_think = True
                    if not quiet:
                        sys.stdout.write(rc)
                        sys.stdout.flush()
                if cc:
                    if in_think and not quiet:
                        sys.stdout.write(f"{RESET}\n\n")
                        in_think = False
                    full_content.append(cc)
                    if not quiet:
                        sys.stdout.write(cc)
                        sys.stdout.flush()

    if not quiet:
        sys.stdout.write(f"{RESET}\n")
        sys.stdout.flush()

    t_end = time.perf_counter()
    ttft_ms = ((t_first or t_end) - t_start) * 1000.0
    gen_sec = max(0.01, t_end - (t_first or t_start))

    dec_tps = float(final_timings.get("predicted_per_second") or (chunks / gen_sec))
    pp_tps = float(final_timings.get("prompt_per_second") or 0.0)
    pred_n = int(final_timings.get("predicted_n") or chunks)
    draft_n = int(final_timings.get("draft_n") or 0)
    draft_acc = int(final_timings.get("draft_n_accepted") or 0)
    acc_pct = (100.0 * draft_acc / draft_n) if draft_n > 0 else 0.0

    messages.append({"role": "assistant", "content": "".join(full_content)})

    return {
        "ttft_ms": ttft_ms,
        "dec_tps": dec_tps,
        "pp_tps": pp_tps,
        "pred_n": pred_n,
        "draft_n": draft_n,
        "draft_acc": draft_acc,
        "acc_pct": acc_pct,
    }


def print_turn_telemetry(stats: Dict[str, float]) -> None:
    spec_part = ""
    if stats["draft_n"] > 0:
        spec_part = (
            f" │ {PURPLE}♟️ Stockfish-PVS: {stats['draft_acc']}/{stats['draft_n']} "
            f"({stats['acc_pct']:.1f}% acc){RESET}"
        )
    pp_part = f" │ {CYAN}🧠 Prefill: {stats['pp_tps']:.1f} tok/s{RESET}" if stats["pp_tps"] > 0 else ""
    print(
        f"{DIM}{'─' * 86}{RESET}\n"
        f"{GREEN}{BOLD}⚡ {stats['dec_tps']:.2f} tok/s{RESET} "
        f"({stats['pred_n']} tok){pp_part} │ "
        f"{YELLOW}⏱️ TTFT: {stats['ttft_ms']:.0f} ms{RESET}{spec_part}\n"
        f"{DIM}{'─' * 86}{RESET}"
    )


REFACTOR_SOURCE_CODE = """```python
class HARPCacheRing:
    def __init__(self, capacity: int = 1024, page_tokens: int = 32):
        self.capacity = capacity
        self.page_tokens = page_tokens
        self.hash_to_page = {}
        self.free_pages = list(range(capacity - 1, -1, -1))
        self.evictions = 0

    def allocate_page(self, block_hash: int) -> int:
        if block_hash in self.hash_to_page:
            return self.hash_to_page[block_hash]
        if not self.free_pages:
            oldest_hash = next(iter(self.hash_to_page))
            freed_id = self.hash_to_page.pop(oldest_hash)
            self.free_pages.append(freed_id)
            self.evictions += 1
        page_id = self.free_pages.pop()
        self.hash_to_page[block_hash] = page_id
        return page_id

    def resolve_prefix(self, block_hashes: list) -> tuple:
        matched = []
        for h in block_hashes:
            if h in self.hash_to_page:
                matched.append(self.hash_to_page[h])
            else:
                break
        return len(matched) * self.page_tokens, matched
```"""


def run_benchmark_suite(base_url: str) -> None:
    is_local = _is_local_url(base_url)
    if is_local:
        cpu, ram, hp = get_host_summary()
        hw_header = f"{BOLD}{cpu}{RESET} ({ram} RAM | {hp})"
        hw_scorecard = f"{cpu} ({ram} RAM, {hp})"
    else:
        hw_header = f"{BOLD}Remote Endpoint{RESET} (run './modchip bench' on the server host for CPU/THP info)"
        hw_scorecard = f"Remote ({base_url} — run './modchip bench' on server host for CPU/THP info)"
    aff_header = get_server_affinity_summary(base_url)
    raw_model, model_display, _ = detect_active_model(base_url)
    print(f"\n{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║ ♟️  llama-modchip v2.4 — Automated 2-Stage CPU Benchmark Suite                      ║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"  • Hardware : {hw_header}")
    if aff_header:
        print(f"  • Affinity : {aff_header}")
    print(f"  • Model    : {BOLD}{GREEN}{model_display}{RESET}")
    print(f"  • Endpoint : {CYAN}{base_url}{RESET}\n")

    # Stage 1: New Code Generation
    print(f"{BOLD}[1/2] Stage 1: From-Scratch C++20 Lock-Free Code Generation (250 tokens)...{RESET}")
    msgs1 = [
        {"role": "system", "content": "You are an expert C++20 systems engineer. Reply with code only."},
        {
            "role": "user",
            "content": "Write a cache-line aligned (alignas(64)) SPSC lock-free ring buffer in C++20 using std::atomic with acquire/release memory ordering.",
        },
    ]
    s1 = stream_chat_turn(base_url, msgs1, max_tokens=250, think_budget=0, quiet=False)
    print_turn_telemetry(s1)

    # Stage 2: In-Context Code Refactoring (Stockfish-PVS Burst)
    print(f"\n{BOLD}[2/2] Stage 2: In-Context Code Refactoring — Stockfish-PVS Burst (350 tokens)...{RESET}")
    msgs2 = [
        {"role": "system", "content": "You are an expert Python systems engineer. Output the refactored class directly."},
        {
            "role": "user",
            "content": (
                "Refactor the following Python class to make it thread-safe using `threading.RLock`, "
                "add type annotations, and preserve all existing method names and logic:\n\n"
                + REFACTOR_SOURCE_CODE
            ),
        },
    ]
    s2 = stream_chat_turn(base_url, msgs2, max_tokens=350, think_budget=0, quiet=False)
    print_turn_telemetry(s2)

    pp_best = max(s1["pp_tps"], s2["pp_tps"])
    if s2["draft_n"] > 0:
        s2_spec_note = f"Stockfish-PVS Acceptance: {s2['acc_pct']:.1f}% ({s2['draft_acc']}/{s2['draft_n']})"
    else:
        s2_spec_note = "Speculative: disabled / 0 drafts"

    git_rev = ""
    try:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        git_rev = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short=7", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    ver_tag = f"v2.4 ({git_rev})" if git_rev else "v2.4"
    banner_text = f"🚀 LLAMA-MODCHIP {ver_tag} BENCHMARK SCORECARD (Share on GitHub / X!)"
    vis_w = sum(2 if ord(ch) > 0xFFFF else 1 for ch in banner_text)
    banner_line = f"║ {banner_text}{' ' * max(1, 83 - vis_w)}║"

    print(f"\n{BOLD}{GREEN}╔════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{GREEN}{banner_line}{RESET}")
    print(f"{BOLD}{GREEN}╠════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"  Hardware            : {hw_scorecard}")
    aff_note = get_server_affinity_summary(base_url)
    if aff_note:
        print(f"  CPU Affinity        : {aff_note}")
    print(f"  Active Model        : {model_display}")
    print(f"  Prompt Prefill      : {pp_best:.2f} tok/s")
    print(f"  Decode (New C++20)  : {s1['dec_tps']:.2f} tok/s (TTFT: {s1['ttft_ms']:.0f} ms)")
    print(f"  Decode (Refactor)   : {s2['dec_tps']:.2f} tok/s ({s2_spec_note})")
    print(f"{BOLD}{GREEN}╚════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"👉 Share your CPU scorecard: {CYAN}https://github.com/synchopate/llama-modchip/issues/new?template=benchmark.yml{RESET}\n")


DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful, accurate, and concise AI assistant."


def run_interactive_chat(base_url: str, one_shot_prompt: Optional[str] = None) -> None:
    cpu, ram, hp = get_host_summary()
    raw_model, model_display, prompt_label = detect_active_model(base_url)
    if one_shot_prompt:
        msgs = [
            {"role": "system", "content": DEFAULT_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": one_shot_prompt},
        ]
        stats = stream_chat_turn(base_url, msgs, max_tokens=512, think_budget=0, quiet=False)
        print_turn_telemetry(stats)
        return

    print(f"{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║ ♟️  llama-modchip v2.4 — Interactive CPU Chat + Live Stockfish-PVS Telemetry        ║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"  • Hardware : {BOLD}{cpu}{RESET} ({ram} RAM | {hp})")
    print(f"  • Model    : {BOLD}{GREEN}{model_display}{RESET}")
    print(f"  • Endpoint : {CYAN}{base_url}{RESET}")
    print(f"  • Commands : {YELLOW}/bench{RESET} │ {YELLOW}/think no|med|max{RESET} │ {YELLOW}/model{RESET} │ {YELLOW}/reset{RESET} │ {YELLOW}/exit{RESET}\n")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DEFAULT_CHAT_SYSTEM_PROMPT}
    ]
    think_budget = 0

    while True:
        try:
            user_in = input(f"{BOLD}{GREEN}You > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if user_in.lower() in ("/reset", "/clear"):
            messages.clear()
            messages.append({"role": "system", "content": DEFAULT_CHAT_SYSTEM_PROMPT})
            print(f"{DIM}[Context cleared]{RESET}\n")
            continue
        if user_in.lower() in ("/model", "/status"):
            raw_model, model_display, prompt_label = detect_active_model(base_url)
            print(f"{DIM}[Active Model: {model_display} | Raw: {raw_model} | Endpoint: {base_url}]{RESET}\n")
            continue
        if user_in.lower() == "/bench":
            run_benchmark_suite(base_url)
            continue
        if user_in.lower().startswith("/think"):
            parts = user_in.split()
            mode = parts[1].lower() if len(parts) > 1 else "no"
            think_budget = {"no": 0, "med": 1024, "max": -1}.get(mode, 0)
            print(f"{DIM}[Thinking budget set to: {mode.upper()} ({think_budget})]{RESET}\n")
            continue

        messages.append({"role": "user", "content": user_in})
        sys.stdout.write(f"\n{BOLD}{CYAN}{prompt_label} > {RESET}")
        sys.stdout.flush()
        try:
            stats = stream_chat_turn(base_url, messages, max_tokens=1024, think_budget=think_budget, quiet=False)
            print_turn_telemetry(stats)
            print()
        except urllib.error.URLError as e:
            print(f"\n{YELLOW}[!] Connection error ({base_url}): {e}{RESET}")
            print(f"    Make sure the server is running via: {BOLD}./modchip start{RESET}\n")
            messages.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="llama-modchip v2.1 Interactive Terminal Chat & Benchmark Client")
    parser.add_argument("--url", default=None, help="Server or Gateway base URL (auto-detected by default)")
    parser.add_argument("--bench", action="store_true", help="Run the 2-stage benchmark suite and print shareable scorecard")
    parser.add_argument("-p", "--prompt", default=None, help="Send a single prompt, stream output, print tok/s, and exit")
    args = parser.parse_args()

    base_url = detect_endpoint(args.url)
    if args.bench:
        run_benchmark_suite(base_url)
    else:
        run_interactive_chat(base_url, one_shot_prompt=args.prompt)


if __name__ == "__main__":
    main()
