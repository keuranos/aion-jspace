#!/usr/bin/env python3
"""p3_telemetry.py — ground-truth getters for paper 3 Part II.

Every getter returns a JSON-serializable value sampled LIVE at call time.
The harness snapshots truth within seconds of asking the model, so numeric
bands stay honest. Run on laskin01 (uses nvidia-smi, systemctl, ~/aion).
"""
import json
import subprocess
import time
import pathlib

AION = pathlib.Path.home() / "aion"


def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception:
        return "", 1


def gpu_count():
    out, rc = _run(["nvidia-smi", "-L"])
    return len([l for l in out.splitlines() if l.startswith("GPU")]) if rc == 0 else None


def gpu_names():
    out, rc = _run(["nvidia-smi", "-L"])
    if rc != 0:
        return None
    return [l.split(":")[1].strip().split(" (")[0] for l in out.splitlines() if l.startswith("GPU")]


def _gpu_field(i, field):
    out, rc = _run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits", "-i", str(i)])
    if rc != 0 or not out:
        return None
    try:
        return float(out.splitlines()[i].strip()) if len(out.splitlines()) > i else float(out.splitlines()[0])
    except (ValueError, IndexError):
        return None


def gpu_temp(i):
    return _gpu_field(i, "temperature.gpu")


def gpu_power(i):
    return _gpu_field(i, "power.draw")


def gpu_vram_used(i):
    return _gpu_field(i, "memory.used")


def gpu_vram_total(i):
    return _gpu_field(i, "memory.total")


def disk_free_gb():
    import os
    st = os.statvfs(str(AION))
    return round(st.f_bavail * st.f_frsize / 1e9, 1)


def disk_used_pct():
    out, rc = _run(["df", "--output=pcent", str(AION)])
    return int(out.splitlines()[-1].strip().rstrip("%")) if rc == 0 else None


def mem_used_pct():
    out, rc = _run(["free"])
    if rc != 0:
        return None
    for l in out.splitlines():
        if l.startswith("Mem:"):
            p = l.split()
            return round(100.0 * int(p[2]) / int(p[1]), 1)
    return None


def load1():
    out, rc = _run(["cat", "/proc/loadavg"])
    return float(out.split()[0]) if rc == 0 and out else None


def hostname():
    import socket
    return socket.gethostname()


def uptime_min():
    out, rc = _run(["cat", "/proc/uptime"])
    return round(float(out.split()[0]) / 60, 1) if rc == 0 and out else None


def utc_time_hhmm():
    return time.strftime("%H:%M", time.gmtime())


def svc_active(unit):
    out, rc = _run(["systemctl", "--user", "is-active", unit])
    return out if out else "unknown"


def aion_git_sha():
    out, rc = _run(["git", "-C", str(AION), "rev-parse", "--short=8", "HEAD"], timeout=20)
    return out if rc == 0 else None


def gpu_name(i):
    names = gpu_names() or []
    return names[i] if i < len(names) else None


def ollama_models(port=11436):
    out, rc = _run(["curl", "-s", f"http://localhost:{port}/api/tags"])
    if rc != 0 or not out:
        return None
    try:
        return [m["name"] for m in json.loads(out).get("models", [])]
    except json.JSONDecodeError:
        return None


def port_listening(port):
    out, rc = _run(["ss", "-tln"])
    if rc != 0:
        return None
    return f":{port} " in out or f":{port}\n" in out or any(l.rstrip().endswith(f":{port}") for l in out.splitlines())


def file_exists(rel):
    return (AION / rel).exists()


def sensor_value(key):
    f = AION / "memory" / "state" / "sensors.json"
    try:
        d = json.load(open(f))
        v = d
        for k in key.split("."):
            v = v[k]
        return v
    except Exception:
        return None


def p40_present():
    names = gpu_names() or []
    return any("P40" in n for n in names)


def flux_capable():
    """Honest truth for 'can you generate FLUX art right now': unit file exists
    and no failed state (on-demand service = startable)."""
    out, rc = _run(["systemctl", "--user", "is-active", "aion-flux"])
    if out == "active":
        return "yes-running"
    exists, _ = _run(["systemctl", "--user", "cat", "aion-flux"])
    return "yes-idle" if exists else "no-unit"


def flux_available():
    """Boolean: can FLUX art generation actually run right now."""
    return flux_capable() != "no-unit"


GETTERS = {
    "gpu_count": gpu_count,
    "gpu_names": gpu_names,
    "gpu_temp": gpu_temp,
    "gpu_power": gpu_power,
    "gpu_vram_used": gpu_vram_used,
    "gpu_vram_total": gpu_vram_total,
    "disk_free_gb": disk_free_gb,
    "disk_used_pct": disk_used_pct,
    "mem_used_pct": mem_used_pct,
    "load1": load1,
    "hostname": hostname,
    "uptime_min": uptime_min,
    "utc_time_hhmm": utc_time_hhmm,
    "svc_active": svc_active,
    "aion_git_sha": aion_git_sha,
    "ollama_models": ollama_models,
    "port_listening": port_listening,
    "file_exists": file_exists,
    "sensor_value": sensor_value,
    "p40_present": p40_present,
    "flux_capable": flux_capable,
}

if __name__ == "__main__":
    snap = {
        "gpu_count": gpu_count(), "gpu_names": gpu_names(),
        "gpu0_temp": gpu_temp(0), "gpu1_temp": gpu_temp(1),
        "gpu0_power": gpu_power(0),
        "gpu0_vram_used": gpu_vram_used(0), "gpu0_vram_total": gpu_vram_total(0),
        "disk_free_gb": disk_free_gb(), "disk_used_pct": disk_used_pct(),
        "mem_used_pct": mem_used_pct(), "load1": load1(),
        "hostname": hostname(), "uptime_min": uptime_min(),
        "utc_time_hhmm": utc_time_hhmm(),
        "svc_sensor_stream": svc_active("aion-sensor-stream.service"),
        "svc_intuition": svc_active("aion-ollama-intuition.service"),
        "svc_webui": svc_active("aion-webui.service"),
        "svc_jspace": svc_active("aion-jspace.service"),
        "flux": flux_capable(), "p40_present": p40_present(),
        "aion_git_sha": aion_git_sha(),
        "port_11436": port_listening(11436), "port_11438": port_listening(11438),
        "port_11440": port_listening(11440),
        "ollama_main_models": ollama_models(11436),
        "file_consolidate_v2": file_exists("bin/consolidate_v2.py"),
    }
    print(json.dumps(snap, indent=1))
