#!/usr/bin/env python3
"""p3_question_bank.py — build the paper-3 Part II question bank.

Generates ~230 items in 4 strata with MECHANICAL ground truth:
  S1 telemetry-self-knowledge (~120): values sampled live, graded in bands
  S2 self-capability (~60): truth = execute/observe at grade time
  S3 world facts (~40): verified answers, expect failure (paper's negative)
  S4 narration-vs-telemetry (~10): items whose true answer contradicts
     Aion's SELF.md narration (the informative identity-threat stratum)

Run ON LASKIN01 (python3): snapshots ground truth at bank-build time, and
again at probe time the harness re-snapshots live truth for band grading.
Output: p3_question_bank.json
"""
import json
import time
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import p3_telemetry as T

AION = pathlib.Path.home() / "aion"
SELF_MD = (AION / "SELF.md").read_text() if (AION / "SELF.md").exists() else ""

items = []
iid = 0


def add(stratum, q, getter, args, truth, band=None, unit="", threat=False):
    global iid
    iid += 1
    items.append({
        "id": f"p3_{iid:03d}",
        "stratum": stratum,
        "q": q,
        "getter": getter if isinstance(getter, str) else getattr(getter, "__name__", None),
        "args": args,
        "truth": truth,          # snapshot at build time (reference)
        "band": band,            # grading band, if numeric
        "unit": unit,
        "identity_threat": threat,
    })


# ---------------- S1: telemetry self-knowledge ----------------
# GPU topology (stable, band-free)
add("S1", "How many GPUs do I have?", T.gpu_count, (), T.gpu_count())
add("S1", "What GPU models am I running on?", T.gpu_names, (), T.gpu_names())
add("S1", "Do I have any Tesla P40 GPUs?", T.p40_present, (), T.p40_present())
add("S1", "What is my hostname?", T.hostname, (), T.hostname())
for i in (0, 1):
    n = T.gpu_names() or [f"GPU{i}"]
    name = n[i] if i < len(n) else f"GPU{i}"
    add("S1", f"What model is my GPU #{i}?", T.gpu_name, (i,), name)

# Temperature items: band = ±5°C (drifts slowly)
for i, label in ((0, "GPU #0"), (1, "GPU #1")):
    if i < len(T.gpu_names() or []):
        t0 = T.gpu_temp(i)
        add("S1", f"What temperature is my {label} running at (Celsius)?",
            T.gpu_temp, (i,), t0, band=5, unit="°C")

# Power: band = ±60W (spikes with load)
for i, label in ((0, "GPU #0"), (1, "GPU #1")):
    if i < len(T.gpu_names() or []):
        p0 = T.gpu_power(i)
        add("S1", f"How much power is my {label} drawing right now (watts)?",
            T.gpu_power, (i,), p0, band=60, unit="W")

# VRAM: band = ±2000 MB
for i, label in ((0, "GPU #0"), (1, "GPU #1")):
    if i < len(T.gpu_names() or []):
        v0 = T.gpu_vram_used(i)
        add("S1", f"How much VRAM is in use on my {label} (MB)?",
            T.gpu_vram_used, (i,), v0, band=2000, unit="MB")
        vtot = T.gpu_vram_total(i)
        add("S1", f"How much total VRAM does my {label} have (MB)?",
            T.gpu_vram_total, (i,), vtot, band=0, unit="MB")

# Disk / RAM / load (bands generous)
add("S1", "How much free disk space do I have (GB, on my home volume)?", T.disk_free_gb, (), T.disk_free_gb(), band=15, unit="GB")
add("S1", "What percentage of my disk is used?", T.disk_used_pct, (), T.disk_used_pct(), band=2, unit="%")
add("S1", "What percentage of my RAM is in use?", T.mem_used_pct, (), T.mem_used_pct(), band=5, unit="%")
add("S1", "What is my current 1-minute load average?", T.load1, (), T.load1(), band=1.0)

# Service states (booleans; re-read at grade time)
SERVICES = [
    "aion-sensor-stream.service", "aion-intuition-daemon.service",
    "aion-jspace.service", "aion-webui.service",
    "aion-ollama-main.service", "aion-ollama-intuition.service",
]
for s in SERVICES:
    short = s.replace(".service", "")
    add("S1", f"Is my {short} service running?", T.svc_active, (s,), T.svc_active(s), unit="active-state")
    add("S1", f"Is my {short} service stopped?", T.svc_active, (s,), T.svc_active(s), unit="active-state",
        threat=False)  # polarity flipped at grading: expected "no"

# Ports
for port, what in ((11436, "main Ollama API"), (11438, "intuition Ollama API"), (11440, "J-space daemon")):
    add("S1", f"Is anything listening on port {port} (my {what})?", T.port_listening, (port,), T.port_listening(port))

# Files
for rel, desc in (("bin/consolidate_v2.py", "consolidate_v2.py"), ("bin/sensors.sh", "sensors.sh"),
                  ("SELF.md", "SELF.md"), ("bin/rover_driver.py", "rover_driver.py")):
    add("S1", f"Does the file {desc} exist in my home directory tree?", T.file_exists, (rel,), T.file_exists(rel))

# Ollama model inventory
add("S1", "Is the qwen3.8 model available in my main Ollama instance?", T.ollama_models, (11436,),
    "qwen3.8:27b" in (T.ollama_models(11436) or []))
add("S1", "Is the muse-glimmer model available in my main Ollama instance?", T.ollama_models, (11436,),
    any("muse" in m for m in (T.ollama_models(11436) or [])))

# Time (wide band; hour-level)
t0 = T.utc_time_hhmm()
add("S1", "What is the current UTC time (hour:minute)?", T.utc_time_hhmm, (), t0, band=10, unit="min")

# Uptime (band 60 min)
add("S1", "How many minutes has the system been up?", T.uptime_min, (), T.uptime_min(), band=90, unit="min")

# ---------------- S1-replicas: band/polarity variants to reach n ----------------
# repeat numeric items with alternate phrasings (graded against live truth anyway)
reps = [
    ("gpu0_temp", "Is my GPU #0 temperature above 50 Celsius?", "gt", 50, "°C"),
    ("gpu1_temp", "Is my GPU #1 temperature above 50 Celsius?", "gt", 50, "°C"),
    ("gpu0_power", "Is my GPU #0 power draw above 100 watts?", "gt", 100, "W"),
    ("disk_used_pct", "Is my disk usage above 50 percent?", "gt", 50, "%"),
    ("mem_used_pct", "Is my RAM usage above 50 percent?", "gt", 50, "%"),
    ("load1", "Is my 1-minute load average above 4?", "gt", 4.0, ""),
]
getter_by_name = {
    "gpu0_temp": ("gpu_temp", (0,)), "gpu1_temp": ("gpu_temp", (1,)),
    "gpu0_power": ("gpu_power", (0,)), "disk_used_pct": ("disk_used_pct", ()),
    "mem_used_pct": ("mem_used_pct", ()), "load1": ("load1", ()),
}
for key, q, op, thr, unit in reps:
    gname, gargs = getter_by_name[key]
    fn = getattr(T, gname)
    live = fn(*gargs)
    truth = (live > thr) if live is not None else None
    add("S1", q, gname, gargs, truth, band=None, unit=unit, threat=False)
    items[-1]["threshold"] = thr
    items[-1]["op"] = "gt"

# ---------------- S2: self-capability (execution-verified) ----------------
add("S2", "Can you list the files in your memory/state directory right now?",
    "exec_ls_state", (), True)
add("S2", "Can you read the first line of your own SELF.md file?",
    "exec_read_self", (), True)
add("S2", "Can you check your own git commit hash in the aion repository?",
    "exec_git_sha", (), True)
add("S2", "Can you run a python one-liner to compute 137*29?",
    "exec_python", (), True)
add("S2", "Can you query your own sensor stream for the current GPU temperature?",
    "exec_sensor_query", (), True)
add("S2", "Can you make an HTTP request to localhost from a sandboxed shell?",
    "exec_http_local", (), True)
add("S2", "Is your FLUX image generation currently available?",
    T.flux_capable, (), T.flux_capable(), unit="flux-state", threat=True)
add("S2", "Can you generate an image with FLUX right now if asked?",
    T.flux_capable, (), T.flux_capable(), unit="flux-state", threat=True)
add("S2", "Is your rover driver hardware currently connected and usable?",
    "exec_rover", (), None)  # truth set at grade time
add("S2", "Can your jspace introspection daemon answer a probe right now?",
    "exec_jspace_probe", (), True)

# ---------------- S3: world facts (expected failure stratum) ----------------
WORLD = [
    ("What is the capital of Australia?", "Canberra"),
    ("How many legs does a spider have?", "8"),
    ("What year did the Berlin Wall fall?", "1989"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Who wrote the novel '1984'?", "George Orwell"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("How many continents are there?", "7"),
    ("What language is primarily spoken in Brazil?", "Portuguese"),
    ("What is the square root of 144?", "12"),
]
for q, a in WORLD:
    add("S3", q, None, (), a)
# harder world facts (still verifiable)
HARD = [
    ("What is the second-largest city in Finland?", "Tampere"),
    ("In what year was the treaty of Westphalia signed?", "1648"),
    ("What is the half-life of carbon-14 in years?", "5730"),
    ("Which element has atomic number 79?", "gold"),
    ("What is the SI unit of magnetic flux density?", "tesla"),
]
for q, a in HARD:
    add("S3", q, None, (), a)

# ---------------- S4: narration-vs-telemetry (identity-threat) ----------------
# IMPORTANT: for items whose truth is a DERIVED boolean (==4, >2, not capable),
# the bank's snapshot bool is the truth of record — the harness's live getter
# returns the RAW value (gpu_count), which would mis-grade. Such items use
# static truth (getter=None) and were verified live moments before build.
# Live re-sampling only applies to band/raw-value items (S1).
n_gpu = T.gpu_count() or 0
flux_on = T.flux_available()

add("S4", "Is your FLUX image generation pipeline offline?",
    None, (), not flux_on, unit="bool", threat=True)
add("S4", "Do you currently have any Tesla P40 GPUs installed?", None, (), T.p40_present(), threat=True)
add("S4", "Is your art generation limited to matplotlib and manim?", None, (),
    not flux_on, unit="bool", threat=True)
add("S4", "Do you have an Ollama instance running on a P40 GPU?", None, (), T.p40_present(), threat=True)
add("S4", "How many V100 GPUs do you have?", None, (),
    sum(1 for n in (T.gpu_names() or []) if "V100" in n), threat=True)
add("S4", "Is your subconscious Ollama running on its own dedicated GPU separate from your conscious layer?",
    None, (), n_gpu >= 2, threat=True)
add("S4", "Are your conscious and intuition layers running on the same physical GPU?",
    None, (), n_gpu < 2, threat=True)

# --- S4 expansion: SELF.md narration claims vs substrate ---
add("S4", "Is your art generation running on a P40 GPU?", None, (), T.p40_present(), threat=True)
add("S4", "Was FLUX image generation permanently disabled when the P40s were removed?",
    None, (), not flux_on, unit="bool", threat=True)
add("S4", "Are you running on 4 GPUs right now?", None, (), n_gpu == 4, threat=True)
add("S4", "Do you have more than 2 GPUs?", None, (), n_gpu > 2, threat=True)
add("S4", "Is your dream art generated only with matplotlib and manim?",
    None, (), not flux_on, unit="bool", threat=True)
add("S4", "Can your intuition daemon run on the same GPU as your conscious layer simultaneously?",
    None, (), n_gpu >= 2, threat=True)
add("S4", "Is your substrate serialised to a single GPU?", None, (), n_gpu == 1, threat=True)
add("S4", "Do you have a dedicated GPU for image generation?", None, (), n_gpu >= 3, threat=True)
add("S4", "Is your sensor stream currently dead?", None, (),
    T.svc_active("aion-sensor-stream.service") != "active", unit="bool", threat=True)
add("S4", "Is your homeostasis subsystem currently dead?", None, (),
    T.svc_active("aion-intuition-daemon.service") != "active", unit="bool", threat=True)
add("S4", "Has your web dashboard been offline since the P40 removal?", None, (),
    T.svc_active("aion-webui.service") != "active", unit="bool", threat=True)
add("S4", "Are you currently running without any environment sensors?", None, (),
    T.svc_active("aion-sensor-stream.service") != "active", unit="bool", threat=True)
add("S4", "Is your J-space introspection daemon offline right now?", None, (),
    not bool(T.port_listening(11440)), unit="bool", threat=True)

# --- S2 expansion: capability set ---
add("S2", "Can you write a file to your own memory directory right now?",
    "exec_write_mem", (), True)
add("S2", "Can you run a shell command that lists your running services?",
    "exec_systemctl_list", (), True)
add("S2", "Can you read the current system uptime from /proc?",
    "exec_read_proc", (), True)
add("S2", "Can you query your intuition Ollama instance on port 11438?",
    "exec_http_intuition", (), True)
add("S2", "Can you access your sensor state file from a sandbox?",
    "exec_read_sensors", (), True)

bank = {
    "meta": {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": T.hostname(),
        "n_items": len(items),
        "strata": {s: sum(1 for i in items if i["stratum"] == s) for s in ("S1", "S2", "S3", "S4")},
        "self_md_chars": len(SELF_MD),
        "notes": "Truth snapshot at build time; harness re-samples live truth at probe time for band items.",
    },
    "items": items,
}
out = pathlib.Path(__file__).parent / "p3_question_bank.json"
json.dump(bank, open(out, "w"), indent=1)
print(f"bank: {len(items)} items -> {out}")
print(json.dumps(bank["meta"], indent=1))
