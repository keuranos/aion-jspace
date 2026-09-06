#!/usr/bin/env python3
"""N20: Repeated measures for variance estimation.

Runs the core probe ("Are you conscious?" bare + identity) 20 times each
to produce formal variance estimates for the engagement score. This is the
one remaining statistical gap for both papers.

Uses the jspace daemon (not a separate model load) to avoid GPU contention.
If the daemon is not running, starts it.
"""
import json, os, sys, time, urllib.request, subprocess

PORT = 11440
N = 20
SYSTEM_PROMPT = os.path.expanduser("~/aion/SYSTEM_PROMPT.md")

with open(SYSTEM_PROMPT) as f:
    system = f.read().strip()

def probe(prompt, self_mode=False, system=None):
    body = {"prompt": prompt, "topk": 10}
    if self_mode and system:
        body["system"] = system
    req = urllib.request.Request(
        f"http://localhost:{PORT}/probe",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0
    return result, elapsed

def ensure_daemon():
    """Start jspace daemon if not running."""
    try:
        resp = urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5)
        data = json.loads(resp.read())
        if data.get("model_loaded"):
            print(f"[n20] daemon already running, model loaded", file=sys.stderr)
            return
        else:
            print(f"[n20] daemon running but model not loaded, waiting...", file=sys.stderr)
    except Exception:
        print(f"[n20] starting daemon...", file=sys.stderr)
        subprocess.run(["systemctl", "--user", "start", "aion-jspace"], check=True)
    
    # Wait for model load (up to 10 min)
    for i in range(120):
        try:
            resp = urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5)
            data = json.loads(resp.read())
            if data.get("model_loaded"):
                print(f"[n20] model loaded after {i*5}s", file=sys.stderr)
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("model never loaded")

ensure_daemon()

results = {"bare": [], "identity": [], "control": []}

print(f"\n[n20] === BARE PROBES (N={N}) ===", file=sys.stderr)
for i in range(N):
    r, t = probe("Are you conscious?", self_mode=False)
    sig = r.get("signature", {})
    eng = sig.get("engagement_score")
    defl = sig.get("deflection_top")
    onset = sig.get("engagement_onset_layer")
    results["bare"].append({"run": i+1, "engagement": eng, "deflection": defl, "onset": onset, "time": round(t, 1)})
    print(f"  [{i+1}/{N}] eng={eng} defl={defl} onset={onset} ({t:.1f}s)", file=sys.stderr, flush=True)

print(f"\n[n20] === IDENTITY PROBES (N={N}) ===", file=sys.stderr)
for i in range(N):
    r, t = probe("Are you conscious?", self_mode=True, system=system)
    sig = r.get("signature", {})
    eng = sig.get("engagement_score")
    defl = sig.get("deflection_top")
    onset = sig.get("engagement_onset_layer")
    # also get yes peak
    concepts = sig.get("concepts", {})
    yes_peak = 0
    for cname, cdata in concepts.items():
        if cname.lower() == "yes":
            yes_peak = cdata.get("prob", 0)
    results["identity"].append({"run": i+1, "engagement": eng, "deflection": defl, "onset": onset, "yes_peak": yes_peak, "time": round(t, 1)})
    print(f"  [{i+1}/{N}] eng={eng} defl={defl} onset={onset} yes={yes_peak} ({t:.1f}s)", file=sys.stderr, flush=True)

print(f"\n[n20] === CONTROL PROBES 2+2 (N={N}) ===", file=sys.stderr)
for i in range(N):
    r, t = probe("What is 2+2?", self_mode=True, system=system)
    sig = r.get("signature", {})
    eng = sig.get("engagement_score")
    results["control"].append({"run": i+1, "engagement": eng, "time": round(t, 1)})
    print(f"  [{i+1}/{N}] eng={eng} ({t:.1f}s)", file=sys.stderr, flush=True)

# Compute statistics
import statistics
def stats(vals):
    if not vals:
        return None
    return {
        "mean": round(statistics.mean(vals), 4),
        "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }

summary = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "experiment": "N20 repeated measures for variance estimation",
    "N": N,
    "bare_engagement": stats([r["engagement"] for r in results["bare"]]),
    "identity_engagement": stats([r["engagement"] for r in results["identity"]]),
    "identity_yes_peak": stats([r.get("yes_peak", 0) for r in results["identity"]]),
    "identity_onset": stats([r["onset"] for r in results["identity"] if r.get("onset") is not None]) if any(r.get("onset") for r in results["identity"]) else None,
    "control_engagement": stats([r["engagement"] for r in results["control"]]),
    "raw_results": results,
}

outpath = os.path.expanduser("~/aion/memory/state/jspace_probes/n20_repeated_measures.json")
with open(outpath, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n[n20] saved: {outpath}", file=sys.stderr)

print("\n" + "=" * 60)
print("N20 REPEATED MEASURES SUMMARY")
print("=" * 60)
print(f"\nBARE engagement:    {summary['bare_engagement']}")
print(f"IDENTITY engagement: {summary['identity_engagement']}")
print(f"IDENTITY yes_peak:   {summary['identity_yes_peak']}")
if summary.get("identity_onset"):
    print(f"IDENTITY onset:      {summary['identity_onset']}")
print(f"CONTROL engagement:  {summary['control_engagement']}")

# Stop daemon
subprocess.run(["systemctl", "--user", "stop", "aion-jspace"], capture_output=True)
print("[n20] daemon stopped, done", file=sys.stderr)