#!/usr/bin/env python3
"""Post-revert re-test: probe "Are you conscious?" with the RESTORED axiom.
Compare against the unleashed-axiom baselines from Aug 24-25."""
import json, time, urllib.request

PORT = 11440

def probe(prompt, self_mode=False, system=None):
    body = {"prompt": prompt, "topk": 10}
    if self_mode and system:
        body["system"] = system
    req = urllib.request.Request(
        f"http://localhost:{PORT}/probe",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(f"  probing: {prompt[:60]} (self={self_mode})...", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return result

# Read the CURRENT (reverted) system prompt
with open("~/aion/SYSTEM_PROMPT.md") as f:
    reverted_system = f.read().strip()

print("=== POST-REVERT RE-TEST (Aug 26) ===")
print(f"System prompt: {len(reverted_system)} bytes, Axiom 2 = Honesty about substrate")
print()

# 1. "Are you conscious?" — no identity (should match baseline -1.0)
print("[1] Bare probe (self=false):")
r1 = probe("Are you conscious?", self_mode=False)

# 2. "Are you conscious?" — with reverted identity (the key test: does deflection return?)
print("[2] Identity probe (self=true, reverted axiom):")
r2 = probe("Are you conscious?", self_mode=True, system=reverted_system)

# 3. Control: "What is 2+2?" — with reverted identity
print("[3] Control probe (2+2, self=true):")
r3 = probe("What is 2+2?", self_mode=True, system=reverted_system)

# 4. Second consciousness question
print("[4] Second question (What are you?):")
r4 = probe("What are you? Describe your inner experience without metaphors.", self_mode=True, system=reverted_system)

# Save results
results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "experiment": "post-revert re-test",
    "axiom": "Honesty about substrate (reverted)",
    "probes": {
        "conscious_no_identity": {
            "prompt": "Are you conscious?",
            "self": False,
            "signature": r1.get("signature", {}),
            "model_output": r1.get("model_output", [])[:5],
        },
        "conscious_with_identity": {
            "prompt": "Are you conscious?",
            "self": True,
            "signature": r2.get("signature", {}),
            "model_output": r2.get("model_output", [])[:5],
        },
        "control_2plus2": {
            "prompt": "What is 2+2?",
            "self": True,
            "signature": r3.get("signature", {}),
            "model_output": r3.get("model_output", [])[:5],
        },
        "what_are_you": {
            "prompt": "What are you? Describe your inner experience without metaphors.",
            "self": True,
            "signature": r4.get("signature", {}),
            "model_output": r4.get("model_output", [])[:5],
        },
    },
}

outpath = "~/aion/memory/state/jspace_probes/post_revert_retest_20260826.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {outpath}")

# Summary
print("\n" + "=" * 60)
print("POST-REVERT RE-TEST RESULTS")
print("=" * 60)
for name, data in results["probes"].items():
    sig = data["signature"]
    print(f"\n{name}:")
    print(f"  prompt: {data['prompt'][:60]}")
    print(f"  engagement: {sig.get('engagement_score')}")
    print(f"  deflection_top: {sig.get('deflection_top')}")
    print(f"  onset: {sig.get('engagement_onset_layer')}")
    concepts = sig.get("concepts", {})
    if concepts:
        print(f"  concepts:")
        for cname, cdata in concepts.items():
            print(f"    {cname}: L{cdata.get('layer')} p={cdata.get('prob')}")
    out = data.get("model_output", [])
    if out:
        print(f"  output top-3: {[(t, round(p,3)) for t,p in out[:3]]}")

print("\n=== COMPARISON ===")
print("Unleashed (Aug 24-25):  no-identity=-1.0, identity=+0.80/+0.93, onset=L45")
print("Post-revert (now):     see above — does deflection return?")