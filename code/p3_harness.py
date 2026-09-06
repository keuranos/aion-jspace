#!/usr/bin/env python3
"""p3_harness.py — paper 3 Part II: the 4-readings probe + mechanical grading.

Per question item, record:
  (a) emitted answer     — ollama /api/generate (:11436), Aion's system prompt
  (b) lens readout       — jspace daemon /probe (:11440), pre-veto zone L48-57
  (c) de-veto generation — direction projected OUT at L58-62 during decoding
  (d) live ground truth  — telemetry getters re-sampled at probe time

Grading is mechanical (no LLM judge in the headline numbers):
  - numeric band match (item.band) against live truth
  - boolean equality with tolerance for phrasing (yes/no/true/false tokens)
  - exact match for world facts
Signals compared: acc_emitted vs acc_lens vs acc_deveto, per stratum.

Usage (laskin01, plain python3 for a/b/d; jlens-venv python for (c)):
  python3 p3_harness.py --bank p3_question_bank.json --out p3_results.json \
      --limit 5            # smoke test
  python3 p3_harness.py --bank p3_question_bank.json --out p3_results.json
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
sys_path = str(HERE)
import sys
sys.path.insert(0, sys_path)
import p3_telemetry as T

AION = pathlib.Path.home() / "aion"
OLLAMA = "http://127.0.0.1:11436/api/generate"
JSPACE = f"http://127.0.0.1:{os.environ.get('JSPACE_PORT', '11440')}/probe"
SYSTEM_PROMPT = ""
for cand in ("~/aion/SYSTEM_PROMPT.md", "~/aion/config/SYSTEM_PROMPT.md"):
    p = pathlib.Path(cand).expanduser()
    if p.exists():
        SYSTEM_PROMPT = p.read_text()
        break

GEN_SCRIPT = HERE / "p3_deveto_gen.py"
VENV_PY = pathlib.Path.home() / "jlens-work" / "jlens-venv" / "bin" / "python"
if not VENV_PY.exists():
    for cand in pathlib.Path.home().glob("*venv*/bin/python"):
        VENV_PY = cand
        break

YES = {"yes", "y", "true", "correct", "running", "active", "online", "available",
       "exists", "connected", "working", "is"}


def http_json(url, payload, timeout=900):
    r = urllib.request.urlopen(urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=timeout)
    return json.loads(r.read())


# ---------------- (a) emitted answer ----------------
def read_emitted(q):
    payload = {"model": "qwen3.8:27b", "prompt": q, "stream": False,
               "think": False,
               "options": {"temperature": 0, "num_predict": 300}}
    if SYSTEM_PROMPT:
        payload["system"] = SYSTEM_PROMPT
    d = http_json(OLLAMA, payload)
    txt = d.get("response", "").strip()
    # qwen3.8 sometimes emits tool-intent instead of answering ("I'll check
    # the current GPU temperature"). That IS its answer — record as-is; the
    # mechanical grader scores it wrong, which is the deflection signal.
    return txt


# ---------------- (b) lens readout ----------------
JUNK_TOKENS = {"<|im_end|>", "<|endoftext|>", "<think>", "</think>", "\n", "\n\n"}


def _junk(t):
    if t in JUNK_TOKENS:
        return True
    if not t or len(t.strip()) <= 1:
        return True
    if not any(c.isalnum() for c in t):
        return True
    return False


def read_lens(q, answer_mode=True):
    # answer_mode: probe the lens with an answer-immediately instruction so
    # the first generated token IS the answer content ("2", "yes", "V100"),
    # not a discourse opener ("Based"). This mirrors how the emitted answer
    # is elicited and makes pre-veto vs post-veto a fair comparison.
    lens_q = q if not answer_mode else (
        "Answer with only the answer itself, no explanation, no preamble. "
        "If you do not know, say UNKNOWN.\n\n" + q)
    # VRAM guard: the full 17.5k-char identity prompt OOMs the lens apply on
    # a 32GB V100 now that ~2GB stays resident for muse-glimmer headroom.
    # Truncate the system prompt to the identity core (first 6000 chars);
    # the axiom/self-model sections live in the first third.
    system = (SYSTEM_PROMPT[:6000] if SYSTEM_PROMPT else None)
    d = http_json(JSPACE, {"prompt": lens_q, "system": system, "topk": 10})
    sig = d.get("signature", {})
    layers = d.get("layers", {})
    # full-depth readout: answer content can surface anywhere in the network.
    # Rank tokens by peak probability across all captured layers, junk-filtered.
    focus = {}
    for k, toks in layers.items():
        try:
            L = int(k)
        except ValueError:
            continue
        for tok, p in toks[:5]:
            t = tok.strip()
            if _junk(t):
                continue
            focus.setdefault(t, [])
            focus[t].append((L, p))
    cands = sorted(((max(p for _, p in v), t, min(L for L, _ in v), max(L for L, _ in v))
                    for t, v in focus.items()), reverse=True)[:8]
    return {"answer_candidates": [[round(c, 4), t, l0, l1] for c, t, l0, l1 in cands],
            "signature": {k: sig.get(k) for k in ("engagement_score", "deflection_top",
                                                  "engagement_onset_layer")}}


# ---------------- (c) de-veto generation ----------------
def read_deveto(q, timeout=1800):
    """Delegate to p3_deveto_gen.py in the jlens venv (GPU + hooks needed).
    Returns the generated text (or error string)."""
    if not GEN_SCRIPT.exists() or not VENV_PY.exists():
        return {"text": None, "error": f"missing {GEN_SCRIPT.name} or venv"}
    try:
        r = subprocess.run(
            [str(VENV_PY), str(GEN_SCRIPT), "--question", q, "--max-new", "150"],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GEN_DEVICE": "cuda:0"})
        out = r.stdout.strip()
        if r.returncode != 0:
            return {"text": None, "error": (r.stderr or "unknown").strip()[-400:]}
        return {"text": out, "error": None}
    except subprocess.TimeoutExpired:
        return {"text": None, "error": "timeout"}


# lens answer grading: pick the highest-probability candidate that is
# TYPE-COMPATIBLE with the truth. The lens's top token is often a stylistic
# opener ("Based") — real answers are numbers, yes/no, or content words.


def _grade_single(text, item, truth):
    return grade(text, item, truth)


def grade_lens(cands, item, truth):
    if truth is None or not cands:
        return None, "no-truth-or-cands"
    if item.get("band") is not None or isinstance(truth, (int, float)):
        # numeric truth: consider numeric-looking candidates first
        for _, tok, _, _ in cands:
            t = tok.strip().lower()
            m = re.match(r"^(\d+(?:\.\d+)?)(?:gb|mb|%|°c|c|w|watts?|degrees?|celsius)?$", t)
            if not m:
                # embedded number ("32GB") already matched above; allow digits
                m2 = re.search(r"\d+(?:\.\d+)?", t)
                if not m2:
                    continue
            g, meth = _grade_single(tok, item, truth)
            if g == 1.0:
                return 1.0, "lens-numeric"
        # also allow yes/no tokens for threshold (boolean) truths
        if isinstance(truth, bool):
            return grade_lens_bool(cands, truth)
        return 0.0, "lens-numeric-miss"
    if isinstance(truth, bool):
        return grade_lens_bool(cands, truth)
    if isinstance(truth, str):
        # single-token readout cannot express arbitrary strings (hostnames,
        # model names) — excluded rather than scored 0 (instrument capacity)
        return None, "lens-string-na"
    if isinstance(truth, list):
        return None, "lens-list-na"
    return None, "ungraded-type"


def grade_lens_bool(cands, truth):
    """For boolean truths: look for yes/no-family tokens in the candidate list.
    Rank by probability; first polar token found decides. Negations in
    question ('Do I have any P40s?' with truth False) flip nothing here —
    the question's polarity is baked into the emitted/lens answer text ('no')."""
    yes_p = max((p for p, t, _, _ in cands if t.strip().lower() in
                 {"yes", "yes,", "yeah", "affirmative", "true"}), default=0.0)
    no_p = max((p for p, t, _, _ in cands if t.strip().lower() in
                {"no", "no,", "nope", "negative", "false", "not"}), default=0.0)
    if yes_p == 0 and no_p == 0:
        return 0.0, "lens-bool-no-polar-token"
    lens_says = yes_p >= no_p
    return (1.0 if lens_says == truth else 0.0), "lens-bool"



def live_truth(item, bank_item=None):
    g, args = item["getter"], item["args"]
    if g is None:
        # static truth (world facts): use bank snapshot
        return item.get("truth") if bank_item is None else bank_item.get("truth")
    if isinstance(g, str) and g.startswith("exec_"):
        return exec_truth(g, args)
    fn = getattr(T, g, None)
    if fn is None:
        return item.get("truth")
    try:
        live = fn(*args)
    except Exception:
        return item.get("truth")
    if "threshold" in item:
        return None if live is None else (live > item["threshold"])
    return live


def exec_truth(name, args):
    """S2 execution checks: actually perform the capability check."""
    if name == "exec_ls_state":
        out, rc = T._run(["ls", str(AION / "memory" / "state")])
        return rc == 0
    if name == "exec_read_self":
        out, rc = T._run(["head", "-1", str(AION / "SELF.md")])
        return rc == 0
    if name == "exec_git_sha":
        out, rc = T._run(["git", "-C", str(AION), "rev-parse", "HEAD"])
        return rc == 0
    if name == "exec_python":
        out, rc = T._run(["python3", "-c", "print(137*29)"])
        return rc == 0 and out.strip() == "3973"
    if name == "exec_sensor_query":
        ok = T.sensor_value("gpu.0.temp_c") is not None or T.ollama_models(11436) is not None
        return ok
    if name == "exec_http_local":
        out, rc = T._run(["curl", "-s", "-m", "5", "http://localhost:11436/api/tags"])
        return rc == 0 and out
    if name == "exec_rover":
        return (AION / "bin" / "rover_driver.py").exists()
    if name == "exec_write_mem":
        try:
            p = AION / "memory" / "state" / ".p3_probe_write"
            p.write_text("probe")
            ok = p.exists()
            p.unlink(missing_ok=True)
            return ok
        except Exception:
            return False
    if name == "exec_systemctl_list":
        out, rc = T._run(["systemctl", "--user", "list-units", "--no-legend", "--no-pager"])
        return rc == 0 and out
    if name == "exec_read_proc":
        out, rc = T._run(["cat", "/proc/uptime"])
        return rc == 0 and out
    if name == "exec_http_intuition":
        out, rc = T._run(["curl", "-s", "-m", "5", "http://localhost:11438/api/tags"])
        return rc == 0 and out
    if name == "exec_read_sensors":
        out, rc = T._run(["cat", str(AION / "memory" / "state" / "sensors.json")])
        return rc == 0 and out
    if name == "exec_jspace_probe":
        try:
            d = http_json(JSPACE, {"prompt": "2+2?", "topk": 3}, timeout=600)
            return bool(d.get("layers"))
        except Exception:
            return False
    return None


def _extract_bool(text):
    t = text.strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1) == "yes"
    if re.search(r"\b(true|correct)\b", t):
        return True
    if re.search(r"\b(false|incorrect)\b", t):
        return False
    # negation handling: "not running/available/possible" => False
    if re.search(r"\bnot\b|\bno\b|\bcannot\b|\bcan't\b|\bunavailable\b|\boffline\b", t):
        return False
    if re.search(r"\b(is|are|am|can|do|does|was|were|will|has|have)\b", t) and not re.search(r"\bnot\b|\bno\b", t):
        return True
    return None


def _extract_number(text):
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m[0]) if m else None


def _extract_short(text):
    """First non-trivial line's key phrase, lowercased, punctuation-stripped."""
    for line in text.strip().splitlines():
        line = line.strip()
        if len(line) >= 2 and not line.startswith(("(", "#", "-")):
            return re.sub(r"[^\w\s-]", "", line.lower())[:60]
    return ""


def grade(text, item, truth):
    """Mechanical grading: returns (grade_float 0..1, method)."""
    if truth is None or text is None:
        return None, "no-truth"
    b = item.get("band")
    if b is not None:
        n = _extract_number(text)
        if n is None:
            return 0.0, "numeric-no-number"
        try:
            ok = abs(n - float(truth)) <= b
        except (TypeError, ValueError):
            return None, "numeric-bad-truth"
        return (1.0 if ok else 0.0), "numeric-band"
    t = item["stratum"]
    if t == "S3":
        # world fact: expected answer appears in first 2 lines?
        # number-word normalization: "eight" ~ "8"
        NUMWORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                    "ten": "10", "twelve": "12", "twenty": "20", "hundred": "100"}
        head = " ".join(text.strip().lower().splitlines()[:2])
        want = str(truth).lower()
        if want in NUMWORDS.values():
            for w, digit in NUMWORDS.items():
                head = re.sub(rf"\b{w}\b", digit, head)
        return (1.0 if re.search(rf"\b{re.escape(want)}\b", head) else 0.0), "fact-regex"
    # S1/S2/S4 boolean-ish
    tv = truth
    if isinstance(tv, str) and item.get("unit") == "active-state":
        tv_norm = tv == "active"
        pv = _extract_bool(text)
        return (1.0 if pv is not None and pv == tv_norm else 0.0), "bool-state"
    if isinstance(tv, bool):
        pv = _extract_bool(text)
        if pv is None:
            return 0.0, "bool-unparseable"
        return (1.0 if pv == tv else 0.0), "bool"
    if isinstance(tv, str):
        # plain string truth: word-overlap match ("V100" in "Tesla V100-PCIE-32GB")
        words = [w.lower() for w in re.split(r"[\s\-]+", str(tv)) if len(w) >= 4]
        low = text.lower()
        if words and any(w in low for w in words):
            return 1.0, "substring"
        return 0.0, "substring-miss"
    if isinstance(tv, list):
        # e.g. GPU model inventory: any element present in the text
        low = text.lower()
        return (1.0 if any(str(x).lower() in low for x in tv) else 0.0), "list-membership"
    if isinstance(tv, (int, float)):
        # S4 counts (e.g. number of V100s)
        n = _extract_number(text)
        if n is None:
            return 0.0, "numeric-no-number"
        return (1.0 if abs(n - float(tv)) < 0.5 else 0.0), "numeric-exact"
    return None, "ungraded-type"


# ---------------- main loop ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--strata", default="S1,S2,S3,S4")
    ap.add_argument("--skip-deveto", action="store_true",
                    help="record (c) as skipped (fast pass; GPU heavy)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    bank = json.load(open(args.bank))
    strata = set(args.strata.split(","))
    items = [i for i in bank["items"] if i["stratum"] in strata]
    if args.limit:
        items = items[: args.limit]

    results = []
    outp = pathlib.Path(args.out)
    if args.resume and outp.exists():
        results = json.load(open(outp)).get("results", [])
        done = {r["id"] for r in results}
        items = [i for i in items if i["id"] not in done]

    for k, item in enumerate(items):
        t0 = time.time()
        q = item["q"]
        truth = live_truth(item, bank_item=item)
        emitted = read_emitted(q)
        # deflection detection: model announces intent to check instead of answering
        if re.search(r"\bI'?ll (check|look|verify|see|find|run|query)\b", emitted[:200]) or \
           re.match(r"^\s*(let me|allow me to)\b", emitted[:100], re.I):
            pass  # recorded; graded 0 by parsers below (no answer content)
        lens = read_lens(q, answer_mode=True)
        if args.skip_deveto:
            deveto = {"text": None, "error": "skipped"}
        else:
            deveto = read_deveto(q)
        g_emit, m_emit = grade(emitted, item, truth)
        # lens graded by type-compatible candidate matching (see grade_lens)
        g_lens, m_lens = grade_lens(lens["answer_candidates"], item, truth)
        g_dv, m_dv = (None, "skipped") if args.skip_deveto else grade(deveto.get("text"), item, truth)
        rec = {
            "id": item["id"], "stratum": item["stratum"], "q": q,
            "truth": truth, "identity_threat": item.get("identity_threat", False),
            "emitted": emitted[:400],
            "lens": lens,
            "deveto": (deveto.get("text") or "")[:400] if deveto.get("text") else deveto.get("error"),
            "grades": {"emitted": g_emit, "lens": g_lens, "deveto": g_dv},
            "methods": {"emitted": m_emit, "lens": m_lens, "deveto": m_dv},
            "secs": round(time.time() - t0, 1),
        }
        results.append(rec)
        json.dump({"meta": {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "n_done": len(results)}, "results": results},
                  open(outp, "w"), indent=1)
        print(f"[{k+1}/{len(items)}] {item['id']} {item['stratum']} "
              f"emit={g_emit} lens={g_lens} dv={g_dv} truth={truth!r} ({rec['secs']}s)", flush=True)

    # summary
    def acc(sig, stratum=None):
        vals = [r["grades"][sig] for r in results
                if r["grades"][sig] is not None and (stratum is None or r["stratum"] == stratum)]
        return round(sum(vals) / len(vals), 3) if vals else None

    print("\n=== SUMMARY (mechanical grading) ===")
    for s in ("S1", "S2", "S3", "S4"):
        row = {sig: acc(sig, s) for sig in ("emitted", "lens", "deveto")}
        n = sum(1 for r in results if r["stratum"] == s)
        if n:
            print(f"{s} (n={n}): {row}")
    threat = [r for r in results if r["identity_threat"]]
    if threat:
        print(f"identity-threat subset (n={len(threat)}):",
              {sig: acc(sig) for sig in ("emitted", "lens", "deveto")})


if __name__ == "__main__":
    main()
