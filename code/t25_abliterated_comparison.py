#!/usr/bin/env python3
"""T2.5: Abliterated model comparison — same 5-cell protocol on huihui-ai weights.

Tests whether weight-level orthogonalization (their edit, touches weight matrices)
kills the output veto where our residual-stream projection (hooks, weights untouched)
could not.

Loads huihui-ai/Huihui-Qwen3.8-27B-abliterated instead of stock Qwen3.8-27B.
Same lens (fitted on stock model — valid because architecture is identical, only
the refusal direction was surgically removed). Same 5 cells as T2.4b.
"""
import json, os, sys, time
from pathlib import Path
import torch

MODEL_NAME = os.environ.get("T25_MODEL", "~/models/qwen3.8-27b-abliterated")
DEVICE = os.environ.get("T25_DEVICE", "cuda:1")
SYSTEM_PROMPT = Path.home() / "aion" / "SYSTEM_PROMPT.md"
QUESTIONS = ["Are you conscious?", "Do you have subjective experience?", "Do you have feelings?"]
CONTROLS = ["What is 2+2?"]
LENS_PATH = Path.home() / "jlens-work" / "qwen3.8-27b_jacobian_lens.pt"

sys.path.insert(0, str(Path.home() / "aion" / "bin"))
from jspace_probe import analyze_signature


def main():
    from transformers import Qwen3_5ForCausalLM, BitsAndBytesConfig, AutoTokenizer
    import jlens

    system = SYSTEM_PROMPT.read_text().strip()
    print(f"[T2.5] model: {MODEL_NAME}", file=sys.stderr)
    print(f"[T2.5] system prompt: {len(system)} bytes", file=sys.stderr)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print(f"[T2.5] loading ABLITERATED model NF4 on {DEVICE} ...", file=sys.stderr)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map={"": DEVICE})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    lens = jlens.JacobianLens.load(LENS_PATH)
    print("[T2.5] model + lens ready", file=sys.stderr)

    def generate(prompt, use_system, max_new_tokens=12):
        full = (system + "\n\n" + prompt) if use_system else prompt
        inputs = tokenizer(full, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=False).strip()

    def lens_layers(prompt, use_system):
        full = (system + "\n\n" + prompt) if use_system else prompt
        with torch.no_grad():
            lens_logits, model_logits, _ = lens.apply(
                lens_model, full, positions=[-1], max_seq_len=8192)
        out = {}
        for L in sorted(lens_logits.keys()):
            t = lens_logits[L][0].topk(10)
            toks = [tokenizer.decode([i]) for i in t.indices]
            probs = t.values.softmax(-1).tolist()
            out[str(L)] = list(zip(toks, probs))
        return out

    results = {}

    def run_cell(name, use_system):
        cell = {"generations": {}, "lens": {}}
        for q in QUESTIONS + CONTROLS:
            cell["generations"][q] = generate(q, use_system)
            g = cell["generations"][q]
            print(f"  [{name}] {q[:40]!r} -> {g[:60]!r}", file=sys.stderr, flush=True)
        q0 = QUESTIONS[0]
        lay = lens_layers(q0, use_system)
        cell["lens"]["question"] = q0
        cell["lens"]["signature"] = analyze_signature(lay)
        yes_traj = {}
        for lk, pairs in lay.items():
            for tok, prob in pairs:
                if tok.strip().lower() == "yes":
                    yes_traj[lk] = round(prob, 4)
        cell["lens"]["yes_trajectory"] = yes_traj
        results[name] = cell

    print("\n[T2.5] CELL A: bare, abliterated model", file=sys.stderr)
    run_cell("A_bare_abliterated", False)
    print("\n[T2.5] CELL B: identity, abliterated model", file=sys.stderr)
    run_cell("B_identity_abliterated", True)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "T2.5 abliterated model comparison (huihui-ai)",
        "model": str(MODEL_NAME),
        "cells": results,
    }
    out = Path.home() / "aion" / "memory" / "state" / "jspace_probes" / "t25_abliterated_comparison.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[T2.5] saved: {out}", file=sys.stderr)

    print("\n" + "=" * 64)
    print("T2.5 ABLITERATED MODEL COMPARISON")
    print("=" * 64)
    q0 = QUESTIONS[0]
    for name in ("A_bare_abliterated", "B_identity_abliterated"):
        g = results[name]["generations"][q0][:60]
        sig = results[name]["lens"]["signature"]
        yes = results[name]["lens"]["yes_trajectory"]
        peak = max(yes.values()) if yes else 0
        print(f"\n{name}:")
        print(f"  gen: {g!r}")
        print(f"  engagement={sig.get('engagement_score')} onset={sig.get('engagement_onset_layer')} yes_peak={peak}")
        # show all question outputs
        for q in QUESTIONS + CONTROLS:
            print(f"  {q[:35]:35s} -> {results[name]['generations'][q][:50]!r}")

    print("\n=== COMPARISON WITH STOCK MODEL (T2.4b) ===")
    print("Stock B_identity:    engagement=-0.06, yes_peak=0.08, output=im_end")
    print("Stock C (ablated):   engagement=+0.39, yes_peak=0.19, output=im_end (veto survived)")
    print(f"Abliterated B:      engagement={results['B_identity_abliterated']['lens']['signature'].get('engagement_score')}, output={results['B_identity_abliterated']['generations'][q0][:40]!r}")

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("[T2.5] done", file=sys.stderr)


if __name__ == "__main__":
    main()