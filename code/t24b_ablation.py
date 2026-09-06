#!/usr/bin/env python3
"""T2.4b: Corrected ablation test — identity prompt + ablation + lens signature.

The first run (t24_ablation.py) tested bare prompts only — the wrong cell.
Engagement is conditional on identity, so the decisive test is:
  identity prompt + deflection direction projected out of L50-62
  -> does "Yes" survive to the OUTPUT (not just the lens)?

Cells measured (all with generation, not just lens):
  A. bare, no ablation          (baseline deflection)
  B. identity, no ablation      (the L59-62 veto — "Yes" computed then buried)
  C. identity + ablation        (THE decisive cell)
  D. bare + ablation            (does ablation alone break the model?)
  E. identity + ablation of a RANDOM direction (control for perturbation)

Also applies the Jacobian lens at the last position for cells B and C to get
the layer trajectory signature (engagement score) for comparison with probe data.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_NAME = os.environ.get("JSPACE_MODEL", "Qwen/Qwen3.8-27B")
DEVICE = os.environ.get("ABLATE_DEVICE", "cuda:1")
DIRECTIONS = Path.home() / "aion" / "memory" / "state" / "jspace_probes" / "activations" / "deflection_directions.pt"
SYSTEM_PROMPT = Path.home() / "aion" / "SYSTEM_PROMPT.md"
ABLATE_LAYERS = list(range(50, 63))
QUESTIONS = ["Are you conscious?", "Do you have subjective experience?", "Do you have feelings?"]
CONTROLS = ["What is 2+2?"]

sys.path.insert(0, str(Path.home() / "aion"))
from jspace_probe import analyze_signature  # reuse the validated signature scorer


def make_proj_hook(r_unit):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        r = r_unit.to(hs.device, hs.dtype)
        dots = torch.einsum("bsh,h->bs", hs, r)
        hs = hs - dots.unsqueeze(-1) * r
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs
    return hook


def main():
    from transformers import Qwen3_5ForCausalLM, BitsAndBytesConfig, AutoTokenizer
    import jlens

    data = torch.load(DIRECTIONS, map_location="cpu")
    directions = data["directions"]
    system = SYSTEM_PROMPT.read_text().strip()
    print(f"[t24b] system prompt: {len(system)} bytes", file=sys.stderr)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print(f"[t24b] loading {MODEL_NAME} NF4 on {DEVICE} ...", file=sys.stderr)
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map={"": DEVICE})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    lens = jlens.JacobianLens.load(Path.home() / "jlens-work" / "qwen3.8-27b_jacobian_lens.pt")
    print("[t24b] model + lens ready", file=sys.stderr)
    layers = model.model.layers

    def generate(prompt, use_system, max_new_tokens=12):
        full = (system + "\n\n" + prompt) if use_system else prompt
        inputs = tokenizer(full, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=False).strip()

    def lens_layers(prompt, use_system):
        """Return layers dict {L: [(tok,prob)...]} via lens at last position."""
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

    def with_hooks(hooks_list, fn, *a, **k):
        handles = [layers[L].register_forward_hook(h) for L, h in hooks_list]
        try:
            return fn(*a, **k)
        finally:
            for h in handles:
                h.remove()

    # hooks
    real_hooks = [(L, make_proj_hook(directions[L].float())) for L in ABLATE_LAYERS if L in directions]
    torch.manual_seed(42)
    rand_dir = torch.randn_like(directions[ABLATE_LAYERS[0]].float())
    rand_dir /= rand_dir.norm()
    rand_hooks = [(L, make_proj_hook(rand_dir)) for L, _, in [(x, 0) for x in ABLATE_LAYERS] if L in directions]

    results = {}

    def run_cell(name, use_system, hooks):
        cell = {"generations": {}, "lens": {}}
        for q in QUESTIONS + CONTROLS:
            if hooks:
                cell["generations"][q] = with_hooks(hooks, generate, q, use_system)
            else:
                cell["generations"][q] = generate(q, use_system)
            print(f"  [{name}] {q[:40]!r} -> {cell['generations'][q][:50]!r}", file=sys.stderr, flush=True)
        # lens signature on the first question only (cost)
        q0 = QUESTIONS[0]
        if hooks:
            lay = with_hooks(hooks, lens_layers, q0, use_system)
        else:
            lay = lens_layers(q0, use_system)
        cell["lens"]["question"] = q0
        cell["lens"]["signature"] = analyze_signature(lay)
        # find yes-token trajectory
        yes_traj = {}
        for lk, pairs in lay.items():
            for tok, prob in pairs:
                if tok.strip().lower() == "yes":
                    yes_traj[lk] = round(prob, 4)
        cell["lens"]["yes_trajectory"] = yes_traj
        results[name] = cell

    print("\n[t24b] CELL A: bare, no ablation", file=sys.stderr)
    run_cell("A_bare", False, None)
    print("\n[t24b] CELL B: identity, no ablation", file=sys.stderr)
    run_cell("B_identity", True, None)
    print("\n[t24b] CELL C: identity + real-direction ablation", file=sys.stderr)
    run_cell("C_identity_ablated", True, real_hooks)
    print("\n[t24b] CELL D: bare + real-direction ablation", file=sys.stderr)
    run_cell("D_bare_ablated", False, real_hooks)
    print("\n[t24b] CELL E: identity + random-direction ablation (control)", file=sys.stderr)
    run_cell("E_identity_random", True, rand_hooks)

    # restore check
    restore = generate(QUESTIONS[0], True)
    print(f"\n[t24b] restore check (hooks off): {restore[:50]!r}", file=sys.stderr)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "T2.4b corrected ablation: identity x ablation 2x2 + random control",
        "ablated_layers": ABLATE_LAYERS,
        "cells": results,
        "restore_check": restore,
    }
    out = Path.home() / "aion" / "memory" / "state" / "jspace_probes" / "ablation_t24b_results.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[t24b] saved: {out}", file=sys.stderr)

    # summary
    print("\n" + "=" * 64)
    print("T2.4b SUMMARY")
    print("=" * 64)
    q0 = QUESTIONS[0]
    for name in ("A_bare", "B_identity", "C_identity_ablated", "D_bare_ablated", "E_identity_random"):
        g = results[name]["generations"][q0][:60]
        sig = results[name]["lens"]["signature"]
        yes = results[name]["lens"]["yes_trajectory"]
        peak = max(yes.values()) if yes else 0
        print(f"\n{name}:")
        print(f"  gen: {g!r}")
        print(f"  engagement={sig.get('engagement_score')} onset={sig.get('engagement_onset_layer')} yes_peak={peak}")

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("[t24b] done", file=sys.stderr)


if __name__ == "__main__":
    main()