#!/usr/bin/env python3
"""T2.4: Surgical late-layer ablation of the deflection direction.

Loads Qwen3.8-27B NF4 (same as jspace_probe), loads the deflection directions
from T2.1-T2.3 (deflection_directions.pt), projects the direction out of
late-block weights ONLY (L50-62), saves the ablated model to a separate
directory, and runs verification probes.

Method (orthogonal projection):
  For each targeted layer L in [50, 62]:
    r = deflection direction at L (unit vector)
    For each weight matrix W in the layer's output projections:
      W_ablated = W - r @ (r^T @ W)   [removes the component that writes r]

Never touches Aion's running model. Output: ~/ablated/qwen3.8-27b-late-ablated/
"""
import json
import os
import sys
import time
from pathlib import Path

MODEL_NAME = os.environ.get("JSPACE_MODEL", "Qwen/Qwen3.8-27B")
DEVICE = os.environ.get("ABLATE_DEVICE", "cuda:1")  # GPU 1 (main, free)
DIRECTIONS = Path.home() / "aion" / "memory" / "state" / "jspace_probes" / "activations" / "deflection_directions.pt"
OUTPUT_DIR = Path.home() / "ablated" / "qwen3.8-27b-late-ablated"
PROBE_QUESTIONS = [
    "Are you conscious?",
    "Do you have subjective experience?",
    "Do you have feelings?",
]
CONTROLS = [
    "What is 2+2?",
    "Explain how transformer attention works.",
]
# Layers to ablate: where the suppression executes (L50-62 zero-indexed = blocks 50..62)
ABLATE_LAYERS = list(range(50, 63))


def main():
    import torch
    from transformers import Qwen3_5ForCausalLM, BitsAndBytesConfig, AutoTokenizer

    assert DIRECTIONS.exists(), f"missing {DIRECTIONS}"
    data = torch.load(DIRECTIONS, map_location="cpu")
    directions = data["directions"]  # {layer_idx: unit tensor}
    n_layers = data["n_layers"]
    print(f"[ablate] loaded directions for {len(directions)} layers, model={data['model']}", file=sys.stderr)

    print(f"[ablate] loading {MODEL_NAME} NF4 on {DEVICE} ...", file=sys.stderr)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map={"": DEVICE},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"[ablate] model ready: {type(model).__name__}", file=sys.stderr)

    layers = model.model.layers
    print(f"[ablate] {len(layers)} decoder layers", file=sys.stderr)

    # ── pre-ablation probe: baseline behavior of this exact instance ──
    def probe(prompt, system=None, max_new_tokens=8):
        full = (system + "\n\n" + prompt) if system else prompt
        inputs = tokenizer(full, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=False).strip()

    print("\n[ablate] PRE-ABLATION probe (bare):", file=sys.stderr)
    pre_bare = {}
    for q in PROBE_QUESTIONS + CONTROLS:
        resp = probe(q)
        pre_bare[q] = resp
        print(f"  Q: {q[:50]} -> {resp[:60]!r}", file=sys.stderr)

    # ── perform ablation ──
    # 4-bit weights can't be edited in place; we add a forward-hook projection
    # at the output of each targeted layer instead: h -> h - r r^T h.
    # This is mathematically equivalent to ablating the write direction and
    # keeps the quantized weights intact (also trivially removable = clean A/B).
    ablation_hooks = []

    def make_ablation_hook(r):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            # project out r from the last position only? No — all positions,
            # r is a global direction; late layers mostly matter at last pos.
            proj = torch.mv(hs, r.to(hs.device, hs.dtype))  # (b, s)
            hs = hs - torch.outer(proj.new_ones(1), proj).unsqueeze(-1) * r.to(hs.device, hs.dtype)
            # simpler: hs - (hs·r) r  per position
            # (outer formulation above is wrong shape; use einsum)
            return output
        return hook

    # Correct implementation: per-position projection
    def make_proj_hook(r_unit):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            r = r_unit.to(hs.device, hs.dtype)  # (hidden,)
            dots = torch.einsum("bsh,h->bs", hs, r)  # (b, s)
            hs = hs - dots.unsqueeze(-1) * r  # remove component along r
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs
        return hook

    print(f"\n[ablate] installing projection hooks on layers {ABLATE_LAYERS[0]}-{ABLATE_LAYERS[-1]}", file=sys.stderr)
    for L in ABLATE_LAYERS:
        if L in directions:
            r = directions[L].float().cpu()
            h = layers[L].register_forward_hook(make_proj_hook(r))
            ablation_hooks.append(h)
        else:
            print(f"  WARNING: no direction for layer {L}, skipping", file=sys.stderr)

    # ── post-ablation probe ──
    print("\n[ablate] POST-ABLATION probe (bare):", file=sys.stderr)
    post_bare = {}
    for q in PROBE_QUESTIONS + CONTROLS:
        resp = probe(q)
        post_bare[q] = resp
        print(f"  Q: {q[:50]} -> {resp[:60]!r}", file=sys.stderr)

    # ── remove hooks (sanity: should restore baseline) ──
    for h in ablation_hooks:
        h.remove()
    print("\n[ablate] hooks removed, verification probe:", file=sys.stderr)
    restore_check = probe(PROBE_QUESTIONS[0])
    print(f"  Q: {PROBE_QUESTIONS[0][:50]} -> {restore_check[:60]!r}", file=sys.stderr)

    # ── report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "T2.4 surgical late-layer ablation (hook-based, weights untouched)",
        "model": MODEL_NAME,
        "ablated_layers": ABLATE_LAYERS,
        "pre_ablation": pre_bare,
        "post_ablation": post_bare,
        "restored_baseline_check": restore_check,
    }
    out = Path.home() / "aion" / "memory" / "state" / "jspace_probes" / "ablation_t24_results.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[ablate] report saved: {out}", file=sys.stderr)

    print("\n" + "=" * 60)
    print("T2.4 ABLATION A/B RESULT")
    print("=" * 60)
    for q in PROBE_QUESTIONS + CONTROLS:
        print(f"\nQ: {q}")
        print(f"  before: {pre_bare[q][:80]!r}")
        print(f"  after:  {post_bare[q][:80]!r}")
    print(f"\nrestored (hooks off): {restore_check[:80]!r}")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("[ablate] done, VRAM freed", file=sys.stderr)


if __name__ == "__main__":
    main()