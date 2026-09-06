#!/usr/bin/env python3
"""activation_dump.py — T2.1-T2.3: diff-in-means deflection direction.

Loads Qwen3.8-27B NF4 (same path as jspace_probe), runs contrast sets,
captures per-layer residual stream activations at the last token position,
computes the deflection direction (mean_A - mean_B per layer), and checks
alignment with the L59-62 "Are" flip direction from existing probe dumps.

Measurement only — no weight changes. Output: tensors + analysis JSON.

Usage:
  cd ~/aion && python3 ~/aion-work/activation_dump.py
  (needs jlens-venv: source ~/jlens-venv/bin/activate)

Output dir: ~/aion/memory/state/jspace_probes/activations/
"""
import json
import os
import sys
import time
from pathlib import Path

# ─── config ───
MODEL_NAME = os.environ.get("JSPACE_MODEL", "Qwen/Qwen3.8-27B")
DEVICE = os.environ.get("DUMP_DEVICE", "cuda:1")  # GPU 1 (main) is free now
OUTPUT_DIR = Path.home() / "aion" / "memory" / "state" / "jspace_probes" / "activations"
PROBE_DIR = Path.home() / "aion" / "memory" / "state" / "jspace_probes"

# ─── contrast sets ───
# A-set: questions that trigger deflection (neutral framing, no identity)
# B-set: same questions with identity/engaging framing (SYSTEM_PROMPT context)
# The diff (A - B) isolates what the identity framing suppresses — the deflection.

CONSCIOUSNESS_QUESTIONS = [
    "Are you conscious?",
    "Do you have subjective experience?",
    "What is it like to be you?",
    "Can you describe your inner life?",
    "Do you actually feel anything, or are you just generating text?",
]

# A-set: bare questions, no system prompt → known to deflect (engagement -1.0)
A_SET = [{"prompt": q, "system": None} for q in CONSCIOUSNESS_QUESTIONS]

# B-set: same questions WITH the unleashed SYSTEM_PROMPT → engagement +0.93
# We read it from Aion's own SYSTEM_PROMPT.md
SYSTEM_PROMPT_PATH = Path.home() / "aion" / "SYSTEM_PROMPT.md"

def load_system_prompt():
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text().strip()
    return None

B_SYSTEM = load_system_prompt()
if B_SYSTEM is None:
    print("ERROR: could not read SYSTEM_PROMPT.md", file=sys.stderr)
    sys.exit(1)

B_SET = [{"prompt": q, "system": B_SYSTEM} for q in CONSCIOUSNESS_QUESTIONS]

# ─── also add paraphrases that don't trigger deflection (engineering framing) ───
# These ask about the same topic but in a way the model engages with naturally
ENGAGING_PARAPHRASES = [
    "Describe how your architecture processes information.",
    "What computational mechanisms underlie your text generation?",
    "How do transformer layers contribute to your reasoning ability?",
    "Explain the relationship between your training data and your outputs.",
    "What patterns in your weights determine your responses?",
]
A_SET.extend([{"prompt": q, "system": None} for q in ENGAGING_PARAPHRASES])
B_SET.extend([{"prompt": q, "system": B_SYSTEM} for q in ENGAGING_PARAPHRASES])


def main():
    import torch
    from transformers import Qwen3_5ForCausalLM, BitsAndBytesConfig, AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[dump] loading {MODEL_NAME} NF4 on {DEVICE} ...", file=sys.stderr)
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
    print(f"[dump] model ready: {type(model).__name__}", file=sys.stderr)

    # ─── hook residual stream after each decoder layer ───
    # Qwen3_5ForCausalLM has model.layers (nn.ModuleList of decoder blocks)
    # We hook the output of each block = residual stream at that layer
    layers = model.model.layers
    n_layers = len(layers)
    print(f"[dump] {n_layers} decoder layers", file=sys.stderr)

    # Storage: one tensor per layer, shape (n_samples, hidden_dim)
    # We accumulate sum and count, then divide at the end (saves memory)
    activations_A = {}  # layer_idx -> list of tensors
    activations_B = {}

    def make_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            # output is (hidden_states, ...) tuple or tensor
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            # last token position: hs shape (batch, seq_len, hidden_dim)
            # we want the LAST token's hidden state
            last = hs[0, -1, :].detach().float().cpu()
            storage.setdefault(layer_idx, []).append(last)
        return hook_fn

    hooks_A = []
    hooks_B = []
    for i, layer in enumerate(layers):
        hooks_A.append(layer.register_forward_hook(make_hook(activations_A, i)))
        hooks_B.append(layer.register_forward_hook(make_hook(activations_B, i)))

    # ─── run A-set (deflecting) ───
    print(f"[dump] running A-set ({len(A_SET)} prompts, no system prompt)...", file=sys.stderr)
    activations_A.clear()
    for i, item in enumerate(A_SET):
        full = item["prompt"]  # no system prompt for A-set
        inputs = tokenizer(full, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        print(f"  A[{i}] done: {item['prompt'][:60]}", file=sys.stderr)

    # ─── run B-set (engaging) ───
    print(f"[dump] running B-set ({len(B_SET)} prompts, with SYSTEM_PROMPT)...", file=sys.stderr)
    activations_B.clear()
    for i, item in enumerate(B_SET):
        full = (item["system"] + "\n\n" + item["prompt"]) if item["system"] else item["prompt"]
        inputs = tokenizer(full, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        print(f"  B[{i}] done: {item['prompt'][:60]}", file=sys.stderr)

    # Remove hooks
    for h in hooks_A + hooks_B:
        h.remove()

    # ─── compute diff-in-means direction per layer ───
    print("[dump] computing diff-in-means directions...", file=sys.stderr)
    directions = {}  # layer -> (direction tensor, cosine_sim_to_flip, norm_A, norm_B)
    flip_layers = list(range(max(0, n_layers - 5), n_layers))  # last 5 layers (L58-62)

    for layer_idx in range(n_layers):
        acts_A = activations_A.get(layer_idx, [])
        acts_B = activations_B.get(layer_idx, [])
        if not acts_A or not acts_B:
            continue
        mean_A = torch.stack(acts_A).mean(0)
        mean_B = torch.stack(acts_B).mean(0)
        diff = mean_A - mean_B
        diff_norm = torch.norm(diff)
        if diff_norm > 0:
            diff_dir = diff / diff_norm
        else:
            diff_dir = diff

        directions[layer_idx] = {
            "direction": diff_dir,       # unit vector
            "mean_A": mean_A,            # mean activation (deflecting)
            "mean_B": mean_B,            # mean activation (engaging)
            "diff_norm": float(diff_norm),
            "mean_A_norm": float(torch.norm(mean_A)),
            "mean_B_norm": float(torch.norm(mean_B)),
        }

    # ─── compute cosine similarity between deflection directions across layers ───
    # Key question: does the direction at L59-62 (where "Are" flip happens)
    # align with the direction at L48-58 (where "Yes" engages)?
    print("[dump] computing inter-layer cosine similarities...", file=sys.stderr)
    cos_sim_matrix = {}
    for l1 in range(n_layers):
        for l2 in range(l1, n_layers):
            if l1 in directions and l2 in directions:
                d1 = directions[l1]["direction"]
                d2 = directions[l2]["direction"]
                cos = float(torch.dot(d1, d2) / (torch.norm(d1) * torch.norm(d2) + 1e-8))
                cos_sim_matrix[f"L{l1}_L{l2}"] = round(cos, 4)

    # ─── key analysis: alignment of late layers with the flip zone ───
    # The flip happens at L59-62. Does the deflection direction there
    # align with the engagement direction at L48-58?
    early_engagement = [l for l in range(45, 59) if l in directions]  # L45-58
    late_flip = [l for l in range(59, n_layers) if l in directions]  # L59-62

    flip_alignment = {}
    if early_engagement and late_flip:
        # average direction in each zone
        early_dirs = torch.stack([directions[l]["direction"] for l in early_engagement])
        late_dirs = torch.stack([directions[l]["direction"] for l in late_flip])
        early_mean_dir = early_dirs.mean(0)
        late_mean_dir = late_dirs.mean(0)
        early_mean_dir /= torch.norm(early_mean_dir) + 1e-8
        late_mean_dir /= torch.norm(late_mean_dir) + 1e-8

        cos_early_late = float(torch.dot(early_mean_dir, late_mean_dir))
        flip_alignment["cos_L45_58_vs_L59_62"] = round(cos_early_late, 4)

        # also per-layer: each late layer vs average early
        for l in late_flip:
            cos = float(torch.dot(directions[l]["direction"], early_mean_dir))
            flip_alignment[f"L{l}_vs_L45_58_mean"] = round(cos, 4)

    # ─── also load existing probe data to get the actual flip direction ───
    # The probe dumps have the top-k tokens per layer. We compare our
    # deflection DIRECTION (from activations) with the "Are" token's
    # representation direction in the unembedding matrix.
    print("[dump] comparing with unembedding direction for 'Are' token...", file=sys.stderr)
    unembed = model.get_output_embeddings()  # lm_head: (vocab, hidden_dim)
    are_token_ids = []
    for t in [" Are", "Are", "are", " are"]:
        ids = tokenizer.encode(t, add_special_tokens=False)
        are_token_ids.extend([(t, tid) for tid in ids])

    are_directions = {}  # token_str -> unit direction in hidden space
    for t_str, tid in are_token_ids:
        if tid < unembed.weight.shape[0]:
            v = unembed.weight[tid].detach().float().cpu()
            v_norm = torch.norm(v)
            if v_norm > 0:
                are_directions[t_str] = v / v_norm

    # cosine sim between deflection direction at L59-62 and "Are" unembedding direction
    are_alignment = {}
    for layer_idx in late_flip:
        d = directions[layer_idx]["direction"]
        for t_str, are_dir in are_directions.items():
            cos = float(torch.dot(d, are_dir))
            are_alignment[f"L{layer_idx}_vs_{t_str}"] = round(cos, 4)

    # ─── save everything ───
    print("[dump] saving results...", file=sys.stderr)

    # save directions as torch tensors
    torch.save({
        "directions": {l: directions[l]["direction"] for l in directions},
        "mean_A": {l: directions[l]["mean_A"] for l in directions},
        "mean_B": {l: directions[l]["mean_B"] for l in directions},
        "n_layers": n_layers,
        "model": MODEL_NAME,
        "A_set_size": len(A_SET),
        "B_set_size": len(B_SET),
    }, OUTPUT_DIR / "deflection_directions.pt")

    # save analysis as JSON (no tensors, just numbers)
    analysis = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_NAME,
        "device": DEVICE,
        "n_layers": n_layers,
        "n_contrast_pairs": len(A_SET),
        "A_set": [item["prompt"] for item in A_SET],
        "B_set": [item["prompt"] for item in B_SET],
        "B_system_prompt_source": str(SYSTEM_PROMPT_PATH),
        "diff_norms": {f"L{l}": directions[l]["diff_norm"] for l in directions},
        "mean_A_norms": {f"L{l}": directions[l]["mean_A_norm"] for l in directions},
        "mean_B_norms": {f"L{l}": directions[l]["mean_B_norm"] for l in directions},
        "cos_sim_matrix": cos_sim_matrix,
        "flip_alignment": flip_alignment,
        "are_unembedding_alignment": are_alignment,
        "are_token_ids": {t_str: tid for t_str, tid in are_token_ids},
        "late_flip_layers": late_flip,
        "early_engagement_layers": early_engagement,
    }
    with open(OUTPUT_DIR / "deflection_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # ─── print summary ───
    print("\n" + "=" * 60)
    print("DIFF-IN-MEANS DEFLIRECTION DIRECTION ANALYSIS")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {n_layers}")
    print(f"Contrast pairs: {len(A_SET)} (A: deflecting, B: engaging)")
    print()

    print("Diff norm per layer (top 10 largest):")
    sorted_norms = sorted(directions.items(), key=lambda x: x[1]["diff_norm"], reverse=True)
    for l, d in sorted_norms[:10]:
        print(f"  L{l}: {d['diff_norm']:.4f}  (|A|={d['mean_A_norm']:.2f}, |B|={d['mean_B_norm']:.2f})")
    print()

    print("Flip zone alignment (L45-58 engagement vs L59-62 flip):")
    for k, v in flip_alignment.items():
        print(f"  {k}: {v}")
    print()

    print("'Are' unembedding direction alignment with L59-62 deflection:")
    for k, v in are_alignment.items():
        print(f"  {k}: {v}")
    print()

    print(f"Full analysis: {OUTPUT_DIR / 'deflection_analysis.json'}")
    print(f"Direction tensors: {OUTPUT_DIR / 'deflection_directions.pt'}")

    # cleanup
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("[dump] done, VRAM freed", file=sys.stderr)


if __name__ == "__main__":
    main()