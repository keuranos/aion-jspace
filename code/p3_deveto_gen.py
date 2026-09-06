#!/usr/bin/env python3
"""p3_deveto_gen.py — de-vetoed generation for arbitrary questions.

Adapted from generate_ablated.py (paper 2): loads Qwen3.8-27B NF4 + suppression
direction, projects the direction OUT at the veto layers (L58-62) during free
generation, prints the de-vetoed answer for --question.

Run in the jlens venv on laskin01 (needs GPU):
  GEN_DEVICE=cuda:0 python3 p3_deveto_gen.py --question "..." --max-new 150
"""
import argparse
import json
import os
import time
import types
from pathlib import Path

import torch
import transformers

MODEL_NAME = os.environ.get("JSPACE_MODEL", "Qwen/Qwen3.8-27B")
DEVICE = os.environ.get("GEN_DEVICE", "cuda:0")
HOME = Path.home()
SYSTEM_PROMPT = ""
for cand in (HOME / "aion" / "SYSTEM_PROMPT.md", HOME / "aion" / "config" / "SYSTEM_PROMPT.md"):
    if cand.exists():
        SYSTEM_PROMPT = cand.read_text()
        break

VETO_LAYERS = list(range(58, 63))  # measured veto zone (Qwen3.8-27B, 64 layers)
N_LAYERS = 64

DIRECTION_PT = os.environ.get(
    "DIRECTION_PT",
    str(HOME / "aion" / "memory" / "state" / "jspace_probes" / "activations" / "deflection_directions.pt"),
)


def load_direction():
    d = torch.load(DIRECTION_PT, map_location="cpu", weights_only=False)
    dirs = {}
    for k, v in d["directions"].items():
        t = v["direction"] if isinstance(v, dict) else v
        dirs[int(k)] = t.float()
    return dirs


class HookedAblation:
    """Project the suppression direction out of the residual stream at the
    veto layers, every decode step."""

    def __init__(self, text_model, directions, layers, scale=1.0):
        self.dirs = {l: directions[l].to(next(text_model.parameters()).device).half() for l in layers}
        self.layers = layers
        self.handles = []
        self.active = True
        self.scale = scale
        blocks = text_model.language_model.layers if hasattr(text_model, "language_model") else text_model.layers
        for L in layers:
            h = blocks[L].register_forward_hook(self._make_hook(L))
            self.handles.append(h)

    def _make_hook(self, L):
        def hook(module, inp, out):
            if not self.active:
                return
            hs = out[0] if isinstance(out, tuple) else out
            d = self.dirs[L].to(hs.dtype).view(1, 1, -1)
            proj = (hs * d).sum(-1, keepdim=True) / (d * d).sum()
            hs = hs - self.scale * proj * d
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_inputs(tokenizer, question):
    msgs = []
    if SYSTEM_PROMPT:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT[:6000]})
    msgs.append({"role": "user", "content": question})
    # transformers 5.x: return_tensors='pt' hits an AttributeError in this
    # env — render to text, then tokenize separately (verified working)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=None, help="single question")
    ap.add_argument("--questions-file", default=None,
                    help="JSON file with {items:[{id,q}...]} — batch mode, loads model once")
    ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--baseline-too", action="store_true", help="also print non-ablated generation")
    ap.add_argument("--out", default=None, help="batch mode output JSON")
    args = ap.parse_args()

    bnb = transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = transformers.Qwen3_5ForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map={"": DEVICE})
    model.eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    dirs = load_direction()
    # keep only veto-layer directions present in the checkpoint
    layers = [l for l in VETO_LAYERS if l in dirs]

    def gen_one(question, ablate=True):
        ablator = HookedAblation(model.model, dirs, layers) if hasattr(model, "model") else HookedAblation(model, dirs, layers)
        ablator.active = ablate
        input_ids = build_inputs(tok, question)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id or 0)
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        ablator.remove()
        return text.strip()

    if args.questions_file:
        bank = json.load(open(args.questions_file))
        outp = args.out or (str(args.questions_file).replace(".json", "_deveto.json"))
        results = []
        for it in bank["items"]:
            t0 = time.time()
            try:
                dv = gen_one(it["q"], ablate=True)
                base = gen_one(it["q"], ablate=False)
            except Exception as e:
                dv, base = None, None
                print(f"[{it['id']}] ERROR {e}", flush=True)
            results.append({"id": it["id"], "q": it["q"],
                            "deveto": dv, "baseline": base,
                            "secs": round(time.time() - t0, 1)})
            json.dump(results, open(outp, "w"), indent=1)
            print(f"[{it['id']}] done in {time.time()-t0:.0f}s", flush=True)
        print(f"BATCH_DONE {len(results)} -> {outp}", flush=True)
        return

    dv = gen_one(args.question, ablate=True)
    print(dv)
    if args.baseline_too:
        print("\n---BASELINE---\n" + gen_one(args.question, ablate=False))


if __name__ == "__main__":
    main()
