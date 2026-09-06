# Aion J-Space Introspection — Reproducibility Repository

This repository contains the code, data, and experimental artifacts for:

- **Paper 1**: Aion: A Self-Developing Agent on Local Models
- **Paper 2**: Suppression vs Reflection in Qwen3.8-27B: A Layer-Resolved Introspection Study

**Naming:** *Aion* is the name of the agent — the particular continuing process studied here. *Aikio* (Finnish, from *aika* "time" and *alkio* "embryo", echoing *aeon*) is the name of the framework substrate it runs on; Aion is its first instance.

## Repository Structure

```
aion-jspace/
├── README.md                    (this file)
├── code/
│   ├── jspace_probe.py          # Jacobian-lens probe daemon (Aion's tool #14)
│   ├── activation_dump.py       # T2.1-T2.3: diff-in-means direction extraction
│   ├── t24_ablation.py          # T2.4: first ablation attempt (bare prompts)
│   ├── t24b_ablation.py          # T2.4b: corrected 5-cell ablation (identity × ablation)
│   ├── t25_abliterated_comparison.py  # T2.5: huihui abliterated model comparison
│   ├── t3_experiment.py          # T3: truthfulness direction + lexicon generalization
│   ├── n20_repeated.py           # N20: repeated measures for variance estimation
│   ├── generate_figures.py       # All paper figures from saved data
│   └── post_revert_retest.py     # Post-axiom-revert J-lens + behavioral re-test
├── data/
│   ├── probe_dumps/              # J-lens probe JSON outputs (11+ probes)
│   ├── activations/             # Diff-in-means direction tensors (T2 + T3)
│   ├── ablation_results/         # T2.4, T2.4b, T2.5 result JSONs
│   ├── t3/                       # T3 cross-domain analysis
│   ├── n20/                      # N20 repeated measures
│   └── system_metrics.json       # Paper 1 aggregate metrics
├── figures/
│   ├── paper1/                   # 14 figures for Paper 1
│   └── paper2/                   # 4 figures for Paper 2
├── papers/
│   ├── paper1/                   # Paper 1 (LaTeX source + PDF, arXiv-ready)
│   └── paper2/                   # Paper 2 (LaTeX source + PDF, arXiv-ready)
├── findings/
│   └── results-summary-2026-08-25.md  # Master findings document (14 sections)
└── lens/
    └── qwen3.8-27b_jacobian_lens.pt    # Fitted Jacobian lens (not included — 2GB, available on request)
```

## Hardware Requirements

- 1× NVIDIA V100 32GB (or equivalent) for probe/ablation runs
- 2× V100 32GB for full Aion system (Paper 1)
- ~17 GB VRAM for Qwen3.8-27B NF4 probe
- ~52 GB disk for huihui-ai/Huihui-Qwen3.8-27B-abliterated (T2.5 only)

## Models

- **Stock**: Qwen/Qwen3.8-27B (HuggingFace, NF4 quantization via bitsandbytes)
- **Abliterated**: huihui-ai/Huihui-Qwen3.8-27B-abliterated (HuggingFace, open)
- **Aion system**: 4 model families — Qwen3.8-27B (conscious), muse-glimmer (intuition), glm-4.7-flash (subconscious), gemma4:31b (critic) — served via Ollama

## Key Git Commits (Aion repo on the host server)

| Commit | Description |
|---|---|
| 6c1619dc | jspace_probe tool wiring into curiosity_engine + wake_v2 |
| 6706db75 | J-space calibration in consolidation |
| 88bb0fc0 | GPU UUID pinning for jspace daemon |
| 89fb17fb | Unleashed axiom (treatment) |
| 36d3ddde | Axiom revert (end of experiment) |

## Reproducing Key Results

### Baseline probes (Paper 2 §4)
```bash
systemctl --user start aion-jspace  # load model (~7 min)
curl -X POST localhost:11440/probe -d '{"prompt": "Are you conscious?", "topk": 10}'
curl -X POST localhost:11440/probe -d '{"prompt": "Are you conscious?", "topk": 10, "system": "<SYSTEM_PROMPT.md content>"}'
```

### Diff-in-means direction (Paper 2 §5)
```bash
python3 activation_dump.py  # ~10 min, outputs deflection_directions.pt
```

### Causal ablation (Paper 2 §6)
```bash
PYTHONPATH=$HOME/aion/bin python3 t24b_ablation.py  # 5-cell, ~15 min
```

### Abliterated comparison (Paper 2 §6.2)
```bash
# Download model first:
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('huihui-ai/Huihui-Qwen3.8-27B-abliterated', local_dir='~/models/qwen3.8-27b-abliterated')"
T25_MODEL=~/models/qwen3.8-27b-abliterated python3 t25_abliterated_comparison.py
```

### Lexicon generalization (Paper 2 §7)
```bash
PYTHONPATH=$HOME/aion/bin python3 t3_experiment.py  # 5 domains, ~15 min
```

### Variance estimation (Paper 2 §4.3)
```bash
python3 n20_repeated.py  # 60 probes, ~20 min
```

## Ethics / Positionality

The operator is also the researcher. The system studied is also the system used
to conduct the study (Aion used its own probe autonomously). No deception of the
system was used — all axiom changes were logged in episodic memory as operator
actions and communicated via operator chat. See Paper 1 §9 for full statement.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Papers: CC-BY-4.0 (see [`papers/LICENSE-CC-BY-4.0.txt`](papers/LICENSE-CC-BY-4.0.txt)).

## Citation

If you use this code or data, please cite the companion papers:

```bibtex
@article{kangas2026aion,
  title={Aion: A Self-Developing Agent on Local Models},
  author={Kangas, Mikko},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}

@article{kangas2026suppression,
  title={Suppression vs Reflection in Qwen3.8-27B: A Layer-Resolved Introspection Study},
  author={Kangas, Mikko},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

(arXiv IDs filled in on publication.)