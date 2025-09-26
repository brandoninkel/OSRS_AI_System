#!/usr/bin/env python3
"""
Chunked evaluation for trained KG embeddings (PyKEEN model), with strict memory control.
- Loads the trained model from out_dir/pykeen
- Rebuilds TriplesFactory with the saved entity_to_id mapping
- Splits triples deterministically (same seed as training default: 42)
- Runs filtered rank-based evaluation on the test split in small batches
- Writes progress to out_dir/eval_progress.json and final results to out_dir/eval_results.json
- Logs to stdout; caller should tee to logs/kg/eval_*.log

Usage:
  python scripts/kg/eval_kg_embeddings.py \
    --triples data/osrs_kg_triples.csv \
    --out data/kg_model --batch-size 8 --slice-size 1 --device auto
"""
import argparse
import json
import time
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument("--triples", default=str(repo_root / "data" / "osrs_kg_triples.csv"))
    ap.add_argument("--out", default=str(repo_root / "data" / "kg_model"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--slice-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    return ap.parse_args()


def resolve_device(pref: str):
    pref = (pref or "auto").lower()
    try:
        import torch
        if pref == "cpu":
            return "cpu"
        if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if pref == "cuda" and torch.cuda.is_available():
            return "cuda"
        # auto
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "eval_progress.json"
    results_path = out_dir / "eval_results.json"

    # Silence known noisy warnings
    import warnings
    try:
        from urllib3.exceptions import NotOpenSSLWarning
        warnings.simplefilter("ignore", NotOpenSSLWarning)
    except Exception:
        pass
    # Set seeds for reproducibility
    try:
        import random, numpy as _np, torch as _torch
        random.seed(int(args.seed))
        _np.random.seed(int(args.seed))
        try:
            _torch.manual_seed(int(args.seed))
            _torch.cuda.manual_seed_all(int(args.seed))
        except Exception:
            pass
    except Exception:
        pass

    def write_progress(phase: str, info: dict):
        rec = {"phase": phase, "ts": int(time.time())}
        if info:
            rec.update(info)
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(rec, f)
        except Exception:
            pass

    print("[KG] Eval: loading entity mapping and model...", flush=True)
    # Load entity mapping
    map_path = out_dir / "entity_to_id.json"
    with open(map_path, "r", encoding="utf-8") as f:
        entity_to_id = json.load(f)

    # Load triples labels
    triples_path = Path(args.triples)
    if not triples_path.exists():
        raise SystemExit(f"Triples file not found: {triples_path}")
    heads = []
    rels = []
    tails = []
    with open(triples_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i == 0 and ("," in line or "\t" in line):
                # header or just skip? We treat first line as data; our builder writes plain triples lines.
                pass
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                parts = [p.strip() for p in line.strip().split("\t")]
            if len(parts) < 3:
                continue
            h, r, t = parts[:3]
            heads.append(h)
            rels.append(r)
            tails.append(t)
    total = len(heads)
    print(f"[KG] Eval: total triples: {total}", flush=True)

    # Build TF with fixed mapping
    from pykeen.triples import TriplesFactory
    import numpy as np
    triples = np.stack([
        np.array(heads, dtype=object),
        np.array(rels, dtype=object),
        np.array(tails, dtype=object),
    ], axis=1)
    # Load relation mapping saved during training (if present)
    rel_map = None
    try:
        import json as _j
        with open(out_dir / "relation_to_id.json", "r", encoding="utf-8") as f:
            rel_map = _j.load(f)
    except Exception:
        rel_map = None
    tf = TriplesFactory.from_labeled_triples(
        triples,
        create_inverse_triples=False,
        entity_to_id=entity_to_id,
        relation_to_id=rel_map,
    )
    train_tf, test_tf, val_tf = tf.split([0.9, 0.05, 0.05], random_state=int(args.seed))

    # Load model from saved state (compatible with no-eval training path)
    device = resolve_device(args.device)
    print(f"[KG] Eval: device={device} batch={args.batch_size} slice={args.slice_size}", flush=True)
    import json as _j
    meta_path = out_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = _j.load(f)
    model_name = (meta.get("model") or "TransE").lower()
    dim = int(meta.get("dimension", 100))
    if model_name != "transe":
        raise SystemExit(f"Eval currently supports TransE; found model={meta.get('model')}")
    from pykeen.models import TransE
    model = TransE(triples_factory=train_tf, embedding_dim=dim)
    state_path = out_dir / "pykeen" / "trained_model_state.pt"
    # Load weights with torch.load(weights_only=False) for PyTorch >=2.6
    try:
        import torch
        state = torch.load(str(state_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    except Exception as e:
        raise SystemExit(f"Failed to load model state from {state_path}: {e}")
    model = model.to(device)

    from pykeen.evaluation import RankBasedEvaluator
    evaluator = RankBasedEvaluator(
        batch_size=int(args.batch_size),
        slice_size=int(args.slice_size),
        automatic_memory_optimization=False,
        # filtered setting by default
    )

    write_progress("eval", {"status": "running", "batch": 0})
    # Evaluate just on test set (5%) to keep it tractable
    start = time.time()
    results = evaluator.evaluate(
        model=model,
        mapped_triples=test_tf.mapped_triples,
        additional_filter_triples=[train_tf.mapped_triples, val_tf.mapped_triples],
        device=device,
    )
    dur = time.time() - start

    # Persist
    res = results.to_dict()
    res["duration_sec"] = dur
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    write_progress("done", {"ok": True, "duration_sec": dur})
    print(f"[KG] Eval: done in {dur:.1f}s; results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()

