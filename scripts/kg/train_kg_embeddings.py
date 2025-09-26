#!/usr/bin/env python3
"""
Train knowledge graph embeddings on data/osrs_kg_triples.csv using PyKEEN
and export a lightweight artifact for the API to consume.

Outputs (in --out directory, default: data/kg_model):
- entity_embeddings.npy          (float32, shape [num_entities, dim])
- entity_to_id.json              (mapping: entity label -> integer id)
- entities.txt                   (one label per line in id order)
- meta.json                      (model info)
- pykeen/                        (full PyKEEN artifacts)

Usage:
  python scripts/kg/train_kg_embeddings.py \
    --triples data/osrs_kg_triples.csv \
    --out data/kg_model --model TransE --dimension 100 \
    --epochs 25 --num-workers 4 --batch-size 256
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import threading


def parse_args():
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument("--triples", default=str(repo_root / "data" / "osrs_kg_triples.csv"), help="CSV with columns head,relation,tail (first 3 columns are used)")
    ap.add_argument("--out", default=str(repo_root / "data" / "kg_model"), help="Output directory for artifacts")
    ap.add_argument("--model", default="TransE", choices=["TransE", "DistMult", "ComplEx", "RotatE"], help="PyKEEN model name")
    ap.add_argument("--dimension", type=int, default=100, choices=[50, 100, 200, 300], help="Embedding dimension")
    ap.add_argument("--epochs", type=int, default=25, help="Training epochs (PyKEEN only)")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size (PyKEEN only)")
    ap.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (0=auto)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for optimizer")
    ap.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD", "Adagrad"], help="Optimizer type")
    ap.add_argument("--backend", choices=["auto", "pykeen", "spectral"], default="auto", help="Training backend: pykeen or spectral (node embedding)")
    ap.add_argument("--no-eval", action="store_true", help="Skip PyKEEN evaluation to avoid large memory usage; train only")
    ap.add_argument("--strict", action="store_true", help="Do not fallback to spectral; if PyKEEN fails, exit with error")
    ap.add_argument("--resume", action="store_true", help="Resume from last completed phase if intermediates exist")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write a PID file for GUI Stop button and set seeds for reproducibility
    import atexit, signal, random, warnings
    repo_root = Path(__file__).resolve().parents[2]
    kg_logs_dir = repo_root / "OSRS_AI_SYSTEM" / "logs" / "kg"
    try:
        kg_logs_dir.mkdir(parents=True, exist_ok=True)
        pid_path = kg_logs_dir / "train.pid"
        pid_path.write_text(str(os.getpid()))
        def _cleanup_pid(*_):
            try:
                pid_path.unlink(missing_ok=True)
            except Exception:
                pass
            # Exit fast on signal
            if _:
                sys.exit(0)
        atexit.register(_cleanup_pid)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _cleanup_pid)
            except Exception:
                pass
    except Exception:
        pass

    # Set deterministic seeds across libs
    try:
        random.seed(int(args.seed))
    except Exception:
        pass
    try:
        np.random.seed(int(args.seed))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        # Also tell PyKEEN
        try:
            from pykeen.utils import set_random_seed as _set_rs
            _set_rs(int(args.seed))
        except Exception:
            pass
    except Exception:
        pass

    # Silence known noisy warnings in GUI output
    try:
        from urllib3.exceptions import NotOpenSSLWarning
        warnings.simplefilter("ignore", NotOpenSSLWarning)
    except Exception:
        pass
    try:
        warnings.filterwarnings("ignore", message=r".*Length of IterableDataset.*")
    except Exception:
        pass

    # Resumable artifact paths
    map_path = out_dir / "entity_to_id.json"
    adj_path = out_dir / "adjacency.npz"
    emb_path = out_dir / "entity_embeddings.npy"
    progress_path = out_dir / "progress.json"

    def write_progress(phase: str, info: dict):
        rec = {"phase": phase, "ts": int(time.time())}
        if info:
            rec.update(info)
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(rec, f)
        except Exception:
            pass

    # Load triples (use only first 3 columns) with coarse progress
    import csv
    heads, rels, tails = [], [], []
    total = 0
    print("[KG] Reading triples...", flush=True)
    with open(args.triples, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for i, row in enumerate(reader, start=1):
            if not row or len(row) < 3:
                continue
            h, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            if not h or not r or not t:
                continue
            heads.append(h)
            rels.append(r)
            tails.append(t)
            total += 1
            if i % 50000 == 0:
                print(f"[KG]  triples read: {i}", flush=True)
    if not heads:
        print("[KG] No triples found; ensure osrs_kg_triples.csv exists and is non-empty", flush=True)
        raise SystemExit(1)
    print(f"[KG] Total triples loaded: {total}", flush=True)

    used_backend = None
    emb = None

    # Phase 1: entity mapping (resume-aware)
    if args.resume and map_path.exists():
        print("[KG] Resume: loading entity_to_id.json", flush=True)
        with open(map_path, "r", encoding="utf-8") as f:
            entity_to_id = json.load(f)
        # Incremental add: append new entities without changing existing IDs
        added = 0
        for i, h in enumerate(heads):
            if h not in entity_to_id:
                entity_to_id[h] = len(entity_to_id); added += 1
            t = tails[i]
            if t not in entity_to_id:
                entity_to_id[t] = len(entity_to_id); added += 1
            if (i+1) % 200000 == 0:
                print(f"[KG]  mapping scan (resume) progress: {i+1}/{total}", flush=True)
        if added:
            print(f"[KG] Resume: added {added} new entities (total now {len(entity_to_id)})", flush=True)
        # Always rewrite mapping so downstream has the union
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(entity_to_id, f, ensure_ascii=False, indent=2)
    else:
        print("[KG] Building entity mapping...", flush=True)
        entity_to_id = {}
        for i, h in enumerate(heads):
            if h not in entity_to_id:
                entity_to_id[h] = len(entity_to_id)
            t = tails[i]
            if t not in entity_to_id:
                entity_to_id[t] = len(entity_to_id)
            if (i+1) % 100000 == 0:
                print(f"[KG]  mapping progress: {i+1}/{total}", flush=True)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(entity_to_id, f, ensure_ascii=False, indent=2)
    write_progress("mapping", {"nodes": len(entity_to_id)})

    # Write entities.txt (keeps order by id)
    entities_txt = out_dir / "entities.txt"
    inv = sorted([(i, e) for e, i in entity_to_id.items()], key=lambda x: x[0])
    with open(entities_txt, "w", encoding="utf-8") as f:
        for _i, e in inv:
            f.write(e + "\n")

    # Phase 2: adjacency construction (resume-aware)
    try:
        import scipy.sparse as sp
    except Exception as e:
        print(f"[KG] ERROR: scipy is required: {e}", file=sys.stderr)
        raise

    need_rebuild_adj = True
    if adj_path.exists():
        try:
            A = sp.load_npz(adj_path)
            need_rebuild_adj = (A.shape[0] != len(entity_to_id))
            # Rebuild if triples newer than adjacency file
            try:
                if os.path.getmtime(args.triples) > os.path.getmtime(adj_path):
                    need_rebuild_adj = True
            except Exception:
                pass
        except Exception:
            A = None
            need_rebuild_adj = True

    if need_rebuild_adj:
        print("[KG] Building adjacency (undirected, symmetric)...", flush=True)
        rows = []; cols = []; data = []
        for i, (h, t) in enumerate(zip(heads, tails), start=1):
            ih = entity_to_id.get(h); it = entity_to_id.get(t)
            if ih is None or it is None:
                continue
            rows.extend([ih, it]); cols.extend([it, ih]); data.extend([1.0, 1.0])
            if i % 200000 == 0:
                print(f"[KG]  adjacency edges processed: {i}/{total}", flush=True)
        N = len(entity_to_id)
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        from scipy.sparse import save_npz
        save_npz(adj_path, A)
    else:
        print("[KG] Reusing existing adjacency.npz (up-to-date)", flush=True)
    write_progress("adjacency", {"nodes": int(A.shape[0]), "edges": int(A.nnz//2)})

    # Phase 3: embeddings
    if args.backend in ("auto", "pykeen"):
        try:
            print("[KG] Training with PyKEEN...", flush=True)
            from pykeen.triples import TriplesFactory
            from pykeen.pipeline import pipeline
            # device selection
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            except Exception:
                device = 'cpu'
            print(f"[KG] Device: {device}", flush=True)

            # checkpoint settings and monitor thread
            ckpt_dir = out_dir / "pykeen_ckpt"
            # If not resuming, clear any stale checkpoints to avoid incompatible torch.load
            if not args.resume and ckpt_dir.exists():
                import shutil
                try:
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                except Exception:
                    pass
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_name = 'pykeen_checkpoint.pt'
            stop_flag = False
            # resolve workers dynamically if 0
            try:
                # Use as many logical cores as available (cap at 32 for safety)
                cores = os.cpu_count() or 4
                auto_workers = max(2, min(32, int(cores)))
            except Exception:
                auto_workers = 4
            workers = int(args.num_workers) if int(args.num_workers) > 0 else auto_workers
            print(f"[KG] DataLoader workers: {workers} (auto={args.num_workers==0})", flush=True)

            def monitor_ckpt():
                import time as _t
                import json as _j
                last_size = -1
                while not stop_flag:
                    p = ckpt_dir / ckpt_name
                    info = {"phase": "embedding", "ts": int(time.time()), "backend": "pykeen", "epochs_target": int(args.epochs)}
                    if p.exists():
                        try:
                            sz = p.stat().st_size
                            info["checkpoint_bytes"] = int(sz)
                            info["checkpoint_mtime"] = int(p.stat().st_mtime)
                            # Try to read current epoch from checkpoint (best-effort)
                            try:
                                import torch as _torch
                                ck = _torch.load(str(p), map_location="cpu")
                                cur_ep = None
                                def _find_epoch(obj):
                                    if isinstance(obj, dict):
                                        for k, v in obj.items():
                                            if k.lower() == 'epoch' and isinstance(v, (int, float)):
                                                return int(v)
                                        for v in obj.values():
                                            r = _find_epoch(v)
                                            if r is not None:
                                                return r
                                    elif isinstance(obj, (list, tuple)):
                                        for v in obj:
                                            r = _find_epoch(v)
                                            if r is not None:
                                                return r
                                    return None
                                cur_ep = _find_epoch(ck)
                                if isinstance(cur_ep, int) and 0 <= cur_ep <= int(args.epochs):
                                    info["epoch_current"] = cur_ep
                            except Exception:
                                pass
                        except Exception:
                            pass
                    try:
                        with open(progress_path, "w", encoding="utf-8") as f:
                            _j.dump(info, f)
                    except Exception:
                        pass
                    _t.sleep(5)

            t = threading.Thread(target=monitor_ckpt, daemon=True)
            t.start()

            # Build TriplesFactory
            triples = np.stack([
                np.array(heads, dtype=object),
                np.array(rels, dtype=object),
                np.array(tails, dtype=object),
            ], axis=1)
            tf = TriplesFactory.from_labeled_triples(
                triples,
                create_inverse_triples=False,
                entity_to_id=entity_to_id,
            )
            # Save relation mapping for eval compatibility
            try:
                rel_map_path = out_dir / "relation_to_id.json"
                with open(rel_map_path, "w", encoding="utf-8") as f:
                    json.dump(tf.relation_to_id, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            # ensure we provide training/testing/validation factories for PyKEEN>=1.10
            if args.no_eval:
                # Train via TrainingLoop directly to avoid any evaluation phase entirely
                from pykeen.models import TransE, DistMult, ComplEx, RotatE
                from pykeen.training import SLCWATrainingLoop
                from pykeen.optimizers import get_optimizer_cls

                train_tf = tf

                # Create model based on args.model
                model_cls = {
                    "transe": TransE,
                    "distmult": DistMult,
                    "complex": ComplEx,
                    "rotate": RotatE
                }[args.model.lower()]

                model = model_cls(triples_factory=train_tf, embedding_dim=int(args.dimension))
                model = model.to(device)

                # Create optimizer with custom learning rate
                optimizer_cls = get_optimizer_cls(args.optimizer)
                optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)

                loop = SLCWATrainingLoop(model=model, triples_factory=train_tf, optimizer=optimizer)
                # Save checkpoint every epoch via our monitor thread + manual save is handled by loop
                loop.train(triples_factory=train_tf, num_epochs=int(args.epochs), batch_size=int(args.batch_size), num_workers=int(workers), use_tqdm=True)
                stop_flag = True
                (out_dir / "pykeen").mkdir(exist_ok=True)
                # Save the trained model directory in a minimal way
                try:
                    from pykeen.pipeline import pipeline as _pp  # for consistent directory structure
                    # Create a tiny Result-like save directory
                    model.save_state(path=(out_dir / "pykeen" / "trained_model_state.pt"))
                except Exception:
                    pass
            else:
                train_tf, test_tf, val_tf = tf.split([0.9, 0.05, 0.05], random_state=int(args.seed))
                # Create optimizer kwargs for pipeline
                optimizer_kwargs = {
                    'lr': args.learning_rate
                }

                result = pipeline(
                    training=train_tf,
                    testing=test_tf,
                    validation=val_tf,
                    model=args.model,
                    model_kwargs={'embedding_dim': int(args.dimension)},
                    optimizer=args.optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    training_kwargs={
                        'num_epochs': int(args.epochs),
                        'batch_size': int(args.batch_size),
                        'checkpoint_name': ckpt_name,
                        'checkpoint_directory': str(ckpt_dir),
                        'checkpoint_frequency': 1,  # save every epoch
                        'checkpoint_on_failure': True,
                        'num_workers': int(workers),
                    },
                    random_seed=int(args.seed),
                    device=device,
                )
                stop_flag = True
                (out_dir / "pykeen").mkdir(exist_ok=True)
                result.save_to_directory(out_dir / "pykeen")
                model = result.model
            with np.errstate(all='ignore'):
                try:
                    emb = model.entity_representations[0]().detach().cpu().numpy().astype('float32')
                except Exception:
                    # fallback for older/newer APIs
                    emb = model.entity_representations[0](indices=None).detach().cpu().numpy().astype('float32')
            used_backend = "pykeen"
        except Exception as e:
            if args.strict or args.backend == "pykeen":
                print(f"[KG] PyKEEN failed (strict mode): {e}", file=sys.stderr)
                raise
            print(f"[KG] PyKEEN unavailable/incompatible: {e}. Falling back to spectral.", flush=True)

    if emb is None:
        print("[KG] Computing spectral embedding... (no fine-grained progress)", flush=True)
        from sklearn.manifold import SpectralEmbedding
        A = sp.load_npz(adj_path) if 'A' not in locals() or A is None else A
        k = min(int(args.dimension), max(2, A.shape[0] - 1))
        se = SpectralEmbedding(n_components=k, affinity='precomputed', random_state=int(args.seed))
        emb = se.fit_transform(A.astype(np.float32)).astype('float32')
        if emb.shape[1] < int(args.dimension):
            pad = np.zeros((emb.shape[0], int(args.dimension) - emb.shape[1]), dtype='float32')
            emb = np.concatenate([emb, pad], axis=1)
        used_backend = "spectral"
    # Save final artifacts incrementally
    np.save(out_dir / "entity_embeddings.npy", emb)

    meta = {
        'backend': used_backend,
        'model': args.model if used_backend == 'pykeen' else 'SpectralEmbedding',
        'dimension': int(args.dimension),
        'epochs': int(args.epochs) if used_backend == 'pykeen' else 0,
        'batch_size': int(args.batch_size) if used_backend == 'pykeen' else 0,
        'learning_rate': args.learning_rate if used_backend == 'pykeen' else 0,
        'optimizer': args.optimizer if used_backend == 'pykeen' else 'N/A',
        'num_workers': int(args.num_workers),
        'triples_count': int(total),
        'nodes_count': int(len(entity_to_id)),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Progress marker
    try:
        with open(out_dir / "progress.json", "w", encoding="utf-8") as f:
            json.dump({"phase": "done", "ts": int(time.time()), "ok": True}, f)
    except Exception:
        pass

    print(f"✅ Trained KG embeddings ({used_backend}) : {emb.shape} saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

