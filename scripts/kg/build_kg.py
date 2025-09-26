#!/usr/bin/env python3
"""
Build a lightweight OSRS Knowledge Graph from the existing wiki JSONL files.
- INPUTS (read-only):
  * data/osrs_wiki_content.jsonl        (cleaned text, categories, revid)
  * data/osrs_wikitext_content.jsonl    (raw wikitext per revision; optional)
- OUTPUTS (new):
  * data/osrs_kg_triples.csv            (head, relation, tail, source_title, revid)
  * data/osrs_kg_nodes.jsonl            (per-node summary)
  * data/osrs_kg_edges.jsonl            (per-edge JSONL for debugging)
  * data/osrs_kg.meta.json              (run metadata)

Notes:
- Pure Python (no new deps). Uses regex to extract internal links and infobox fields from wikitext when available.
- Does NOT modify inputs. Optionally snapshots inputs to a temp copy to avoid mid-write interference.
- Designed to run alongside the existing embedding pipeline without changing it.

Usage:
  python3 OSRS_AI_SYSTEM/scripts/kg/build_kg.py --snapshot --max-pages 0
"""

import os
import re
import json
import csv
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../OSRS_AI_SYSTEM
DATA_DIR = REPO_ROOT / "data"
CONTENT_JSONL = DATA_DIR / "osrs_wiki_content.jsonl"
WIKITEXT_JSONL = DATA_DIR / "osrs_wikitext_content.jsonl"
OUT_TRIPLES_CSV = DATA_DIR / "osrs_kg_triples.csv"
OUT_NODES_JSONL = DATA_DIR / "osrs_kg_nodes.jsonl"
OUT_EDGES_JSONL = DATA_DIR / "osrs_kg_edges.jsonl"
OUT_META_JSON = DATA_DIR / "osrs_kg.meta.json"
TMP_DIR = DATA_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Regex for internal links: [[Target|text]] or [[Target]]; skip File:, Category:, etc.
LINK_RE = re.compile(r"\[\[([^\]|#:]+)(?:#[^\]]*)?(?:\|[^\]]*)?\]\]")
# Infobox capture (very lightweight): {{Infobox ... \n |key = value \n ... }}
INFOBOX_RE = re.compile(r"\{\{\s*Infobox[^\n}]*\n(?P<body>.*?)\n\}\}", re.DOTALL | re.IGNORECASE)
INFOBOX_KV_RE = re.compile(r"^\s*\|\s*([^=|]+?)\s*=\s*(.+)$")

SKIP_LINK_PREFIXES = (
    "File:", "Image:", "Category:", "Template:", "Wikipedia:", "RuneScape:", "MediaWiki:",
)


def load_jsonl(path: Path, max_lines: int = 0) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if max_lines and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def snapshot_file(src: Path) -> Path:
    ts = int(time.time())
    dst = TMP_DIR / f"{src.name}.snapshot-{ts}.jsonl"
    try:
        # cheap copy without loading whole file in memory
        with src.open("rb") as r, dst.open("wb") as w:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                w.write(chunk)
        return dst
    except Exception:
        # fall back to using the live file (read-only)
        return src


def normalize_title(t: str) -> str:
    t = (t or "").strip()
    # Remove HTML tags if any leaked
    t = re.sub(r"<[^>]*>", "", t)
    return t


def extract_links_from_wikitext(wikitext: str) -> List[str]:
    targets: List[str] = []
    if not wikitext:
        return targets
    for m in LINK_RE.finditer(wikitext):
        tgt = m.group(1).strip()
        if not tgt:
            continue
        if any(tgt.startswith(p) for p in SKIP_LINK_PREFIXES):
            continue
        # Normalize common underscore/space issues
        tgt = tgt.replace("_", " ")
        targets.append(tgt)
    return targets


def extract_infobox_kv(wikitext: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not wikitext:
        return out
    m = INFOBOX_RE.search(wikitext)
    if not m:
        return out
    body = m.group("body")
    for line in body.splitlines():
        km = INFOBOX_KV_RE.match(line)
        if not km:
            continue
        key = km.group(1).strip().lower().replace(" ", "_")
        val = km.group(2).strip()
        # strip wiki link markup inside value
        val = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", lambda mm: mm.group(2) or mm.group(1), val)
        # strip templates braces minimally
        val = re.sub(r"\{\{([^}]+)\}\}", r"\1", val)
        out[key] = val
    return out

# Top-level page processor (usable by multiprocessing workers)
def process_page(rec: Dict[str, Any], title_to_wt: Dict[str, str], w_csv, w_nodes, w_edges) -> int:
    title = normalize_title(rec.get("title", ""))
    if not title:
        return 0
    cats = rec.get("categories") or []
    revid = rec.get("revid") or rec.get("revisionId")
    clean_cats: List[str] = []
    for c in cats:
        name = None
        if isinstance(c, str):
            name = c
        elif isinstance(c, dict) and "category" in c:
            name = str(c["category"])
        if name is None:
            continue
        if name.startswith("Category:"):
            name = name[len("Category:"):]
        clean_cats.append(name)
    for cat in clean_cats:
        w_csv.writerow([title, "is_a", f"Category:{cat}", title, revid])
        w_edges.write(json.dumps({"head": title, "relation": "is_a", "tail": f"Category:{cat}", "source_title": title, "revid": revid}) + "\n")
    wikitext = title_to_wt.get(title, "")
    for tgt in extract_links_from_wikitext(wikitext):
        tgt = normalize_title(tgt)
        if not tgt:
            continue
        w_csv.writerow([title, "links_to", tgt, title, revid])
        w_edges.write(json.dumps({"head": title, "relation": "links_to", "tail": tgt, "source_title": title, "revid": revid}) + "\n")
    info = extract_infobox_kv(wikitext)
    for k, v in info.items():
        rel = f"has_{k}"
        w_csv.writerow([title, rel, v, title, revid])
        w_edges.write(json.dumps({"head": title, "relation": rel, "tail": v, "source_title": title, "revid": revid}) + "\n")
    w_nodes.write(json.dumps({"title": title, "categories": clean_cats, "revid": revid}) + "\n")
    return 1

# Top-level worker function (pickleable)
def worker_run(idx: int, chunk: List[Dict[str, Any]], wikitext_path: str) -> Tuple[int, str, str, str]:
    # Build per-chunk wikitext map
    need_titles = {normalize_title(rec.get("title", "")) for rec in chunk if rec.get("title")}
    title_to_wt: Dict[str, str] = {}
    wt_p = Path(wikitext_path)
    if wt_p.exists():
        for wrec in load_jsonl(wt_p):
            t = normalize_title(wrec.get("title", ""))
            if t in need_titles:
                wt = wrec.get("rawWikitext") or wrec.get("wikitext") or wrec.get("content") or wrec.get("text") or ""
                title_to_wt[t] = wt
    t_csv = TMP_DIR / f"kg_part_{idx}.csv"
    t_nodes = TMP_DIR / f"kg_nodes_{idx}.jsonl"
    t_edges = TMP_DIR / f"kg_edges_{idx}.jsonl"
    with t_csv.open("w", newline="", encoding="utf-8") as cf, \
         t_nodes.open("w", encoding="utf-8") as nf, \
         t_edges.open("w", encoding="utf-8") as ef:
        wcsv = csv.writer(cf)
        wcsv.writerow(["head", "relation", "tail", "source_title", "revid"])  # header per part
        pages_done = 0
        for rec in chunk:
            pages_done += process_page(rec, title_to_wt, wcsv, nf, ef)
    return pages_done, str(t_csv), str(t_nodes), str(t_edges)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="Make temp snapshot of input JSONL files before processing")
    ap.add_argument("--max-pages", type=int, default=0, help="For quick tests: limit number of pages processed (0 = all)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (>=1). When >1, chunks are processed in parallel and merged.")
    args = ap.parse_args()

    content_src = CONTENT_JSONL
    wikitext_src = WIKITEXT_JSONL
    created_snapshots: List[Path] = []
    if args.snapshot:
        if CONTENT_JSONL.exists():
            cs = snapshot_file(CONTENT_JSONL)
            content_src = cs
            try:
                if Path(cs).parent == TMP_DIR:
                    created_snapshots.append(Path(cs))
            except Exception:
                pass
        if WIKITEXT_JSONL.exists():
            ws = snapshot_file(WIKITEXT_JSONL)
            wikitext_src = ws
            try:
                if Path(ws).parent == TMP_DIR:
                    created_snapshots.append(Path(ws))
            except Exception:
                pass

    # Page processing handled by top-level process_page() for both single and parallel paths.

    # Parallel path
    if args.workers and args.workers > 1:
        # Load all content records (respecting max-pages)
        records = [rec for rec in load_jsonl(content_src, max_lines=args.max_pages)]
        total = len(records)
        workers = max(1, min(args.workers, mp.cpu_count() or args.workers))
        # Partition records
        chunks: List[List[Dict[str, Any]]] = []
        if total == 0:
            chunks = []
        else:
            step = max(1, (total + workers - 1) // workers)
            for i in range(0, total, step):
                chunks.append(records[i:i+step])
        # Spawn workers using top-level worker_run()
        with mp.Pool(processes=workers) as pool:
            async_results = [pool.apply_async(worker_run, (i, chunk, str(wikitext_src))) for i, chunk in enumerate(chunks)]
            results = [r.get() for r in async_results]
        pages_total = 0
        part_csvs: List[str] = []
        part_nodes: List[str] = []
        part_edges: List[str] = []
        for pages_done, pcsv, pnodes, pedges in results:
            pages_total += pages_done
            part_csvs.append(pcsv)
            part_nodes.append(pnodes)
            part_edges.append(pedges)
        # Merge
        OUT_TRIPLES_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_TRIPLES_CSV.open("w", newline="", encoding="utf-8") as cf:
            wfinal = csv.writer(cf)
            wfinal.writerow(["head", "relation", "tail", "source_title", "revid"])  # single header
            for i, p in enumerate(part_csvs):
                with open(p, "r", encoding="utf-8") as pf:
                    # skip header line
                    first = True
                    for line in pf:
                        if first:
                            first = False
                            continue
                        cf.write(line)
        with OUT_NODES_JSONL.open("w", encoding="utf-8") as nf:
            for p in part_nodes:
                with open(p, "r", encoding="utf-8") as pf:
                    for line in pf:
                        nf.write(line)
        with OUT_EDGES_JSONL.open("w", encoding="utf-8") as ef:
            for p in part_edges:
                with open(p, "r", encoding="utf-8") as pf:
                    for line in pf:
                        ef.write(line)
        # Cleanup parts
        for p in (part_csvs + part_nodes + part_edges):
            try:
                os.remove(p)
            except Exception:
                pass
        meta = {
            "generated_at": int(time.time()),
            "content_src": str(content_src),
            "wikitext_src": str(wikitext_src) if Path(wikitext_src).exists() else None,
            "triples_csv": str(OUT_TRIPLES_CSV),
            "nodes_jsonl": str(OUT_NODES_JSONL),
            "edges_jsonl": str(OUT_EDGES_JSONL),
            "pages_processed": pages_total,
            "workers": workers,
        }
        OUT_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Success: auto-clean snapshots created for this build
        try:
            for sp in created_snapshots:
                if sp and sp.exists() and sp.is_file() and sp.parent == TMP_DIR:
                    os.remove(sp)
        except Exception:
            pass
        print(f"✅ Built OSRS KG (parallel x{workers}): {pages_total} pages → {OUT_TRIPLES_CSV.name}, {OUT_NODES_JSONL.name}, {OUT_EDGES_JSONL.name}")
        return

    # ---- Single-process path (original) ----
    # Build title->wikitext map (latest per title)
    title_to_wikitext: Dict[str, str] = {}
    if Path(wikitext_src).exists():
        for rec in load_jsonl(Path(wikitext_src)):
            t = normalize_title(rec.get("title", ""))
            if not t:
                continue
            wt = rec.get("rawWikitext") or rec.get("wikitext") or rec.get("content") or rec.get("text") or ""
            title_to_wikitext[t] = wt

    OUT_TRIPLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TRIPLES_CSV.open("w", newline="", encoding="utf-8") as csvf, \
         OUT_NODES_JSONL.open("w", encoding="utf-8") as ndf, \
         OUT_EDGES_JSONL.open("w", encoding="utf-8") as edf:
        w = csv.writer(csvf)
        w.writerow(["head", "relation", "tail", "source_title", "revid"])  # header
        pages = 0
        for rec in load_jsonl(content_src, max_lines=args.max_pages):
            pages += process_page(rec, title_to_wikitext, w, ndf, edf)
    meta = {
        "generated_at": int(time.time()),
        "content_src": str(content_src),
        "wikitext_src": str(wikitext_src) if Path(wikitext_src).exists() else None,
        "triples_csv": str(OUT_TRIPLES_CSV),
        "nodes_jsonl": str(OUT_NODES_JSONL),
        "edges_jsonl": str(OUT_EDGES_JSONL),
        "pages_processed": pages,
        "workers": 1,
    }
    OUT_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Success: auto-clean snapshots created for this build
    try:
        for sp in created_snapshots:
            if sp and sp.exists() and sp.is_file() and sp.parent == TMP_DIR:
                os.remove(sp)
    except Exception:
        pass
    print(f"✅ Built OSRS KG: {pages} pages → {OUT_TRIPLES_CSV.name}, {OUT_NODES_JSONL.name}, {OUT_EDGES_JSONL.name}")


if __name__ == "__main__":
    main()

