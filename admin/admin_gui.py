#!/usr/bin/env python3
"""
Admin Control Panel (Local) using DearPyGui

Controls:
- Start/Stop API, Watchdog, Embedder, GUI/Frontend
- Shows health, embeddings count, and tails of recent logs
- Displays last chat requests parsed from API logs (best effort)

Requirements:
  pip install dearpygui requests

Run:
  python3 OSRS_AI_SYSTEM/admin/admin_gui.py
"""
import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
import re


# Optional: requests for /health polling
try:
    import requests
except Exception:
    requests = None
# Optional: psutil for CPU/RAM gauges
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# DearPyGui
try:
    from dearpygui import dearpygui as dpg
except Exception as e:
    print("DearPyGui not installed. Install with: pip install dearpygui")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs" / "osrs_ai"
API_DIR = REPO_ROOT / "OSRS_AI_SYSTEM" / "api"
SCRIPTS_DIR = REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts"
CPU_CORES = os.cpu_count() or 8
TURBO_WORKERS = str(min(24, max(8, int(CPU_CORES))))
KG_PID_PATH = REPO_ROOT / "OSRS_AI_SYSTEM" / "logs" / "kg" / "train.pid"

# Global log buffer for KG operations
KG_LOG_BUFFER = []

def append_log(message: str):
    """Append a message to the KG log display"""
    global KG_LOG_BUFFER
    timestamp = time.strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    KG_LOG_BUFFER.append(formatted_message)

    # Keep only last 50 messages to prevent memory issues
    if len(KG_LOG_BUFFER) > 50:
        KG_LOG_BUFFER = KG_LOG_BUFFER[-50:]

    # Update the KG log display immediately
    try:
        log_text = "\n".join(KG_LOG_BUFFER)
        dpg.set_value("log_kg", log_text)
    except:
        pass  # GUI might not be ready yet

PID_FILES = {
    "api": LOG_DIR / "api.pid",
    "watchdog": LOG_DIR / "watchdog.pid",
    "embedder": LOG_DIR / "embedder.pid",
    "frontend": LOG_DIR / "frontend.pid",
    "gui": LOG_DIR / "gui.pid",
}

COMMANDS = {
    "start_all": [str(API_DIR / "start-gui.command"), "--with-watchdog"],
    "stop_all": [str(API_DIR / "stop-all.command")],
    "start_api": [str(API_DIR / "start-gui.command")],
    "stop_api": [str(API_DIR / "stop-all.command")],
    "start_data": [str(API_DIR / "start-data.command")],
    "stop_data": [str(API_DIR / "stop-data.command")],
    # Tools
    "build_kg": [str(SCRIPTS_DIR / "knowledge-graph.command"), "--workers", "4"],
    # Strict PyKEEN training (no eval, no fallback), tuned for performance
    "train_kg_pykeen": [
        str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "train-kg-embeddings.command"),
        "--backend", "pykeen", "--no-eval", "--strict",
        "--triples", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "osrs_kg_triples.csv"),
        "--out", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model"),
        "--model", "TransE", "--dimension", "100",
        "--epochs", "25", "--num-workers", "0", "--batch-size", "512"
    ],
    # Optional spectral fallback (fast, low memory)
    "train_kg_spectral": [
        str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "train-kg-embeddings.command"),
        "--backend", "spectral",
        "--triples", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "osrs_kg_triples.csv"),
        "--out", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model"),
        "--dimension", "64"
    ],
    # Chunked evaluation (safe memory)
    "eval_kg": [
        str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "eval-kg-embeddings.command"),
        "--triples", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "osrs_kg_triples.csv"),
        "--out", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model"),
        "--batch-size", "8", "--slice-size", "1", "--device", "auto"
    ],
    # Incremental update (fast): resume artifacts, few epochs
    "train_kg_incremental": [
        str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "train-kg-embeddings.command"),
        "--backend", "pykeen", "--no-eval", "--strict", "--resume",
        "--triples", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "osrs_kg_triples.csv"),
        "--out", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model"),
        "--model", "TransE", "--dimension", "100",
        "--epochs", "5", "--num-workers", "8", "--batch-size", "256"
    ],
    # Clean one-epoch training to align vocab for eval
    "train_kg_one_epoch": [
        str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "train-kg-embeddings.command"),
        "--backend", "pykeen", "--no-eval", "--strict",
        "--triples", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "osrs_kg_triples.csv"),
        "--out", str(REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model"),
        "--model", "TransE", "--dimension", "100",
        "--epochs", "1", "--num-workers", "0", "--batch-size", "512"
    ],
}

LOG_FILES = {
    "api": LOG_DIR / "api.out",
    "watchdog": LOG_DIR / "watchdog.out",
    "embedder": LOG_DIR / "embedder.out",
}

LOG_KG_DIR_PRIMARY = REPO_ROOT / "OSRS_AI_SYSTEM" / "logs" / "kg"
LOG_KG_DIR_FALLBACK = REPO_ROOT / "logs" / "kg"
LOG_KG_DIR = LOG_KG_DIR_PRIMARY if LOG_KG_DIR_PRIMARY.exists() else LOG_KG_DIR_FALLBACK
PROGRESS_PATH = REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model" / "progress.json"
EVAL_PROGRESS_PATH = REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model" / "eval_progress.json"

STATE = {
    "last_health": None,
    "last_stats": None,
    "last_api_log": "",
    "last_watchdog_log": "",
    "last_embedder_log": "",
    "last_kg_log": "",
    "last_eval_log": "",
}

# --- Helpers ---

def run_command(cmd: list[str]):
    try:
        # Run via bash to support .command
        subprocess.Popen(["/bin/bash", "-lc", " ".join(map(lambda s: f'"{s}"' if ' ' in s else s, cmd))])
    except Exception as e:
        print(f"Failed to run: {cmd}: {e}")


def read_tail(path: Path, n: int = 40) -> str:
    try:
        if not path.exists():
            return "(no log)"
        with path.open("r", errors="ignore") as f:
            lines = f.readlines()[-n:]
        return "".join(lines)
    except Exception as e:
        return f"(error reading log: {e})"


def pid_status(name: str) -> str:
    pid_path = PID_FILES.get(name)
    if not pid_path or not pid_path.exists():
        return "stopped"
    try:


        pid = int(pid_path.read_text().strip() or "0")
        if pid <= 0:
            return "stopped"
        # Check if process exists
        os.kill(pid, 0)
        return f"running (PID {pid})"
    except Exception:
        return "unknown"


def latest_kg_log_text(n: int = 80) -> str:
    try:
        # Search both OSRS_AI_SYSTEM/logs/kg and repo-root logs/kg, pick latest
        dirs = [LOG_KG_DIR_PRIMARY, LOG_KG_DIR_FALLBACK]
        candidates = []
        for d in dirs:
            try:
                if d.exists():
                    for p in d.glob("*.log"):
                        candidates.append(p)
            except Exception:
                pass
        if not candidates:
            return "(no kg logs)"
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return read_tail(latest, n=n)
    except Exception as e:
        return f"(error reading kg log: {e})"


def latest_eval_log_text(n: int = 80) -> str:
    try:
        dirs = [LOG_KG_DIR_PRIMARY, LOG_KG_DIR_FALLBACK]
        candidates = []
        for d in dirs:
            try:
                if d.exists():
                    for p in d.glob("eval_*.log"):
                        candidates.append(p)
            except Exception:
                pass
        if not candidates:
            return "(no eval logs)"
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return read_tail(latest, n=n)
    except Exception as e:
        return f"(error reading eval log: {e})"



def clean_kg_for_fresh_eval():
    """Remove stale model artifacts so the next 1-epoch train aligns vocab and eval.
    Deletes: pykeen/, pykeen_ckpt/, entity_to_id.json, relation_to_id.json,
             eval_results.json, eval_progress.json.
    """
    try:
        base = REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model"
        import shutil
        for p in [base / "pykeen", base / "pykeen_ckpt"]:
            try:
                if p.exists():
                    shutil.rmtree(p)
            except Exception:
                pass
        for p in [
            base / "entity_to_id.json",
            base / "relation_to_id.json",
            base / "eval_results.json",
            base / "eval_progress.json",
        ]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
    except Exception as e:
        print(f"(Clean KG) Warning: {e}")


def human_bytes(n: int) -> str:
    try:
        for unit in ("B","KB","MB","GB","TB","PB"):
            if n < 1024.0:
                return f"{n:.1f} {unit}"
            n /= 1024.0
        return f"{n:.1f} EB"
    except Exception:
        return "-"


def read_progress() -> tuple[str, str]:
    """Return pretty progress JSON and a one-line checkpoint summary."""
    try:
        if not PROGRESS_PATH.exists():
            return "(no progress.json)", ""
        with PROGRESS_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        # Parse for a summary line
        try:
            obj = json.loads(txt)
        except Exception:
            obj = {}
        phase = obj.get("phase") or "?"
        epochs_target = obj.get("epochs_target")
        ckpt_bytes = obj.get("checkpoint_bytes")
        ckpt_mtime = obj.get("checkpoint_mtime")
        # Build summary
        summary = [f"phase: {phase}"]
        if epochs_target is not None:
            summary.append(f"target epochs: {epochs_target}")
        if ckpt_mtime:
            try:
                age = max(0.0, time.time() - float(ckpt_mtime))
                summary.append(f"checkpoint: {human_bytes(int(ckpt_bytes or 0))}, updated {age:.0f}s ago")
            except Exception:
                pass
        return txt, " | ".join(summary)
    except Exception as e:
        return f"(error reading progress: {e})", ""


def parse_kg_progress_from_log(text: str):
    """Parse percent/current/total (and optional epoch, speed) from KG training log tail.
    Returns (percent_float_0_1 or None, detail_str, epoch_cur_or_None, epoch_total_or_None, cur_or_None, tot_or_None).
    """
    try:
        lines = text.splitlines() if text else []
        percent = None
        cur = tot = None
        bps = None
        # Find latest batch progress line
        prog_re = re.compile(r"Training batches on .*?:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+).*?(?:,\s*([0-9.]+)batch/s)?")
        for line in reversed(lines):
            m = prog_re.search(line)
            if m:
                percent = float(m.group(1)) / 100.0
                cur = int(m.group(2))
                tot = int(m.group(3))
                if m.group(4):
                    bps = m.group(4)
                break
        # Try to find epoch info (Epoch 1/10, epoch 1 / 10, epoch 1 of 10)
        ep_cur = ep_tot = None
        for line in reversed(lines):
            m = re.search(r"[Ee]poch[^0-9]{0,5}(\d+)\s*(?:/|of)\s*(\d+)", line)
            if m:
                ep_cur, ep_tot = m.group(1), m.group(2)
                break
        # Compose detail string
        parts = []
        if cur is not None and tot is not None and percent is not None:
            parts.append(f"batch {cur}/{tot} ({int(percent*100)}%)")
        if bps:
            parts.append(f"{bps} batch/s")
        detail = " | ".join(parts) if parts else "(no per-batch progress yet)"
        return percent, detail, (int(ep_cur) if ep_cur else None), (int(ep_tot) if ep_tot else None), cur, tot
    except Exception:
        return None, "(no per-batch progress yet)", None, None, None, None

def status_to_color(status: str):
    s = (status or "").lower()
    if s.startswith("running"):
        return (50, 200, 90, 255)   # green
    if s == "stopped":
        return (160, 160, 160, 255) # gray
    return (230, 90, 90, 255)       # red/unknown


def get_system_stats():
    try:
        if psutil is None:
            return None

        cpu = float(psutil.cpu_percent(interval=None))
        mem = float(psutil.virtual_memory().percent)
        return {"cpu": cpu, "mem": mem}
    except Exception:
        return None


def poll_health():
    if not requests:
        return None
    try:
        r = requests.get("http://localhost:5001/health", timeout=1.5)
        if r.ok:
            return r.json()
    except Exception:
        return None


def read_kg_meta_text() -> str:
    try:
        meta_path = REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model" / "meta.json"
        if not meta_path.exists():
            return "(no meta.json)"
        with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        keys = [
            ("backend", obj.get("backend")),
            ("model", obj.get("model")),
            ("dimension", obj.get("dimension")),
            ("epochs", obj.get("epochs")),
            ("batch_size", obj.get("batch_size")),
            ("nodes_count", obj.get("nodes_count")),
            ("triples_count", obj.get("triples_count")),
        ]
        lines = [f"{k}: {v}" for k, v in keys if v is not None]
        return "\n".join(lines) if lines else "(empty meta)"
    except Exception as e:
        return f"(error reading meta.json: {e})"


def parse_recent_chats_from_api_log(text: str, max_items: int = 12):
    out = []
    for line in reversed(text.splitlines()):
        if "/chat" in line and ("POST" in line or "Processing" in line):
            out.append(line.strip())
        if len(out) >= max_items:
            break
    return list(reversed(out))


# --- UI callbacks ---

def refresh_cb():
    STATE["last_api_log"] = read_tail(LOG_FILES["api"]) if LOG_FILES["api"] else ""
    STATE["last_watchdog_log"] = read_tail(LOG_FILES["watchdog"]) if LOG_FILES["watchdog"] else ""
    STATE["last_embedder_log"] = read_tail(LOG_FILES["embedder"]) if LOG_FILES["embedder"] else ""
    STATE["last_health"] = poll_health()

    # KG latest log + progress
    STATE["last_kg_log"] = latest_kg_log_text(300)
    prog_txt, prog_summary = read_progress()
    try:
        dpg.set_value("kg_progress_json", prog_txt)
    except Exception:
        pass
    try:
        dpg.set_value("kg_progress_summary", prog_summary or "(no progress)")
    except Exception:
        pass
    try:
        dpg.set_value("kg_meta", read_kg_meta_text())
    except Exception:
        pass

    # Eval latest log + eval progress
    STATE["last_eval_log"] = latest_eval_log_text(200)
    try:
        if EVAL_PROGRESS_PATH.exists():
            with open(EVAL_PROGRESS_PATH, "r", encoding="utf-8", errors="ignore") as f:
                dpg.set_value("eval_progress_json", f.read())
        else:
            dpg.set_value("eval_progress_json", "(no eval progress)")
    except Exception:
        pass

    # Update status labels with colors
    for name in ["api", "watchdog", "embedder", "frontend"]:
        st = pid_status(name)
        dpg.set_value(f"status_{name}", st)
        try:
            dpg.configure_item(f"status_{name}", color=status_to_color(st))
        except Exception:
            pass

    # Parse KG per-batch progress from log and update the progress bar/detail
    try:
        perc, detail, ep_cur_log, ep_tot_log, cur_b, tot_b = parse_kg_progress_from_log(STATE.get("last_kg_log", ""))
        # parse epochs_target and epoch_current from progress json (preferred)
        ep_cur = None; ep_tot = None
        try:
            obj = json.loads(dpg.get_value("kg_progress_json") or "{}")
            if obj:
                if obj.get("epoch_current") is not None:
                    ep_cur = int(obj.get("epoch_current"))
                if obj.get("epochs_target") is not None:
                    ep_tot = int(obj.get("epochs_target"))
        except Exception:
            pass
        # Fallback to what we parsed from the log
        if ep_cur is None:
            ep_cur = ep_cur_log
        if ep_tot is None:
            ep_tot = ep_tot_log
        # If still unknown, estimate epoch by detecting batch counter wrap-around
        try:
            est = STATE.get("kg_epoch_est") or 1
            prev_cur = STATE.get("kg_prev_cur")
            prev_tot = STATE.get("kg_prev_tot")
            if ep_cur is None and cur_b is not None and tot_b is not None:
                if prev_cur is not None and prev_tot == tot_b and cur_b < max(1, prev_cur // 4):
                    est += 1
                STATE["kg_prev_cur"] = cur_b
                STATE["kg_prev_tot"] = tot_b
                STATE["kg_epoch_est"] = est
                if ep_tot is not None:
                    ep_cur = min(est, ep_tot)
                else:
                    ep_cur = est
        except Exception:
            pass
        # Show epoch line
        if ep_cur is not None and ep_tot is not None:
            rem = max(0, int(ep_tot) - int(ep_cur))
            dpg.set_value("kg_epoch_line", f"epoch {ep_cur}/{ep_tot} (remaining {rem})")
        elif ep_tot is not None:
            dpg.set_value("kg_epoch_line", f"epoch ?/{ep_tot}")
        else:
            dpg.set_value("kg_epoch_line", "epoch ?/ ?")



        # Update bar
        if perc is not None:
            dpg.configure_item("kg_train_bar", default_value=max(0.0, min(1.0, float(perc))), overlay=f"{int(perc*100)}%")
        else:
            dpg.configure_item("kg_train_bar", default_value=0.0, overlay="0%")
        dpg.set_value("kg_train_detail", detail)
    except Exception:
        pass

    # Health JSON and recent chat lines
    dpg.set_value("health_text", json.dumps(STATE["last_health"], indent=2) if STATE["last_health"] else "(no health)")
    chats = parse_recent_chats_from_api_log(STATE["last_api_log"]) if STATE["last_api_log"] else []
    dpg.set_value("recent_chats", "\n".join(chats) if chats else "(no recent chats detected)")

    # System Gauges


def build_kg_cb():
    run_command(COMMANDS["build_kg"])
    time.sleep(0.5); refresh_cb()


def train_kg_incremental_cb():
    run_command(COMMANDS["train_kg_incremental"])  # fast incremental top-up
    time.sleep(0.5); refresh_cb()


def train_kg_cb():
    run_command(COMMANDS["train_kg_pykeen"])  # strict PyKEEN, no eval
    time.sleep(0.5); refresh_cb()

def train_kg_custom_cb(epochs=5, model="TransE", lr=0.01, optimizer="Adam", dimension=100):
    """Custom training with specified hyperparameters"""
    cmd = [
        "bash", str(SCRIPTS_DIR / "train-kg-embeddings.command"),
        "--epochs", str(epochs),
        "--model", model,
        "--learning-rate", str(lr),
        "--optimizer", optimizer,
        "--dimension", str(dimension),
        "--no-eval", "--strict",
        f"--num-workers={TURBO_WORKERS}",
        "--batch-size=512"
    ]
    run_command(" ".join(cmd))
    time.sleep(0.5); refresh_cb()

def train_kg_spectral_cb():
    run_command(COMMANDS["train_kg_spectral"])  # low-memory fallback
    time.sleep(0.5); refresh_cb()

def train_kg_one_epoch_cb():
    clean_kg_for_fresh_eval()
    run_command(COMMANDS["train_kg_one_epoch"])  # 1 epoch, no eval, strict
    time.sleep(0.5); refresh_cb()

# KG Query Interface callbacks
def kg_find_similar_cb():
    """Find similar entities using KG embeddings"""
    try:
        import sys
        sys.path.append(str(REPO_ROOT / "OSRS_AI_SYSTEM" / "api"))
        from kg_query_service import OSRSKGQueryService

        entity_name = dpg.get_value("kg_entity_input")
        if not entity_name.strip():
            dpg.set_value("kg_query_results", "Please enter an entity name")
            return

        kg_service = OSRSKGQueryService()
        similar_entities = kg_service.find_similar_entities(entity_name.strip(), top_k=10)

        if similar_entities:
            results = f"Similar entities to '{entity_name}':\n\n"
            for entity, score in similar_entities:
                results += f"• {entity} (similarity: {score:.3f})\n"
        else:
            results = f"No similar entities found for '{entity_name}'"

        dpg.set_value("kg_query_results", results)

    except Exception as e:
        dpg.set_value("kg_query_results", f"Error: {e}")

def kg_explore_relations_cb():
    """Explore entity relationships"""
    try:
        import sys
        sys.path.append(str(REPO_ROOT / "OSRS_AI_SYSTEM" / "api"))
        from kg_query_service import OSRSKGQueryService

        entity_name = dpg.get_value("kg_entity_input")
        if not entity_name.strip():
            dpg.set_value("kg_query_results", "Please enter an entity name")
            return

        kg_service = OSRSKGQueryService()
        neighborhood = kg_service.explore_entity_neighborhood(entity_name.strip(), max_hops=2)

        if neighborhood and neighborhood.get('total_relationships', 0) > 0:
            results = f"Entity neighborhood for '{entity_name}':\n\n"
            results += f"Total entities: {neighborhood['total_entities']}\n"
            results += f"Total relationships: {neighborhood['total_relationships']}\n\n"

            for level in neighborhood['levels']:
                results += f"Hop {level['hop']} ({level['entity_count']} entities, {level['relationship_count']} relationships):\n"
                for rel in level['relationships'][:10]:  # Show first 10
                    results += f"  {rel['subject']} --{rel['predicate']}--> {rel['object']}\n"
                if level['relationship_count'] > 10:
                    results += f"  ... and {level['relationship_count'] - 10} more\n"
                results += "\n"
        else:
            results = f"No relationships found for '{entity_name}'"

        dpg.set_value("kg_query_results", results)

    except Exception as e:
        dpg.set_value("kg_query_results", f"Error: {e}")

def kg_query_relations_cb():
    """Query by relation type"""
    try:
        import sys
        sys.path.append(str(REPO_ROOT / "OSRS_AI_SYSTEM" / "api"))
        from kg_query_service import OSRSKGQueryService

        relation_type = dpg.get_value("kg_relation_input")
        if not relation_type.strip():
            dpg.set_value("kg_query_results", "Please enter a relation type")
            return

        kg_service = OSRSKGQueryService()
        triples = kg_service.query_by_relation(relation_type.strip(), limit=20)

        if triples:
            results = f"Triples with relation '{relation_type}':\n\n"
            for triple in triples:
                results += f"{triple['subject']} --{triple['predicate']}--> {triple['object']}\n"
        else:
            results = f"No triples found with relation '{relation_type}'"

        dpg.set_value("kg_query_results", results)

    except Exception as e:
        dpg.set_value("kg_query_results", f"Error: {e}")

def kg_stats_cb():
    """Show KG statistics"""
    try:
        import sys
        sys.path.append(str(REPO_ROOT / "OSRS_AI_SYSTEM" / "api"))
        from kg_query_service import OSRSKGQueryService

        kg_service = OSRSKGQueryService()
        stats = kg_service.get_kg_statistics()

        results = "Knowledge Graph Statistics:\n\n"
        results += f"Entities: {stats['entities_count']:,}\n"
        results += f"Relations: {stats['relations_count']:,}\n"
        results += f"Triples: {stats['triples_count']:,}\n"
        results += f"Embeddings loaded: {stats['embeddings_loaded']}\n"
        results += f"Embedding dimension: {stats['embedding_dimension']}\n\n"

        if stats.get('relation_types'):
            results += "Top relation types:\n"
            for rel_type in stats['relation_types']:
                results += f"• {rel_type}\n"

        dpg.set_value("kg_query_results", results)

    except Exception as e:
        dpg.set_value("kg_query_results", f"Error: {e}")

def create_unified_kg_embeddings_cb():
    """Create unified mxbai KG embeddings with live progress"""
    try:
        # Check entity count first
        import json
        entity_map_path = REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model" / "entity_to_id.json"
        if not entity_map_path.exists():
            dpg.set_value("kg_query_results", "❌ No KG entities found. Train KG first!")
            return

        with open(entity_map_path, 'r') as f:
            entity_count = len(json.load(f))

        # Warn about large entity counts
        if entity_count > 10000:
            hours_estimate = entity_count / 1000  # Rough estimate: 1000 entities per hour
            dpg.set_value("kg_query_results", f"⚠️ WARNING: {entity_count:,} entities detected!\n\nThis will take approximately {hours_estimate:.1f} hours.\n\nFor testing, consider:\n1. Training KG with fewer epochs first\n2. Using a subset of entities\n\nPress the button again to proceed anyway.")
            return

        dpg.set_value("kg_query_results", f"🚀 Starting unified mxbai KG embeddings...\nEntities: {entity_count:,}\nEstimated time: {entity_count/1000:.1f} hours\n\nProgress will update every 50 entities...")

        # Start the process in background with live progress
        import subprocess
        import threading

        def run_with_progress():
            try:
                process = subprocess.Popen([
                    "python3",
                    str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "kg" / "create_mxbai_kg_embeddings.py")
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                   cwd=str(REPO_ROOT))

                output_lines = []
                for line in process.stdout:
                    output_lines.append(line.strip())
                    # Update GUI with latest progress
                    if "Processed" in line and "entities" in line:
                        recent_output = '\n'.join(output_lines[-10:])  # Last 10 lines
                        dpg.set_value("kg_query_results", f"🔄 IN PROGRESS...\n\n{recent_output}")

                process.wait()

                if process.returncode == 0:
                    final_output = '\n'.join(output_lines[-20:])  # Last 20 lines
                    dpg.set_value("kg_query_results", f"✅ SUCCESS!\n\n{final_output}")
                else:
                    error_output = '\n'.join(output_lines[-20:])
                    dpg.set_value("kg_query_results", f"❌ FAILED!\n\n{error_output}")

            except Exception as e:
                dpg.set_value("kg_query_results", f"❌ Error: {e}")

        # Start in background thread
        thread = threading.Thread(target=run_with_progress, daemon=True)
        thread.start()

    except Exception as e:
        dpg.set_value("kg_query_results", f"❌ Error: {e}")

def create_sample_kg_embeddings_cb():
    """Create sample mxbai KG embeddings (1000 entities for testing)"""
    try:
        dpg.set_value("kg_query_results", "🧪 Starting SAMPLE unified mxbai KG embeddings...\nSample size: 1000 entities\nEstimated time: ~1 hour\n\nProgress will update every 25 entities...")

        # Start the sample process in background with live progress
        import subprocess
        import threading

        def run_sample_with_progress():
            try:
                process = subprocess.Popen([
                    "python3",
                    str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "kg" / "create_mxbai_kg_embeddings_sample.py"),
                    "--sample-size", "1000"
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                   cwd=str(REPO_ROOT))

                output_lines = []
                for line in process.stdout:
                    output_lines.append(line.strip())
                    # Update GUI with latest progress
                    if "Processed" in line and "entities" in line:
                        recent_output = '\n'.join(output_lines[-8:])  # Last 8 lines
                        dpg.set_value("kg_query_results", f"🔄 SAMPLE IN PROGRESS...\n\n{recent_output}")

                process.wait()

                if process.returncode == 0:
                    final_output = '\n'.join(output_lines[-15:])  # Last 15 lines
                    dpg.set_value("kg_query_results", f"✅ SAMPLE SUCCESS!\n\n{final_output}\n\n🎯 Test this sample with RAG chat, then run full version if satisfied!")
                else:
                    error_output = '\n'.join(output_lines[-15:])
                    dpg.set_value("kg_query_results", f"❌ SAMPLE FAILED!\n\n{error_output}")

            except Exception as e:
                dpg.set_value("kg_query_results", f"❌ Sample Error: {e}")

        # Start in background thread
        thread = threading.Thread(target=run_sample_with_progress, daemon=True)
        thread.start()

    except Exception as e:
        dpg.set_value("kg_query_results", f"❌ Error: {e}")

def test_mxbai_performance_cb():
    """Test mxbai embedding performance to find optimal worker count"""
    try:
        dpg.set_value("kg_query_results", "🧪 Testing mxbai embedding performance...\nThis will test different worker counts to find optimal settings.\nTesting 100 entities with various worker configurations...")

        # Start the performance test in background with live progress
        import subprocess
        import threading

        def run_performance_test():
            try:
                process = subprocess.Popen([
                    "python3",
                    str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "kg" / "test_mxbai_performance.py")
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                   cwd=str(REPO_ROOT))

                output_lines = []
                for line in process.stdout:
                    output_lines.append(line.strip())
                    # Update GUI with latest progress
                    if any(keyword in line for keyword in ["Testing", "Workers:", "Optimal", "Recommended"]):
                        recent_output = '\n'.join(output_lines[-15:])  # Last 15 lines
                        dpg.set_value("kg_query_results", f"🧪 PERFORMANCE TEST IN PROGRESS...\n\n{recent_output}")

                process.wait()

                if process.returncode == 0:
                    final_output = '\n'.join(output_lines[-25:])  # Last 25 lines
                    dpg.set_value("kg_query_results", f"✅ PERFORMANCE TEST COMPLETE!\n\n{final_output}")
                else:
                    error_output = '\n'.join(output_lines[-20:])
                    dpg.set_value("kg_query_results", f"❌ PERFORMANCE TEST FAILED!\n\n{error_output}")

            except Exception as e:
                dpg.set_value("kg_query_results", f"❌ Performance Test Error: {e}")

        # Start in background thread
        thread = threading.Thread(target=run_performance_test, daemon=True)
        thread.start()

    except Exception as e:
        dpg.set_value("kg_query_results", f"❌ Error: {e}")

def create_optimized_kg_embeddings_cb():
    """Create optimized full KG embeddings using performance test results"""
    try:
        # Check if performance results exist
        results_file = REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "mxbai_performance_results.json"
        optimal_workers = 16  # Default fallback

        if results_file.exists():
            import json
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                # Find best result
                if results:
                    successful_results = [r for r in results if r.get('success_rate', 0) > 0.8]
                    if successful_results:
                        best_result = max(successful_results, key=lambda x: x.get('entities_per_second', 0) * (1.0 if x.get('cpu_percent', 100) < 90 else 0.5))
                        optimal_workers = best_result.get('workers', 16)
            except:
                pass

        # Check entity count first
        import json
        entity_map_path = REPO_ROOT / "OSRS_AI_SYSTEM" / "data" / "kg_model" / "entity_to_id.json"
        if not entity_map_path.exists():
            dpg.set_value("kg_query_results", "❌ No KG entities found. Train KG first!")
            return

        with open(entity_map_path, 'r') as f:
            entity_count = len(json.load(f))

        # Estimate time with optimal workers
        estimated_speed = 2.0  # Conservative estimate: 2 entities/second
        hours_estimate = entity_count / (estimated_speed * optimal_workers) / 3600

        dpg.set_value("kg_query_results", f"🚀 Starting OPTIMIZED unified mxbai KG embeddings...\n\nConfiguration:\n• Entities: {entity_count:,}\n• Workers: {optimal_workers}\n• Estimated time: {hours_estimate:.1f} hours\n\nThis will run in the background. Check system monitor for progress.\nThe process will continue even if you close this GUI.")

        # Start the optimized process in background
        import subprocess
        import threading

        def run_optimized_embeddings():
            try:
                # Run in completely detached mode
                subprocess.Popen([
                    "nohup",
                    "python3",
                    str(REPO_ROOT / "OSRS_AI_SYSTEM" / "scripts" / "kg" / "create_mxbai_kg_embeddings.py"),
                    "--workers", str(optimal_workers)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   cwd=str(REPO_ROOT), start_new_session=True)

                # Update GUI to show it started
                time.sleep(2)
                dpg.set_value("kg_query_results", f"✅ OPTIMIZED EMBEDDING CREATION STARTED!\n\n• Process running in background with {optimal_workers} workers\n• Estimated completion: {hours_estimate:.1f} hours\n• Check system monitor for CPU/memory usage\n• Output file: kg_entity_embeddings_mxbai.jsonl\n\n🎯 The process will continue running even if you close this application.")

            except Exception as e:
                dpg.set_value("kg_query_results", f"❌ Failed to start optimized process: {e}")

        # Start in background thread
        thread = threading.Thread(target=run_optimized_embeddings, daemon=True)
        thread.start()

    except Exception as e:
        dpg.set_value("kg_query_results", f"❌ Error: {e}")

def create_enhanced_kg_embeddings_cb():
    """Create enhanced KG embeddings with Apple Metal GPU acceleration"""
    def run_enhanced():
        try:
            append_log("🚀 Starting mxbai KG embeddings creation...")
            append_log("✅ Features: Apple Metal GPU acceleration, parallel processing")
            append_log("📊 Processing ~148,000 entities with 8 workers")
            append_log("⏱️  Estimated time: ~1.4 hours")
            append_log("")
            append_log("🔄 Process running in background...")
            append_log("📁 Output: OSRS_AI_SYSTEM/data/kg_entity_embeddings_mxbai.jsonl")

            # Run in background without capturing output (so it doesn't block)
            process = subprocess.Popen([
                sys.executable,
                "scripts/kg/create_mxbai_kg_embeddings.py",
                "--workers", "8"
            ], cwd=REPO_ROOT / "OSRS_AI_SYSTEM")

            append_log(f"✅ Process started with PID: {process.pid}")
            append_log("💡 Check terminal or file system for progress updates")
            append_log("🎯 When complete, RAG will automatically use unified embeddings!")

        except Exception as e:
            append_log(f"❌ Failed to start KG embeddings: {e}")

    threading.Thread(target=run_enhanced, daemon=True).start()

def stop_kg_cb():
    try:
        pid_path = KG_PID_PATH
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        if not pid_path.exists():
            print("(Stop KG) No PID file found; attempting pattern kill…")
            # Attempt a pattern-based graceful stop as fallback
            try:
                subprocess.run(["/bin/bash", "-lc",
                                "pkill -TERM -f 'OSRS_AI_SYSTEM/scripts/kg/train_kg_embeddings.py|train-kg-embeddings.command' || true"],
                               timeout=3)
                time.sleep(2)
                subprocess.run(["/bin/bash", "-lc",
                                "pkill -KILL -f 'OSRS_AI_SYSTEM/scripts/kg/train_kg_embeddings.py|train-kg-embeddings.command' || true"],
                               timeout=3)
            except Exception:
                pass
            return
        pid = int(pid_path.read_text().strip() or "0")
        if pid > 0:
            try:
                os.kill(pid, 15)  # SIGTERM
            except Exception:
                pass
            # Wait up to ~5s, then SIGKILL if still alive
            for _ in range(5):
                try:
                    os.kill(pid, 0)
                    time.sleep(1)
                except Exception:
                    break
            else:
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
        try:
            pid_path.unlink(missing_ok=True)
        except Exception:
            pass
    finally:
        time.sleep(0.5)
        refresh_cb()

    if stats is not None:
        try:
            cpu_v = max(0.0, min(100.0, float(stats.get("cpu", 0.0))))
            mem_v = max(0.0, min(100.0, float(stats.get("mem", 0.0))))


            dpg.configure_item("cpu_bar", default_value=cpu_v/100.0, overlay=f"CPU {cpu_v:.0f}%")
            dpg.configure_item("ram_bar", default_value=mem_v/100.0, overlay=f"RAM {mem_v:.0f}%")
        except Exception:
            pass

    # Logs (tabbed)
    dpg.set_value("log_api", STATE["last_api_log"])
    dpg.set_value("log_watchdog", STATE["last_watchdog_log"])
    dpg.set_value("log_embedder", STATE["last_embedder_log"])
    dpg.set_value("log_kg", STATE["last_kg_log"])
    dpg.set_value("log_eval", STATE["last_eval_log"])


def start_all_cb():
    run_command(COMMANDS["start_all"])
    time.sleep(0.5); refresh_cb()



def eval_kg_cb():
    run_command(COMMANDS["eval_kg"])  # chunked evaluation
    time.sleep(0.5); refresh_cb()


def stop_all_cb():
    run_command(COMMANDS["stop_all"])
    time.sleep(0.5); refresh_cb()


def start_data_cb():
    run_command(COMMANDS["start_data"])
    time.sleep(0.5); refresh_cb()


def stop_data_cb():
    run_command(COMMANDS["stop_data"])
    time.sleep(0.5); refresh_cb()


def start_api_cb():
    run_command(COMMANDS["start_api"])
    time.sleep(0.5); refresh_cb()


def stop_api_cb():
    run_command(COMMANDS["stop_api"])
    time.sleep(0.5); refresh_cb()



# Create DearPyGui context BEFORE creating any windows/items
dpg.create_context()

with dpg.window(label="Tools", width=530, height=200, pos=(550, 470)):
    dpg.add_text("Knowledge Graph")
    with dpg.group(horizontal=False):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Build KG (links/categories)", callback=build_kg_cb)
            dpg.add_text("Build the triples CSV from wiki data")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Train KG (Quick Test)", callback=lambda: train_kg_custom_cb(epochs=1, model="TransE"))
            dpg.add_text("1 epoch TransE - fast test run")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Train KG (Standard)", callback=lambda: train_kg_custom_cb(epochs=5, model="TransE"))
            dpg.add_text("5 epochs TransE - balanced training")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Train KG (High Performance)", callback=lambda: train_kg_custom_cb(epochs=10, model="ComplEx", lr=0.005))
            dpg.add_text("10 epochs ComplEx - best quality")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Stop KG", callback=stop_kg_cb)
            dpg.add_text("Gracefully stop current KG training")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Train KG 1 Epoch (clean)", callback=train_kg_one_epoch_cb)
            dpg.add_text("Reset artifacts and run 1 epoch to align vocab for eval")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Evaluate KG (chunked)", callback=eval_kg_cb)
            dpg.add_text("Run filtered metrics on test split in small batches; safe memory")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Train KG (Spectral)", callback=train_kg_spectral_cb)
            dpg.add_text("Fast low-memory node embedding (fallback)")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Incremental Update (fast)", callback=train_kg_incremental_cb)
            dpg.add_text("Append new pages/entities; quick 5-epoch top-up, no eval")

        dpg.add_separator()
        dpg.add_text("KG Query Interface", color=[255, 255, 0])

        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="Enter entity name (e.g., 'Abyssal whip')", width=300, tag="kg_entity_input")
            dpg.add_button(label="Find Similar", callback=kg_find_similar_cb)
            dpg.add_button(label="Explore Relations", callback=kg_explore_relations_cb)

        with dpg.group(horizontal=True):
            dpg.add_input_text(hint="Enter relation type (e.g., 'drops')", width=300, tag="kg_relation_input")
            dpg.add_button(label="Query Relations", callback=kg_query_relations_cb)
            dpg.add_button(label="KG Stats", callback=kg_stats_cb)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Test Performance", callback=test_mxbai_performance_cb)
            dpg.add_text("Find optimal worker count for your system")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Create Sample KG Embeddings", callback=create_sample_kg_embeddings_cb)
            dpg.add_text("Generate sample mxbai embeddings (1000 entities, ~1 hour)")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Create Full KG Embeddings (Optimized)", callback=create_optimized_kg_embeddings_cb)
            dpg.add_text("Generate ALL mxbai embeddings with optimal performance")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Create KG Embeddings (mxbai)", callback=create_enhanced_kg_embeddings_cb)
            dpg.add_text("Unified mxbai embedding space + Apple Metal GPU (~1.4 hours)")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Test Log Display", callback=lambda: append_log("🧪 Test message - logging is working!"))
            dpg.add_text("Click to test if logging works (check KG tab below)")


# --- Build UI ---
dpg.create_viewport(title="OSRS Admin Console", width=1100, height=800)

with dpg.window(label="Controls", tag="controls_window", width=1080, height=140, pos=(10, 10)):
    dpg.add_text("System Status:")
    with dpg.group(horizontal=True):
        dpg.add_text("API:")
        dpg.add_text("stopped", tag="status_api")
        dpg.add_spacer(width=12)
        dpg.add_text("Watchdog:")
        dpg.add_text("stopped", tag="status_watchdog")
        dpg.add_spacer(width=12)
        dpg.add_text("Embedder:")
        dpg.add_text("stopped", tag="status_embedder")
        dpg.add_spacer(width=12)
        dpg.add_text("Frontend:")
        dpg.add_text("stopped", tag="status_frontend")

    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_button(label="Start ALL (API+GUI+Watchdog+Embedder)", callback=start_all_cb)
        dpg.add_button(label="Stop ALL", callback=stop_all_cb)
        dpg.add_button(label="Start DATA (Watchdog+Embedder)", callback=start_data_cb)
        dpg.add_button(label="Stop DATA", callback=stop_data_cb)
        dpg.add_button(label="Start API", callback=start_api_cb)
        dpg.add_button(label="Stop API", callback=stop_api_cb)
        dpg.add_button(label="Refresh", callback=refresh_cb)

with dpg.window(label="API Health / Stats", width=530, height=300, pos=(10, 160)):
    dpg.add_text("/health JSON:")
    dpg.add_input_text(multiline=True, width=500, height=240, readonly=True, tag="health_text")

with dpg.window(label="Recent Chats (from api.out)", width=540, height=300, pos=(550, 160)):
    dpg.add_input_text(multiline=True, width=510, height=260, readonly=True, tag="recent_chats")

with dpg.window(label="System Gauges", width=530, height=140, pos=(10, 470)):
    dpg.add_text("CPU / RAM")
    dpg.add_progress_bar(tag="cpu_bar", width=500, overlay="CPU --%")
    dpg.add_spacer(height=5)
    dpg.add_progress_bar(tag="ram_bar", width=500, overlay="RAM --%")

with dpg.window(label="Logs", width=1080, height=170, pos=(10, 620)):
    with dpg.tab_bar():
        with dpg.tab(label="API"):
            dpg.add_input_text(multiline=True, width=1040, height=120, readonly=True, tag="log_api")
        with dpg.tab(label="Watchdog"):
            dpg.add_input_text(multiline=True, width=1040, height=120, readonly=True, tag="log_watchdog")
        with dpg.tab(label="Embedder"):
            dpg.add_input_text(multiline=True, width=1040, height=120, readonly=True, tag="log_embedder")
        with dpg.tab(label="KG"):
            dpg.add_text("phase: -", tag="kg_progress_summary")
            dpg.add_text("epoch ?/ ?", tag="kg_epoch_line")
            dpg.add_progress_bar(tag="kg_train_bar", width=1040, overlay="0%")
            dpg.add_text("(no per-batch progress yet)", tag="kg_train_detail")
            dpg.add_input_text(multiline=True, width=1040, height=90, readonly=True, tag="kg_progress_json")
            dpg.add_text("— Model meta —")
            dpg.add_input_text(multiline=True, width=1040, height=70, readonly=True, tag="kg_meta")
            dpg.add_separator()
            dpg.add_input_text(multiline=True, width=1040, height=120, readonly=True, tag="log_kg")
        with dpg.tab(label="KG Eval"):
            dpg.add_text("Eval progress (eval_progress.json):")
            dpg.add_input_text(multiline=True, width=1040, height=120, readonly=True, tag="eval_progress_json")
            dpg.add_separator()
            dpg.add_text("Eval log:")
            dpg.add_input_text(multiline=True, width=1040, height=120, readonly=True, tag="log_eval")
        with dpg.tab(label="KG Query"):
            dpg.add_text("Knowledge Graph Query Results:")
            dpg.add_input_text(multiline=True, width=1040, height=400, readonly=True, tag="kg_query_results", default_value="Use the KG Query Interface buttons in the Tools section to explore the knowledge graph.")

refresh_cb()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("controls_window", True)

# Manual render loop with 1s auto-refresh throttle
_last = 0.0
while dpg.is_dearpygui_running():
    now = time.time()
    if now - _last > 1.0:
        try:
            refresh_cb()
        except Exception:
            pass
        _last = now
    dpg.render_dearpygui_frame()

# Clean shutdown
dpg.destroy_context()
