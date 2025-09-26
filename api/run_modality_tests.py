#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure this script's directory is on sys.path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Default fast settings unless overridden by env
os.environ.setdefault("OSRS_USE_RERANKER", "0")
os.environ.setdefault("OSRS_EXCERPTS_PER_DOC", "1")

from osrs_rag_service import OSRSRAGService


def run_battery(reranker: bool, tests: list[tuple[str, str]], top_k: int = 10) -> dict:
    os.environ["OSRS_USE_RERANKER"] = "1" if reranker else "0"
    svc = OSRSRAGService()
    results = []
    for q, cid in tests:
        t0 = time.time()
        out = svc.query(question=q, top_k=top_k, show_sources=True, chat_id=cid)
        dt = time.time() - t0
        results.append({
            "chat_id": cid,
            "query": q,
            "response_preview": (out.get("response", "")[:900] + ("..." if len(out.get("response", "")) > 900 else "")),
            "sources_count": len(out.get("sources", [])) if isinstance(out.get("sources"), list) else 0,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(dt, 2),
        })
    return {
        "reranker": reranker,
        "ran": len(results),
        "results": results,
    }


def main():
    # Diverse modalities across entities (unique chat_ids for zero contamination)
    tests = [
        # Bosses / Drops / Strategy / Damage modality / Stats
        ("What are Vorkath's unique drops? Cite sources.", "boss_vorkath_drops"),
        ("Where is Vorkath located? Cite sources.", "boss_vorkath_loc"),
        ("What is Vorkath's combat level? Cite sources.", "boss_vorkath_stats"),
        ("Describe Zulrah's phases briefly. Cite sources.", "boss_zulrah_strategy"),
        ("Can Zulrah be damaged by melee? Answer Yes or No and cite sources.", "boss_zulrah_modality"),
        ("What is Zulrah's max hit? Cite sources.", "boss_zulrah_maxhit"),
        ("What are the mechanics of Verzik's green orb, and how do you avoid it? Cite sources.", "boss_verzik_orb"),
        # Items / Equipment stats / Requirements
        ("What are the requirements to wear Barrows gloves? Cite sources.", "item_barrows_gloves_req"),
        ("What is the attack speed of the Abyssal whip? Cite sources.", "item_whip_speed"),
        ("What is the strength bonus of the Dragon scimitar? Cite sources.", "item_dscim_str"),
        # Locations / Minigames
        ("Where is Zul-Andra located? Cite sources.", "loc_zul_andra"),
        ("Where is the Barrows minigame located? Cite sources.", "loc_barrows"),
        # Economy
        ("Which Vorkath drops are commonly noted as profitable? Cite sources.", "econ_vorkath_profitable"),
        # Slayer
        ("Which Slayer masters can assign Kraken? Cite sources.", "slayer_kraken_assign"),
        # Clue
        ("What are coordinate clues? Cite sources.", "clue_coordinate"),
        # Spells/Prayers
        ("What does Protect from Magic do? Cite sources.", "pray_protect_magic"),
        # Timeline
        ("When was Zulrah released? Cite sources.", "timeline_zulrah_release"),
    ]

    # Output directory
    out_dir = SCRIPT_DIR.parent / "data" / "test_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass A: reranker OFF (fast, deterministic for extraction)
    try:
        res_off = run_battery(False, tests)
        fn_off = out_dir / f"modality_run_off_{int(time.time())}.json"
        fn_off.write_text(json.dumps(res_off, ensure_ascii=False, indent=2))

        # Print summary for quick read
        print(json.dumps({
            "summary": {
                "reranker": False,
                "ran": res_off["ran"],
                "avg_elapsed": round(sum(r["elapsed_sec"] for r in res_off["results"]) / max(1, res_off["ran"]), 2)
            }
        }, ensure_ascii=False))
    except Exception as e:
        err = {"error": str(e), "when": "reranker_off", "timestamp": datetime.now().isoformat()}
        fn_err = out_dir / f"modality_run_off_error_{int(time.time())}.json"
        fn_err.write_text(json.dumps(err, ensure_ascii=False, indent=2))
        print(json.dumps({"summary": {"reranker": False, "error": str(e)}}, ensure_ascii=False))

    # Optionally Pass B: reranker ON if allowed by env (can be heavy)
    if os.environ.get("OSRS_ALLOW_RERANKER", "0") == "1":
        try:
            res_on = run_battery(True, tests)
            fn_on = out_dir / f"modality_run_on_{int(time.time())}.json"
            fn_on.write_text(json.dumps(res_on, ensure_ascii=False, indent=2))
            print(json.dumps({
                "summary": {
                    "reranker": True,
                    "ran": res_on["ran"],
                    "avg_elapsed": round(sum(r["elapsed_sec"] for r in res_on["results"]) / max(1, res_on["ran"]), 2)
                }
            }, ensure_ascii=False))
        except Exception as e:
            err = {"error": str(e), "when": "reranker_on", "timestamp": datetime.now().isoformat()}
            fn_err = out_dir / f"modality_run_on_error_{int(time.time())}.json"
            fn_err.write_text(json.dumps(err, ensure_ascii=False, indent=2))
            print(json.dumps({"summary": {"reranker": True, "error": str(e)}}, ensure_ascii=False))


if __name__ == "__main__":
    main()

