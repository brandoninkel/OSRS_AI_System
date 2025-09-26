#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass
import requests

API = os.environ.get("OSRS_RAG_BASE", "http://localhost:5001")
TESTS = [
    ("What are Vorkath's unique drops?", "boss_vorkath_drops"),
    ("Where is Vorkath located?", "boss_vorkath_loc"),
    ("What is Vorkath's combat level?", "boss_vorkath_stats"),
    ("Describe Zulrah's phases briefly.", "boss_zulrah_strategy"),
    ("Can Zulrah be damaged by melee?", "boss_zulrah_modality"),
    ("What is Zulrah's max hit?", "boss_zulrah_maxhit"),
    ("What are the mechanics of Verzik's green orb?", "boss_verzik_orb"),
    ("What are the requirements to wear Barrows gloves?", "item_barrows_gloves_req"),
    ("What is the attack speed of the Abyssal whip?", "item_whip_speed"),
    ("What is the strength bonus of the Dragon scimitar?", "item_dscim_str"),
    ("Where is Zul-Andra located?", "loc_zul_andra"),
    ("Where is the Barrows minigame located?", "loc_barrows"),
    ("Which Vorkath drops are commonly noted as profitable?", "econ_vorkath_profitable"),
    ("Which Slayer masters can assign Kraken?", "slayer_kraken_assign"),
    ("What are coordinate clues?", "clue_coordinate"),
    ("What does Protect from Magic do?", "pray_protect_magic"),
    ("When was Zulrah released?", "timeline_zulrah_release"),
]


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data" / "test_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # health
    try:
        h = requests.get(f"{API}/health", timeout=10)
        ok = h.status_code == 200
    except Exception:
        ok = False

    results = []
    for q, cid in TESTS:
        t0 = time.time()
        try:
            r = requests.post(f"{API}/search", json={"query": q, "top_k": 10}, timeout=30)
            dt = time.time() - t0
            data = r.json() if r.headers.get('content-type','').startswith('application/json') else {"raw": r.text}
            results.append({
                "chat_id": cid,
                "query": q,
                "status": r.status_code,
                "elapsed_sec": round(dt, 2),
                "total_results": int(data.get("total_results", 0)),
                "top_titles": [e.get("title") for e in data.get("results", [])[:3]],
            })
        except Exception as e:
            dt = time.time() - t0
            results.append({
                "chat_id": cid,
                "query": q,
                "status": None,
                "elapsed_sec": round(dt, 2),
                "error": str(e),
            })

    payload = {
        "timestamp": datetime.now().isoformat(),
        "health": ok,
        "base": API,
        "ran": len(results),
        "results": results,
    }

    fp = out_dir / f"search_battery_{int(time.time())}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(str(fp))


if __name__ == "__main__":
    main()

