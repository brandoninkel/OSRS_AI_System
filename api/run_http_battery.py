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

ROOT = Path(__file__).resolve().parent.parent
API = os.environ.get("OSRS_RAG_BASE", "http://localhost:5001")
TESTS = [
    ("What are Vorkath's unique drops? Cite sources.", "boss_vorkath_drops"),
    ("Where is Vorkath located? Cite sources.", "boss_vorkath_loc"),
    ("What is Vorkath's combat level? Cite sources.", "boss_vorkath_stats"),
    ("Describe Zulrah's phases briefly. Cite sources.", "boss_zulrah_strategy"),
    ("Can Zulrah be damaged by melee? Answer Yes or No and cite sources.", "boss_zulrah_modality"),
    ("What is Zulrah's max hit? Cite sources.", "boss_zulrah_maxhit"),
    ("What are the mechanics of Verzik's green orb, and how do you avoid it? Cite sources.", "boss_verzik_orb"),
    ("What are the requirements to wear Barrows gloves? Cite sources.", "item_barrows_gloves_req"),
    ("What is the attack speed of the Abyssal whip? Cite sources.", "item_whip_speed"),
    ("What is the strength bonus of the Dragon scimitar? Cite sources.", "item_dscim_str"),
    ("Where is Zul-Andra located? Cite sources.", "loc_zul_andra"),
    ("Where is the Barrows minigame located? Cite sources.", "loc_barrows"),
    ("Which Vorkath drops are commonly noted as profitable? Cite sources.", "econ_vorkath_profitable"),
    ("Which Slayer masters can assign Kraken? Cite sources.", "slayer_kraken_assign"),
    ("What are coordinate clues? Cite sources.", "clue_coordinate"),
    ("What does Protect from Magic do? Cite sources.", "pray_protect_magic"),
    ("When was Zulrah released? Cite sources.", "timeline_zulrah_release"),
    # Paraphrase robustness checks (modality intent wording variations)
    ("List Vorkath-only loot; cite pages.", "p_vorkath_drops"),
    ("Pin down Vorkath's location with references.", "p_vorkath_loc"),
    ("Give Vorkath combat level from sources only.", "p_vorkath_stats"),
    ("Briefly outline Zulrah phase order; cite.", "p_zulrah_strategy"),
    ("Is melee viable on Zulrah? Answer yes/no with a quote.", "p_zulrah_modality"),
    ("State Zulrah maximum damage; attach sources.", "p_zulrah_maxhit"),
]


def main():
    out_dir = ROOT / "data" / "test_runs"
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
            r = requests.post(f"{API}/chat", json={"query": q, "chat_id": cid, "top_k": 10, "show_sources": True}, timeout=180)
            dt = time.time() - t0
            try:
                data = r.json()
            except Exception:
                data = {"success": False, "raw": r.text}
            srcs = data.get("sources", [])
            # Normalize and truncate sources for evaluation without huge files
            norm_sources = []
            if isinstance(srcs, list):
                for s in srcs[:10]:
                    if not isinstance(s, dict):
                        continue
                    norm_sources.append({
                        "title": s.get("title"),
                        "url": s.get("url"),
                        "snippet": (s.get("snippet", "")[:600] + ("..." if len(s.get("snippet", "")) > 600 else ""))
                    })
            resp = data.get("response", "")
            results.append({
                "chat_id": cid,
                "query": q,
                "status": r.status_code,
                "elapsed_sec": round(dt, 2),
                "success": bool(data.get("success", False)),
                "response_preview": (resp[:900] + ("..." if len(resp) > 900 else "")),
                "sources_count": len(srcs) if isinstance(srcs, list) else 0,
                "sources": norm_sources,
            })
        except Exception as e:
            dt = time.time() - t0
            results.append({
                "chat_id": cid,
                "query": q,
                "status": None,
                "elapsed_sec": round(dt, 2),
                "success": False,
                "error": str(e),
            })

    payload = {
        "timestamp": datetime.now().isoformat(),
        "health": ok,
        "base": API,
        "ran": len(results),
        "results": results,
    }

    fp = out_dir / f"http_battery_{int(time.time())}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(str(fp))


if __name__ == "__main__":
    main()

