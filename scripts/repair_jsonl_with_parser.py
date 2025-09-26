#!/usr/bin/env python3
"""
Repair osrs_wiki_content.jsonl using the real template parser when rawWikitext is present.
- Reads data/osrs_wiki_content.jsonl line-by-line (no full-file load)
- For entries that have 'rawWikitext', re-run OSRSWikiTemplateParser.process_wiki_content(raw)
  and replace the 'text' field with the properly labeled output (colons, breaks, etc.)
- For entries without 'rawWikitext', leave as-is and record titles to data/repair_missing_raw.txt
- Writes data/osrs_wiki_content.repaired.jsonl; if --write is passed, backs up original to
  data/osrs_wiki_content.jsonl.bak and atomically replaces the original

This script does NOT refetch. It uses only existing rawWikitext when available.
"""
import os
import sys
import json
import tempfile
from typing import Tuple

# Repo-relative paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
IN_PATH = os.path.join(DATA_DIR, 'osrs_wiki_content.jsonl')
OUT_PATH = os.path.join(DATA_DIR, 'osrs_wiki_content.repaired.jsonl')
MISS_PATH = os.path.join(DATA_DIR, 'repair_missing_raw.txt')

# Import parser
API_DIR = os.path.join(ROOT, 'api')
sys.path.append(API_DIR)
from wiki_template_parser import OSRSWikiTemplateParser  # type: ignore


def process_one(parser: OSRSWikiTemplateParser, rec: dict) -> Tuple[dict, bool, bool]:
    """Process a single JSONL record. Returns (updated_rec, changed, had_raw)."""
    had_raw = bool(rec.get('rawWikitext'))
    if not had_raw:
        return rec, False, False

    try:
        raw = rec.get('rawWikitext', '')
        processed = parser.process_wiki_content(raw)
        # Replace text with parser output; preserve everything else
        if processed and processed.strip() and processed.strip() != rec.get('text', '').strip():
            rec['text'] = processed.strip()
            rec['corrected'] = True
            return rec, True, True
        return rec, False, True
    except Exception as e:
        # Keep original on error
        rec.setdefault('parser_error', str(e))
        return rec, False, True


def main():
    write_mode = '--write' in sys.argv
    if not os.path.exists(IN_PATH):
        print(f"❌ Input not found: {IN_PATH}")
        sys.exit(1)

    parser = OSRSWikiTemplateParser()

    total = 0
    changed = 0
    had_raw = 0
    missing = 0

    # Prepare temp output and missing list
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix='osrs_wiki_content.', suffix='.jsonl', dir=DATA_DIR)
    os.close(tmp_fd)

    miss_titles = []

    with open(IN_PATH, 'r', encoding='utf-8') as fin, open(tmp_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            # Always preserve title/categories
            title = rec.get('title')
            cats = rec.get('categories')
            if not isinstance(title, str) or not title:
                # Skip invalid entries
                continue

            updated, did_change, had = process_one(parser, rec)
            if had:
                had_raw += 1
            else:
                missing += 1
                miss_titles.append(title)

            if did_change:
                changed += 1

            fout.write(json.dumps(updated, ensure_ascii=False) + "\n")

    # Write repaired file
    os.replace(tmp_path, OUT_PATH)

    # Write missing list
    if miss_titles:
        with open(MISS_PATH, 'w', encoding='utf-8') as mf:
            mf.write("\n".join(sorted(set(miss_titles))) + "\n")

    print(f"✅ Repair pass complete")
    print(f"   Total lines: {total}")
    print(f"   With rawWikitext: {had_raw}")
    print(f"   Missing rawWikitext: {missing} (listed in {MISS_PATH if miss_titles else 'n/a'})")
    print(f"   Updated with labeled parsing: {changed}")
    print(f"   Output: {OUT_PATH}")

    if write_mode:
        # Backup and swap
        bak_path = IN_PATH + '.bak'
        if os.path.exists(bak_path):
            os.remove(bak_path)
        os.replace(IN_PATH, bak_path)
        os.replace(OUT_PATH, IN_PATH)
        print(f"📝 Replaced original with repaired file. Backup at {bak_path}")
    else:
        print("ℹ️ Dry run: original not replaced. Pass --write to apply.")


if __name__ == '__main__':
    main()

