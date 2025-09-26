#!/usr/bin/env python3
"""
Wiki Attribution Service (Python)
- Provides per-revision attestation for a specific snippet from a wiki page
- Uses MediaWiki REST API endpoints:
  * GET /rest.php/v1/page/{title}/history  (recent revisions list)
  * GET /rest.php/v1/revision/{id}         (revision content, field 'source')
- STRICT SERIAL requests, respectful User-Agent, short sleeps for etiquette
"""
from __future__ import annotations

import requests
import time
from typing import Dict, Any, List, Optional
import re
import urllib.parse
import os
import json
import fcntl
import hashlib
from datetime import datetime

DEFAULT_UA = "OSRS-AI/1.0 (contact: dev@local)"
WIKI_REST_BASE = "https://oldschool.runescape.wiki/rest.php/v1"
CACHE_DIR = "/Users/brandon/Documents/projects/GE/data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "attribution_cache.json")


def _normalize_text(s: str) -> str:
    # collapse whitespace, decode HTML entities, normalize quotes
    import html
    s = html.unescape(s or "")
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()


class WikiAttributionService:
    def __init__(self, user_agent: str = DEFAULT_UA, sleep_s: float = 0.10):
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
        self.sleep_s = max(0.05, min(0.5, sleep_s))
        # Persistent cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._cache_lock = None
        try:
            self._cache_lock = open(CACHE_FILE + ".lock", "a+")
        except Exception:
            self._cache_lock = None
        self._load_cache()

    def _lock_cache(self):
        if self._cache_lock:
            try:
                fcntl.flock(self._cache_lock, fcntl.LOCK_EX)
            except Exception:
                pass

    def _unlock_cache(self):
        if self._cache_lock:
            try:
                fcntl.flock(self._cache_lock, fcntl.LOCK_UN)
            except Exception:
                pass

    def _load_cache(self):
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._cache = data
        except Exception:
            # Corrupt cache; start fresh
            self._cache = {}

    def _save_cache(self):
        tmp = CACHE_FILE + ".tmp"
        try:
            self._lock_cache()
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
            os.replace(tmp, CACHE_FILE)
        finally:
            self._unlock_cache()

    @staticmethod
    def _norm_title(title: str) -> str:
        return (title or "").strip()

    @staticmethod
    def _cache_key(title: str, snippet: str) -> str:
        base = f"{(title or '').lower()}|{_normalize_text(snippet)}"
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()
        return f"{(title or '').lower()}|{h}"

    @staticmethod
    def _first_rev_id_from_history(hist: Dict[str, Any]) -> Optional[int]:
        items = hist.get("items") or hist.get("revisions") or []
        if items:
            top = items[0]
            return top.get("id")
        return None

    def _page_history(self, title: str) -> Dict[str, Any]:
        slug = urllib.parse.quote((title or "").replace(" ", "_"))
        url = f"{WIKI_REST_BASE}/page/{slug}/history"
        resp = requests.get(url, headers=self.headers, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            if resp.status_code == 404:
                # Page deleted or missing: purge cache entries for this title
                prefix = f"{(title or '').lower()}|"
                self._lock_cache()
                try:
                    self._cache = {k: v for k, v in self._cache.items() if not k.startswith(prefix)}
                    self._save_cache()
                finally:
                    self._unlock_cache()
            raise
        time.sleep(self.sleep_s)
        return resp.json()

    def _revision_rendered_text(self, rev_id: int) -> Optional[str]:
        """Fetch rendered HTML for a revision and strip tags to plain text for matching.
        Uses action=parse with oldid for maximum compatibility with MediaWiki.
        """
        api = "https://oldschool.runescape.wiki/api.php"
        params = {
            "action": "parse",
            "oldid": str(rev_id),
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
        resp = requests.get(api, params=params, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        html_text = (data.get("parse", {}) or {}).get("text")
        if not html_text:
            time.sleep(self.sleep_s)
            return None
        # Strip HTML tags with a conservative regex and unescape entities in _normalize_text
        txt = re.sub(r"<[^>]+>", " ", html_text)
        time.sleep(self.sleep_s)
        return txt

    def _revision_source(self, rev_id: int) -> Optional[str]:
        url = f"{WIKI_REST_BASE}/revision/{rev_id}"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        time.sleep(self.sleep_s)
        # Some REST responses use 'source' for wikitext content
        return data.get("source")

    def _flatten_revisions(self, history_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        # REST returns { items: [ { id, timestamp, user: {id,name} , size, delta, ... }, ... ], ... }
        items = history_json.get("items") or history_json.get("revisions") or []
        return list(items)

    def find_text_contributor(self, title: str, raw_snippet: str, max_checks: int = 12) -> Dict[str, Any]:
        """Find a revision that contains the given snippet and report author info.
        Persistent cache per (title,snippet), invalidated when page top revision changes or is deleted.
        Returns { found, revisionId, author, timestamp, comment, isOriginalAuthor, revision_url, message, checks }
        """
        result: Dict[str, Any] = {"found": False, "message": ""}
        snippet = _normalize_text((raw_snippet or "").strip())
        if not title or not snippet:
            result["message"] = "missing title or snippet"
            return result

        key = self._cache_key(title, snippet)
        try:
            # Always fetch current history head to detect updates/deletions
            hist = self._page_history(title)
            revisions = self._flatten_revisions(hist)
            if not revisions:
                result["message"] = "no revisions"
                return result
            current_top = self._first_rev_id_from_history(hist)

            # Cache hit and still valid?
            cached = self._cache.get(key)
            if cached and cached.get("top_rev_id") == current_top:
                att = cached.get("attestation") or {}
                att.setdefault("checks", 0)
                return att

            # Cache miss or stale -> do targeted search
            checks = 0
            found_idx = None
            found_rev = None

            for i, rev in enumerate(revisions):
                if checks >= max_checks:
                    break
                rev_id = rev.get("id")
                if not rev_id:
                    continue
                matched_on = None
                content = self._revision_source(rev_id)
                if content:
                    checks += 1
                    text = _normalize_text(content)
                    if snippet and snippet in text:
                        matched_on = "wikitext"
                if not matched_on:
                    # Fallback: compare against rendered HTML text
                    rendered = self._revision_rendered_text(rev_id)
                    if rendered:
                        checks += 1
                        rtext = _normalize_text(rendered)
                        if snippet and snippet in rtext:
                            matched_on = "rendered"
                if matched_on:
                    found_idx = i
                    found_rev = dict(rev)
                    found_rev["_matched_on"] = matched_on
                    break

            if not found_rev:
                result.update({"message": f"snippet not found in recent {checks} revisions", "checks": checks})
                # update cache as negative result pinned to current top rev
                self._lock_cache()
                try:
                    self._cache[key] = {
                        "top_rev_id": current_top,
                        "attestation": result,
                        "title": self._norm_title(title),
                        "snippet_sha1": key.split("|")[-1],
                        "updated": datetime.utcnow().isoformat() + "Z",
                    }
                    self._save_cache()
                finally:
                    self._unlock_cache()
                return result

            # Determine if this revision likely introduced the text
            is_original = True
            prev_idx = (found_idx + 1) if (found_idx is not None) else None
            if prev_idx is not None and prev_idx < len(revisions):
                prev_rev = revisions[prev_idx]
                prev_id = prev_rev.get("id")
                if prev_id:
                    prev_content = self._revision_source(prev_id)
                    if prev_content:
                        prev_text = _normalize_text(prev_content)
                        if snippet in prev_text:
                            is_original = False

            author_name = (found_rev.get("user") or {}).get("name") if isinstance(found_rev.get("user"), dict) else found_rev.get("user")
            author_name = author_name or "Anonymous"
            revid = found_rev.get("id")
            ts = found_rev.get("timestamp")
            comment = found_rev.get("comment") or "(No edit summary provided)"

            attestation = {
                "found": True,
                "revisionId": revid,
                "author": author_name,
                "timestamp": ts,
                "comment": comment,
                "isOriginalAuthor": bool(is_original),
                "revision_url": self._build_revision_url(title, revid),
                "checks": checks,
                "matched_on": found_rev.get("_matched_on", "unknown"),
            }

            # Write to cache
            self._lock_cache()
            try:
                self._cache[key] = {
                    "top_rev_id": current_top,
                    "attestation": attestation,
                    "title": self._norm_title(title),
                    "snippet_sha1": key.split("|")[-1],
                    "updated": datetime.utcnow().isoformat() + "Z",
                }
                self._save_cache()
            finally:
                self._unlock_cache()

            return attestation
        except Exception as e:
            result["message"] = f"attribution error: {e}"
            return result

    @staticmethod
    def _build_revision_url(title: str, revid: Optional[int]) -> Optional[str]:
        if not revid:
            return None
        slug = urllib.parse.quote((title or "").replace(" ", "_"))
        # Link to specific oldid view
        return f"https://oldschool.runescape.wiki/w/index.php?title={slug}&oldid={revid}"

