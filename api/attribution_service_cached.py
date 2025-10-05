#!/usr/bin/env python3
"""
Cached Attribution Service
Uses revision cache to minimize API calls while providing accurate attribution.
Incrementally builds cache over time.
"""
import requests
import time
import re
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from revision_cache import RevisionCache

class CachedAttributionService:
    """Attribution service that uses local cache and incrementally fetches from API."""
    
    def __init__(self, cache: RevisionCache = None):
        self.cache = cache or RevisionCache()
        self.base_url = "https://oldschool.runescape.wiki"
        self.rest_api = f"{self.base_url}/rest.php/v1"

        # Import proper User-Agent from config
        from config import get_headers
        self.headers = get_headers()
        # Wikimedia REST API allows 200 requests/second
        # Use conservative limit of 10 concurrent requests to be respectful
        self.max_concurrent_requests = 10
        
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a single API request with proper MediaWiki etiquette.

        Per MediaWiki API:Etiquette:
        - User-Agent with contact info ✓
        - GZip compression ✓
        - maxlag parameter for non-interactive tasks ✓
        - Rate limit: 200 req/sec (we use 10 concurrent max to be conservative)
        """
        if params is None:
            params = {}
        params['maxlag'] = 5  # Prevent running when servers are under load

        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"⚠️  API request failed: {e}")
            return {}

    async def _fetch_revision_content_async(self, session: aiohttp.ClientSession,
                                           semaphore: asyncio.Semaphore,
                                           rev_id: int) -> tuple[int, str]:
        """
        Fetch a single revision's content asynchronously with rate limiting.

        Returns:
            Tuple of (revision_id, wikitext_content)
        """
        async with semaphore:  # Limit concurrent requests
            url = f"{self.rest_api}/revision/{rev_id}"
            params = {'maxlag': 5}

            try:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return (rev_id, data.get("source", ""))
            except Exception as e:
                print(f"⚠️  Failed to fetch revision {rev_id}: {e}")
                return (rev_id, "")

    async def _fetch_multiple_revisions_async(self, rev_ids: List[int]) -> Dict[int, str]:
        """
        Fetch multiple revision contents concurrently with rate limiting.

        Args:
            rev_ids: List of revision IDs to fetch

        Returns:
            Dict mapping revision_id -> wikitext_content
        """
        if not rev_ids:
            return {}

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Create aiohttp session with proper headers
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Create tasks for all revisions
            tasks = [
                self._fetch_revision_content_async(session, semaphore, rev_id)
                for rev_id in rev_ids
            ]

            # Execute all tasks concurrently (but limited by semaphore)
            results = await asyncio.gather(*tasks)

            # Convert to dict
            return dict(results)
    
    def fetch_and_cache_revisions(self, page_title: str, limit: int = 50,
                                   older_than: Optional[int] = None,
                                   newer_than: Optional[int] = None,
                                   fetch_content: bool = True) -> List[Dict]:
        """
        Fetch revisions from API and add to cache.
        Computes diffs to track what actually changed in each revision.

        Args:
            page_title: Page title
            limit: Number of revisions to fetch
            older_than: Fetch revisions older than this ID
            newer_than: Fetch revisions newer than this ID
            fetch_content: Whether to fetch full content (slower) or just metadata (faster)

        Returns:
            List of cached revision entries
        """
        # Fetch revision history (metadata only)
        url = f"{self.rest_api}/page/{page_title}/history"
        params = {"limit": limit}

        if older_than:
            params["older_than"] = older_than
        if newer_than:
            params["newer_than"] = newer_than

        print(f"📡 Fetching {limit} revisions for {page_title} (older_than={older_than}, newer_than={newer_than})")
        data = self._make_request(url, params)
        revisions_metadata = data.get("revisions", [])

        if not revisions_metadata:
            return []

        print(f"   Received {len(revisions_metadata)} revisions")

        # Fetch all content concurrently (much faster than serial)
        # (Revisions come in newest-first order from API)
        revision_contents = {}

        if fetch_content:
            # Collect revision IDs that need content fetching
            rev_ids_to_fetch = []
            for rev_meta in revisions_metadata:
                rev_id = rev_meta.get("id")
                # Skip if already cached with changes data
                existing = self.cache.get_revision(page_title, rev_id)
                if not existing or not existing.get("changes"):
                    rev_ids_to_fetch.append(rev_id)

            if rev_ids_to_fetch:
                print(f"   📄 Fetching content for {len(rev_ids_to_fetch)} revisions concurrently (max {self.max_concurrent_requests} at a time)...")
                # Fetch all revisions concurrently with rate limiting
                revision_contents = asyncio.run(self._fetch_multiple_revisions_async(rev_ids_to_fetch))

        # Now create cache entries with diffs (comparing to next revision in list = previous in time)
        cached_entries = []

        for i, rev_meta in enumerate(revisions_metadata):
            rev_id = rev_meta.get("id")

            # Skip if already cached
            existing = self.cache.get_revision(page_title, rev_id)
            if existing and existing.get("changes"):
                cached_entries.append(existing)
                continue

            # Get this revision's content
            wikitext = revision_contents.get(rev_id)

            # Get previous revision's content (next in list = older in time)
            previous_wikitext = None
            if i + 1 < len(revisions_metadata):
                prev_rev_id = revisions_metadata[i + 1].get("id")
                previous_wikitext = revision_contents.get(prev_rev_id)

            # Create cache entry with diff
            entry = self.cache.create_revision_entry(
                page_title,
                rev_meta,
                wikitext,
                previous_wikitext
            )
            cached_entries.append(entry)

        # Add to cache
        new_entries = [e for e in cached_entries if not self.cache.has_revision(page_title, e["revision_id"])]
        if new_entries:
            print(f"   💾 Caching {len(new_entries)} new revisions with diffs")
            self.cache.add_revisions(new_entries)

        return cached_entries
    
    def ensure_page_cached(self, page_title: str, fetch_all: bool = True) -> int:
        """
        Ensure we have cached revisions for a page.
        - Checks for NEW revisions (updates since last cache)
        - Fetches complete history going back to page creation
        - Continues where it left off (doesn't re-fetch old revisions)

        Args:
            page_title: Page title
            fetch_all: If True, fetches complete history. If False, only checks for updates.

        Returns:
            Number of revisions cached
        """
        cached_revs = self.cache.get_page_revisions(page_title)

        if not cached_revs:
            print(f"🔄 No cache for {page_title}, fetching initial revisions...")
            self.fetch_and_cache_revisions(page_title, limit=50, fetch_content=True)
            cached_revs = self.cache.get_page_revisions(page_title)

        # STEP 1: Check for NEWER revisions (updates since last cache)
        newest_cached_id = self.cache.get_newest_revision_id(page_title)
        if newest_cached_id:
            print(f"🔄 Checking for updates since revision {newest_cached_id}...")
            new_revs = self.fetch_and_cache_revisions(
                page_title,
                limit=50,
                newer_than=newest_cached_id,
                fetch_content=True
            )
            if new_revs:
                print(f"   ✅ Found {len(new_revs)} new revisions!")
            else:
                print(f"   ✅ No new revisions (page is up to date)")

        if not fetch_all:
            final_count = len(self.cache.get_page_revisions(page_title))
            print(f"✅ Cache up to date: {final_count} revisions for {page_title}")
            return final_count

        # STEP 2: Fetch OLDER revisions going back to page creation
        # Get the OLDEST revision we have (not newest!)
        oldest_cached_id = self.cache.get_oldest_revision_id(page_title)
        total_fetched = len(self.cache.get_page_revisions(page_title))

        # Try fetching one batch to see if there's more history
        print(f"🔄 Checking for older revisions (currently have {total_fetched})...")
        test_batch = self.fetch_and_cache_revisions(
            page_title,
            limit=50,
            older_than=oldest_cached_id,
            fetch_content=True
        )

        if not test_batch:
            # No more revisions - we have complete history
            final_count = len(self.cache.get_page_revisions(page_title))
            print(f"✅ Cache COMPLETE: {final_count} total revisions for {page_title}")
            return final_count

        # Update after first batch
        total_fetched = len(self.cache.get_page_revisions(page_title))
        oldest_cached_id = test_batch[-1]["revision_id"]
        print(f"   📊 Progress: {total_fetched} revisions cached (batch 1)")

        # Continue fetching older revisions
        iterations = 1
        max_iterations = 200  # Safety limit (200 batches * 50 = 10,000 revisions max)

        while iterations < max_iterations:
            iterations += 1

            new_revs = self.fetch_and_cache_revisions(
                page_title,
                limit=50,
                older_than=oldest_cached_id,
                fetch_content=True
            )

            if not new_revs:
                print("   ✅ Reached the beginning of page history")
                break

            # Update counts
            total_fetched = len(self.cache.get_page_revisions(page_title))
            oldest_cached_id = new_revs[-1]["revision_id"]

            print(f"   📊 Progress: {total_fetched} revisions cached (batch {iterations})")

        final_count = len(self.cache.get_page_revisions(page_title))
        print(f"✅ Cache COMPLETE: {final_count} total revisions for {page_title}")
        return final_count
    
    def find_field_attribution(self, page_title: str, field: str, current_value: str,
                               fetch_all: bool = True) -> Dict[str, Any]:
        """
        Find attribution for a specific field value.
        Uses cache first, fetches from API only if needed.
        Fetches COMPLETE page history by default.

        Args:
            page_title: Page title
            field: Field name (e.g., "combat", "name", "release", etc.)
            current_value: Current value to attribute (e.g., "725")
            fetch_all: If True, fetches complete history. If False, uses existing cache only.

        Returns:
            Attribution info dict
        """
        print(f"\n🔍 Finding attribution for {page_title} | {field} = {current_value}")
        print("=" * 80)

        # Ensure we have complete cached data
        cached_count = self.ensure_page_cached(page_title, fetch_all=fetch_all)
        print(f"📊 Cache has {cached_count} revisions for {page_title}")
        
        # Search through cached revisions
        changes = self.cache.find_field_changes(page_title, field)
        
        print(f"\n📈 Found {len(changes)} changes to '{field}' field:")
        for i, change in enumerate(changes, 1):
            print(f"   {i}. Rev {change['revision_id']} ({change['timestamp']})")
            print(f"      {change['old_value']} → {change['new_value']}")
            print(f"      By: {change['author']}")
            print(f"      Comment: {change['comment'][:60]}")
        
        # Find the change that set the current value
        for change in reversed(changes):  # Start from most recent
            if change['new_value'] == current_value:
                return {
                    "found": True,
                    "revision_id": change['revision_id'],
                    "timestamp": change['timestamp'],
                    "author": change['author'],
                    "comment": change['comment'],
                    "old_value": change['old_value'],
                    "new_value": change['new_value'],
                    "revisions_checked": cached_count,
                    "source": "cache"
                }
        
        # If not found, the value might have been set at page creation
        if changes and changes[0]['new_value'] == current_value:
            first_change = changes[0]
            return {
                "found": True,
                "revision_id": first_change['revision_id'],
                "timestamp": first_change['timestamp'],
                "author": first_change['author'],
                "comment": first_change['comment'],
                "old_value": None,
                "new_value": first_change['new_value'],
                "revisions_checked": cached_count,
                "source": "cache",
                "note": "Value was set when page was created or in earliest cached revision"
            }
        
        return {
            "found": False,
            "message": f"Could not find when '{field} = {current_value}' was set",
            "revisions_checked": cached_count,
            "source": "cache"
        }
    
    def find_text_attribution(self, page_title: str, search_text: str, fetch_all: bool = True) -> Dict[str, Any]:
        """
        Find attribution for ANY text on the page (not just infobox fields).
        Searches through revision diffs to find when the text was added.

        Args:
            page_title: Page title
            search_text: Text to search for (can be from anywhere on the page)
            fetch_all: If True, fetches complete history

        Returns:
            Attribution info dict
        """
        print(f"\n🔍 Finding attribution for text: '{search_text[:50]}...'")
        print(f"   Page: {page_title}")
        print("=" * 80)

        # Ensure we have complete cached data with diffs
        cached_count = self.ensure_page_cached(page_title, fetch_all=fetch_all)
        print(f"📊 Cache has {cached_count} revisions for {page_title}")

        # Try exact match first
        matches = self.cache.find_text_additions(page_title, search_text)

        # If no exact match and search text looks like it might be a formatted field (e.g., "Combat Level: 725")
        # Try extracting key terms and searching more flexibly
        if not matches and ':' in search_text:
            # Extract the value part (e.g., "725" from "Combat Level: 725")
            parts = search_text.split(':')
            if len(parts) == 2:
                key_term = parts[0].strip().lower()
                value = parts[1].strip()

                # Map common terms to wiki field names
                term_mappings = {
                    'attack speed': ['speed', 'aspeed', 'attack_speed'],
                    'combat level': ['combat', 'level'],
                    'hitpoints': ['hp', 'hitpoints'],
                    'max hit': ['max_hit', 'maxhit'],
                }

                # Get all possible field names for this term
                field_names = []
                for mapping_key, mapping_values in term_mappings.items():
                    if mapping_key in key_term:
                        field_names.extend(mapping_values)
                        break

                # If no mapping found, use the first word
                if not field_names:
                    first_word = key_term.split()[0] if ' ' in key_term else key_term
                    field_names = [first_word]

                # Try searching with all possible field names
                patterns = []
                for field_name in field_names:
                    patterns.extend([
                        f"|{field_name} = {value}",  # Wiki infobox format
                        f"{field_name}={value}",      # Compact format
                        f"{field_name}: {value}",     # Colon format
                    ])

                # Only try bare value as last resort if it's long enough to be specific
                if len(value) > 3:
                    patterns.append(value)

                for pattern in patterns:
                    matches = self.cache.find_text_additions(page_title, pattern)
                    if matches:
                        print(f"   ℹ️  Found using flexible pattern: '{pattern}'")
                        break

        if matches:
            print(f"\n📈 Found {len(matches)} revisions where this text was added:")
            for i, match in enumerate(matches[:10], 1):
                print(f"   {i}. Rev {match['revision_id']} ({match['timestamp']})")
                print(f"      By: {match['author']}")
                print(f"      Section: {match.get('section', 'Unknown')}")
                print(f"      Line {match.get('line_number', '?')}: {match['added_line'][:80]}")
                print(f"      Comment: {match['comment'][:60]}")

            # Return the most recent addition (last in list)
            most_recent = matches[-1]
            return {
                "found": True,
                "revision_id": most_recent['revision_id'],
                "timestamp": most_recent['timestamp'],
                "author": most_recent['author'],
                "comment": most_recent['comment'],
                "snippet": most_recent['added_line'],
                "line_number": most_recent.get('line_number'),
                "section": most_recent.get('section'),
                "context": most_recent.get('context', []),
                "added_count": most_recent['added_count'],
                "removed_count": most_recent['removed_count'],
                "total_matches": len(matches),
                "revisions_checked": cached_count,
                "source": "cache",
                "wiki_url": f"https://oldschool.runescape.wiki/w/Special:Diff/{most_recent['revision_id']}"
            }
        else:
            return {
                "found": False,
                "message": f"Could not find when '{search_text}' was added",
                "revisions_checked": cached_count,
                "source": "cache"
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_cache_stats()

def main():
    """Test the cached attribution service."""
    import sys

    service = CachedAttributionService()

    # Show cache stats
    print("\n📊 CACHE STATISTICS:")
    print("=" * 80)
    stats = service.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Allow command line args for testing different pages/fields
    page_title = sys.argv[1] if len(sys.argv) > 1 else "Zulrah"
    field = sys.argv[2] if len(sys.argv) > 2 else "combat"
    value = sys.argv[3] if len(sys.argv) > 3 else "725"
    fetch_all = sys.argv[4].lower() != "false" if len(sys.argv) > 4 else True

    print(f"\n🎯 Testing: {page_title} | {field} = {value}")
    print(f"   Fetch complete history: {fetch_all}")

    # Test: Find attribution
    result = service.find_field_attribution(
        page_title=page_title,
        field=field,
        current_value=value,
        fetch_all=fetch_all
    )

    print("\n" + "=" * 80)
    print("🎯 ATTRIBUTION RESULT:")
    print("=" * 80)

    if result["found"]:
        print(f"✅ Found attribution!")
        print(f"   Revision: {result['revision_id']}")
        print(f"   Author: {result['author']}")
        print(f"   Date: {result['timestamp']}")
        print(f"   Comment: {result['comment']}")
        print(f"   Change: {result['old_value']} → {result['new_value']}")
        print(f"   Source: {result['source']}")
        if result.get('note'):
            print(f"   Note: {result['note']}")
    else:
        print(f"❌ {result['message']}")

    print(f"\n📈 Revisions checked: {result['revisions_checked']}")

    # Show updated cache stats
    print("\n📊 UPDATED CACHE STATISTICS:")
    print("=" * 80)
    stats = service.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print(f"\n💾 Cache file: {stats['cache_file']}")
    print("   Next query will use cached data (instant!)")

if __name__ == "__main__":
    main()

