#!/usr/bin/env python3
"""
Wiki Revision Cache System
Stores revision metadata and extracted fields in JSONL format.
Enables fast attribution lookups with minimal API calls.
"""
import json
import os
import re
from typing import Dict, List, Optional, Any
from collections import defaultdict
import fcntl
import time

class RevisionCache:
    """Manages a JSONL cache of wiki revisions with extracted field data."""
    
    def __init__(self, cache_file: str = None):
        if cache_file is None:
            # Default to data/cache/wiki_revisions.jsonl
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, "data", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "wiki_revisions.jsonl")
        
        self.cache_file = cache_file
        self._index = None  # Lazy-loaded index
        
    def _load_index(self) -> Dict[str, Dict[int, Dict]]:
        """Load all cached revisions into memory, indexed by page_title -> revision_id -> data."""
        if self._index is not None:
            return self._index
        
        self._index = defaultdict(dict)
        
        if not os.path.exists(self.cache_file):
            return self._index
        
        with open(self.cache_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        rev = json.loads(line)
                        page_title = rev.get("page_title")
                        rev_id = rev.get("revision_id")
                        if page_title and rev_id:
                            self._index[page_title][rev_id] = rev
                    except json.JSONDecodeError:
                        continue
        
        return self._index
    
    def get_page_revisions(self, page_title: str) -> List[Dict]:
        """Get all cached revisions for a page, sorted by timestamp (newest first)."""
        index = self._load_index()
        revisions = list(index.get(page_title, {}).values())
        revisions.sort(key=lambda r: r.get("timestamp", ""), reverse=True)  # Newest first
        return revisions
    
    def get_revision(self, page_title: str, revision_id: int) -> Optional[Dict]:
        """Get a specific cached revision."""
        index = self._load_index()
        return index.get(page_title, {}).get(revision_id)
    
    def has_revision(self, page_title: str, revision_id: int) -> bool:
        """Check if a revision is cached."""
        return self.get_revision(page_title, revision_id) is not None
    
    def add_revision(self, revision_data: Dict):
        """Add a single revision to the cache."""
        self.add_revisions([revision_data])
    
    def add_revisions(self, revisions: List[Dict]):
        """Add multiple revisions to the cache (atomic operation with file locking)."""
        if not revisions:
            return

        # Update in-memory index
        index = self._load_index()
        for rev in revisions:
            page_title = rev.get("page_title")
            rev_id = rev.get("revision_id")
            if page_title and rev_id:
                index[page_title][rev_id] = rev

        # Append to file with file locking
        with open(self.cache_file, 'a') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                for rev in revisions:
                    f.write(json.dumps(rev) + '\n')
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def deduplicate_and_sort(self):
        """
        Deduplicate and sort the cache file.
        Sorts by: page_title (alphabetically), then timestamp (newest first within each page).
        """
        print("🔄 Deduplicating and sorting cache...")

        # Load all entries
        index = self._load_index()

        # Flatten to list and deduplicate
        all_revisions = []
        seen = set()

        for page_title, revisions in index.items():
            for rev_id, rev in revisions.items():
                key = (page_title, rev_id)
                if key not in seen:
                    seen.add(key)
                    all_revisions.append(rev)

        # Sort by page_title (alphabetically), then timestamp (newest first)
        all_revisions.sort(key=lambda r: (r.get("page_title", ""), r.get("timestamp", "")), reverse=False)
        # Within each page, reverse timestamp order (newest first)
        from itertools import groupby
        sorted_revisions = []
        for page_title, group in groupby(all_revisions, key=lambda r: r.get("page_title", "")):
            page_revs = list(group)
            page_revs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)  # Newest first
            sorted_revisions.extend(page_revs)

        # Write back to file
        with open(self.cache_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                for rev in sorted_revisions:
                    f.write(json.dumps(rev) + '\n')
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Clear in-memory index to force reload
        self._index = None

        print(f"✅ Cache sorted: {len(sorted_revisions)} unique entries")

        # Show page counts
        from collections import defaultdict
        page_counts = defaultdict(int)
        for rev in sorted_revisions:
            page_counts[rev.get("page_title")] += 1

        print(f"📄 Pages in cache:")
        for page, count in sorted(page_counts.items()):
            print(f"   {page}: {count} revisions")

        return len(sorted_revisions)
    
    def get_latest_revision_id(self, page_title: str) -> Optional[int]:
        """Get the most recent cached revision ID for a page."""
        revisions = self.get_page_revisions(page_title)
        if revisions:
            return revisions[-1].get("revision_id")
        return None
    
    def get_oldest_revision_id(self, page_title: str) -> Optional[int]:
        """Get the oldest cached revision ID for a page."""
        revisions = self.get_page_revisions(page_title)
        if revisions:
            return revisions[-1].get("revision_id")  # Last in list = oldest
        return None

    def get_newest_revision_id(self, page_title: str) -> Optional[int]:
        """Get the newest (most recent) cached revision ID for a page."""
        revisions = self.get_page_revisions(page_title)
        if revisions:
            return revisions[0].get("revision_id")  # First in list = newest
        return None
    
    def extract_infobox_fields(self, wikitext: str) -> Dict[str, str]:
        """
        Extract ALL infobox fields from wikitext generically.
        Does not limit to specific fields - captures everything.
        """
        fields = {}

        # Extract ALL fields from infobox templates (|field = value)
        # This pattern matches any field in wiki infobox format
        pattern = r'\|([a-zA-Z0-9_]+)\s*=\s*([^\n|]+)'
        matches = re.finditer(pattern, wikitext)

        for match in matches:
            field_name = match.group(1).strip().lower()
            value = match.group(2).strip()

            # Clean up common wiki markup
            value = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', value)  # [[link|text]] -> text
            value = re.sub(r"'{2,}", '', value)  # Remove bold/italic
            value = re.sub(r'<[^>]+>', '', value)  # Remove HTML tags
            value = value.strip()

            # Only store non-empty values
            if value:
                fields[field_name] = value

        return fields
    
    def create_revision_entry(self, page_title: str, revision_metadata: Dict, wikitext: Optional[str] = None,
                             previous_wikitext: Optional[str] = None) -> Dict:
        """
        Create a cache entry from revision metadata and optional wikitext.
        Stores COMPLETE revision details including what changed.

        Args:
            page_title: Page title
            revision_metadata: Revision metadata from API
            wikitext: Full wikitext content of this revision
            previous_wikitext: Full wikitext content of previous revision (for diff)
        """
        entry = {
            "page_title": page_title,
            "revision_id": revision_metadata.get("id"),
            "timestamp": revision_metadata.get("timestamp"),
            "author": revision_metadata.get("user", {}).get("name"),
            "author_id": revision_metadata.get("user", {}).get("id"),
            "comment": revision_metadata.get("comment", ""),
            "size": revision_metadata.get("size"),
            "minor": revision_metadata.get("minor", False),
            "delta": revision_metadata.get("delta", 0),
        }

        if wikitext:
            # Extract infobox fields for quick lookup
            entry["fields"] = self.extract_infobox_fields(wikitext)
            entry["content_length"] = len(wikitext)

            # Compute diff if we have previous content
            if previous_wikitext:
                entry["changes"] = self.compute_diff(previous_wikitext, wikitext)
            else:
                entry["changes"] = {"added_lines": [], "removed_lines": [], "modified_sections": []}
        else:
            entry["fields"] = {}
            entry["content_length"] = 0
            entry["changes"] = {"added_lines": [], "removed_lines": [], "modified_sections": []}

        return entry

    def compute_diff(self, old_text: str, new_text: str) -> Dict[str, Any]:
        """
        Compute what changed between two versions of wikitext.
        Returns added/removed lines with line numbers, sections, and context.
        """
        import difflib

        old_lines = old_text.split('\n')
        new_lines = new_text.split('\n')

        # Track current section as we go through the file
        def get_section(lines, line_num):
            """Find the most recent section header before this line."""
            if not lines or line_num < 0:
                return "Top of page"
            for i in range(min(line_num - 1, len(lines) - 1), -1, -1):
                if i >= len(lines):
                    continue
                line = lines[i].strip()
                # Wiki section headers: ==Section==, ===Subsection===, etc.
                if line.startswith('==') and line.endswith('=='):
                    return line.strip('=').strip()
            return "Top of page"

        # Use difflib to compute line-by-line diff with context
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm='', n=2))

        added_changes = []
        removed_changes = []
        current_line_num = 0

        for line in diff:
            if line.startswith('@@'):
                # Parse line numbers from unified diff header: @@ -old_start,old_count +new_start,new_count @@
                import re
                match = re.search(r'@@ -(\d+),?\d* \+(\d+),?\d* @@', line)
                if match:
                    current_line_num = int(match.group(2))
            elif line.startswith('+') and not line.startswith('+++'):
                # Added line
                content = line[1:].strip()
                if content:  # Skip empty lines
                    section = get_section(new_lines, current_line_num)
                    # Get context (2 lines before and after)
                    context_start = max(0, current_line_num - 2)
                    context_end = min(len(new_lines), current_line_num + 3)
                    context = new_lines[context_start:context_end]

                    added_changes.append({
                        "line": content,
                        "line_number": current_line_num,
                        "section": section,
                        "context": [l.strip() for l in context if l.strip()][:5]  # Max 5 context lines
                    })
                current_line_num += 1
            elif line.startswith('-') and not line.startswith('---'):
                # Removed line
                content = line[1:].strip()
                if content:
                    section = get_section(old_lines, current_line_num)
                    removed_changes.append({
                        "line": content,
                        "section": section
                    })
            elif not line.startswith('---') and not line.startswith('+++'):
                # Context line (unchanged)
                current_line_num += 1

        return {
            "added_changes": added_changes[:100],  # Limit to first 100 for storage
            "removed_changes": removed_changes[:100],
            "added_count": len(added_changes),
            "removed_count": len(removed_changes),
            # Keep old format for compatibility
            "added_lines": [c["line"] for c in added_changes[:100]],
            "removed_lines": [c["line"] for c in removed_changes[:100]]
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        index = self._load_index()
        
        total_revisions = sum(len(revs) for revs in index.values())
        pages_cached = len(index)
        
        # Count revisions with extracted fields
        revisions_with_fields = 0
        for page_revs in index.values():
            for rev in page_revs.values():
                if rev.get("fields"):
                    revisions_with_fields += 1
        
        return {
            "total_revisions": total_revisions,
            "pages_cached": pages_cached,
            "revisions_with_fields": revisions_with_fields,
            "cache_file": self.cache_file,
            "file_size_mb": os.path.getsize(self.cache_file) / (1024 * 1024) if os.path.exists(self.cache_file) else 0
        }
    
    def search_by_field_value(self, page_title: str, field: str, value: str) -> List[Dict]:
        """Find all revisions where a field has a specific value."""
        revisions = self.get_page_revisions(page_title)
        matches = []
        
        for rev in revisions:
            fields = rev.get("fields", {})
            if fields.get(field.lower()) == value:
                matches.append(rev)
        
        return matches
    
    def find_field_changes(self, page_title: str, field: str) -> List[Dict[str, Any]]:
        """Find all revisions where a field value changed."""
        revisions = self.get_page_revisions(page_title)
        changes = []
        prev_value = None

        for rev in revisions:
            fields = rev.get("fields", {})
            current_value = fields.get(field.lower())

            if current_value != prev_value:
                changes.append({
                    "revision_id": rev.get("revision_id"),
                    "timestamp": rev.get("timestamp"),
                    "author": rev.get("author"),
                    "comment": rev.get("comment"),
                    "old_value": prev_value,
                    "new_value": current_value
                })
                prev_value = current_value

        return changes

    def find_text_additions(self, page_title: str, search_text: str) -> List[Dict[str, Any]]:
        """
        Find all revisions where specific text was added to the page.
        Searches through the 'changes' data to find when text appeared.
        Includes line numbers, sections, and context.

        Args:
            page_title: Page title
            search_text: Text to search for (case-insensitive)

        Returns:
            List of revisions where the text was added with location info
        """
        revisions = self.get_page_revisions(page_title)
        matches = []
        search_lower = search_text.lower()

        for rev in revisions:
            changes = rev.get("changes", {})
            added_changes = changes.get("added_changes", [])

            # Check if search text appears in any added change
            for change in added_changes:
                line = change.get("line", "")
                if search_lower in line.lower():
                    matches.append({
                        "revision_id": rev.get("revision_id"),
                        "timestamp": rev.get("timestamp"),
                        "author": rev.get("author"),
                        "comment": rev.get("comment"),
                        "added_line": line,
                        "line_number": change.get("line_number"),
                        "section": change.get("section"),
                        "context": change.get("context", []),
                        "added_count": changes.get("added_count", 0),
                        "removed_count": changes.get("removed_count", 0)
                    })
                    break  # Only add this revision once

        return matches
    
    def clear_cache(self):
        """Clear the entire cache."""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        self._index = None
    
    def clear_page_cache(self, page_title: str):
        """Clear cache for a specific page (requires rebuilding the file)."""
        index = self._load_index()
        if page_title in index:
            del index[page_title]
        
        # Rebuild the file
        with open(self.cache_file, 'w') as f:
            for page_revs in index.values():
                for rev in page_revs.values():
                    f.write(json.dumps(rev) + '\n')

