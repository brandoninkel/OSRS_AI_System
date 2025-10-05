#!/usr/bin/env python3
"""
Wiki Attribution Service
Provides attribution for wiki content with location tracking and caching.

This is a wrapper around the cached attribution service that maintains
backward compatibility with the existing API.
"""
from attribution_service_cached import CachedAttributionService
from typing import Dict, Any, Optional


class WikiAttributionService:
    """
    Wrapper around CachedAttributionService for backward compatibility.
    Provides attribution for wiki content with location tracking.
    """
    def __init__(self):
        self.service = CachedAttributionService()
    
    def find_attribution(self, page_title: str, snippet: str) -> Dict[str, Any]:
        """
        Find attribution for a text snippet from a wiki page.
        
        Args:
            page_title: Wiki page title (e.g., "Zulrah")
            snippet: Text snippet to find attribution for
        
        Returns:
            Attribution dict with:
            - found: bool
            - author: str (contributor name)
            - timestamp: str (ISO format)
            - revision_id: int
            - comment: str (edit comment)
            - snippet: str (exact text that was added)
            - section: str (page section where it was added)
            - line_number: int (line number in the page)
            - context: list (surrounding lines)
            - wiki_url: str (direct link to revision)
        """
        # Use the cached attribution service to find when this text was added
        result = self.service.find_text_attribution(
            page_title=page_title,
            search_text=snippet,
            fetch_all=True  # Fetch complete history for accurate attribution
        )
        
        return result


def main():
    """Test the attribution service."""
    service = WikiAttributionService()
    
    # Test 1: Find attribution for shark drop table text
    print("=" * 80)
    print("Test 1: Find attribution for 'shark drop table'")
    print("=" * 80)
    
    result = service.find_attribution("Zulrah", "shark drop table")
    
    if result["found"]:
        print(f"\n✅ FOUND:")
        print(f"   Author: {result['author']}")
        print(f"   Date: {result['timestamp']}")
        print(f"   Section: {result.get('section', 'Unknown')}")
        print(f"   Line: {result.get('line_number', '?')}")
        print(f"   Snippet: {result['snippet'][:100]}")
        print(f"   URL: {result.get('wiki_url', 'N/A')}")
    else:
        print(f"\n❌ NOT FOUND: {result.get('message', 'Unknown error')}")
    
    # Test 2: Find attribution for combat level
    print("\n" + "=" * 80)
    print("Test 2: Find attribution for 'combat = 725'")
    print("=" * 80)
    
    result2 = service.find_attribution("Zulrah", "combat = 725")
    
    if result2["found"]:
        print(f"\n✅ FOUND:")
        print(f"   Author: {result2['author']}")
        print(f"   Date: {result2['timestamp']}")
        print(f"   Section: {result2.get('section', 'Unknown')}")
        print(f"   Snippet: {result2['snippet']}")
    else:
        print(f"\n❌ NOT FOUND: {result2.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()

