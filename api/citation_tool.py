#!/usr/bin/env python3
"""
Citation Tool for OSRS AI System

Automatically generates proper citations from structured wiki data.
The AI should call this tool instead of trying to format citations manually.

This tool:
1. Takes a wiki page title and field/text reference
2. Looks up the exact content from osrs_wiki_content.jsonl
3. Finds contributors from attribution cache
4. Returns properly formatted citation with attribution
"""

import json
import os
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CitationTool:
    """Tool for generating citations from structured wiki data"""
    
    def __init__(self, wiki_content_path: str = None, attribution_service=None):
        """
        Initialize citation tool
        
        Args:
            wiki_content_path: Path to osrs_wiki_content.jsonl
            attribution_service: Attribution service instance
        """
        if wiki_content_path is None:
            # Default to project root data directory
            api_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(api_dir)
            wiki_content_path = os.path.join(project_root, "data", "osrs_wiki_content.jsonl")
        
        self.wiki_content_path = wiki_content_path
        self.attribution_service = attribution_service
        
        # Load wiki content into memory for fast lookup
        self.wiki_pages = {}
        self._load_wiki_content()
    
    def _load_wiki_content(self):
        """Load parsed wiki content into memory"""
        if not os.path.exists(self.wiki_content_path):
            logger.warning(f"Wiki content file not found: {self.wiki_content_path}")
            return
        
        logger.info(f"Loading wiki content from {self.wiki_content_path}")
        count = 0
        with open(self.wiki_content_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    page = json.loads(line.strip())
                    title = page.get('title')
                    if title:
                        self.wiki_pages[title] = page
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"✅ Loaded {count} wiki pages for citation lookup")
    
    def get_page_content(self, page_title: str) -> Optional[Dict[str, Any]]:
        """Get parsed content for a wiki page"""
        return self.wiki_pages.get(page_title)
    
    def find_field_value(self, page_title: str, field_name: str) -> Optional[str]:
        """
        Find a specific field value from a wiki page

        Args:
            page_title: Wiki page title
            field_name: Field name (e.g., "combat", "hitpoints", "max_hit")

        Returns:
            Field value as string, or None if not found
        """
        page = self.get_page_content(page_title)
        if not page:
            return None

        # Check infobox fields first (most structured)
        infobox = page.get('infobox', {})
        if field_name in infobox:
            value = infobox[field_name]
            # Clean up the value
            if isinstance(value, (list, dict)):
                value = str(value)
            return str(value).strip()

        # Check for common field name variations
        field_variations = [
            field_name,
            field_name.lower(),
            field_name.replace('_', ' '),
            field_name.replace(' ', '_'),
            field_name.title()
        ]

        for variation in field_variations:
            if variation in infobox:
                value = infobox[variation]
                if isinstance(value, (list, dict)):
                    value = str(value)
                return str(value).strip()

        # Check top-level fields
        if field_name in page:
            value = page[field_name]
            if isinstance(value, (list, dict)):
                value = str(value)
            return str(value).strip()

        return None
    
    def create_citation(
        self,
        page_title: str,
        field_or_text: str,
        paraphrased_text: str,
        include_attribution: bool = True
    ) -> Dict[str, Any]:
        """
        Create a properly formatted citation
        
        Args:
            page_title: Wiki page title (source)
            field_or_text: Field name (e.g., "combat") or exact text snippet
            paraphrased_text: The text that will appear in the response
            include_attribution: Whether to look up contributors
        
        Returns:
            Citation dict with:
            - formatted: The formatted citation string
            - source_title: Page title
            - source_text: Exact text from wiki
            - paraphrased: The paraphrased text
            - attribution: Contributor info (if available)
        """
        page = self.get_page_content(page_title)
        if not page:
            return {
                'error': f'Page not found: {page_title}',
                'formatted': paraphrased_text  # Return unformatted if page not found
            }
        
        # Try to find the exact source text
        source_text = None

        # First, try as a field name
        field_value = self.find_field_value(page_title, field_or_text)
        if field_value:
            # Use just the field value, not the field name
            # This makes citations cleaner: "725" instead of "combat: 725"
            source_text = field_value
        else:
            # Try to find the text in the page content
            text_content = page.get('text', '')
            if field_or_text.lower() in text_content.lower():
                # Find the exact snippet (case-insensitive search, but preserve original case)
                import re
                pattern = re.compile(re.escape(field_or_text), re.IGNORECASE)
                match = pattern.search(text_content)
                if match:
                    # Get a clean sentence or phrase around the match
                    # Look for sentence boundaries
                    start = match.start()
                    end = match.end()

                    # Expand to sentence boundaries (period, newline, or start/end of text)
                    while start > 0 and text_content[start-1] not in '.\\n':
                        start -= 1
                        if match.start() - start > 100:  # Max 100 chars before
                            break

                    while end < len(text_content) and text_content[end] not in '.\\n':
                        end += 1
                        if end - match.end() > 100:  # Max 100 chars after
                            break

                    source_text = text_content[start:end].strip()
                    # Clean up any leading/trailing punctuation
                    source_text = source_text.strip('.\\n ')

        # If still not found, use the field_or_text as-is
        if not source_text:
            source_text = field_or_text
        
        # Format the citation
        formatted = f'[CITE:source="{page_title}"|text="{source_text}"]{paraphrased_text}[/CITE]'
        
        result = {
            'formatted': formatted,
            'source_title': page_title,
            'source_text': source_text,
            'paraphrased': paraphrased_text,
            'wiki_url': f"https://oldschool.runescape.wiki/w/{page_title.replace(' ', '_')}"
        }
        
        # Look up attribution if requested
        if include_attribution and self.attribution_service:
            try:
                attribution = self.attribution_service.find_attribution(
                    page_title=page_title,
                    snippet=source_text
                )
                if attribution.get('found'):
                    result['attribution'] = {
                        'author': attribution.get('author'),
                        'timestamp': attribution.get('timestamp'),
                        'revision_id': attribution.get('revision_id'),
                        'wiki_url': attribution.get('wiki_url')
                    }
            except Exception as e:
                logger.warning(f"Failed to get attribution: {e}")
        
        return result
    
    def create_multi_citation(
        self,
        citations: List[Dict[str, str]]
    ) -> str:
        """
        Create multiple citations at once
        
        Args:
            citations: List of dicts with:
                - page_title: Wiki page title
                - field_or_text: Field name or text snippet
                - paraphrased_text: Text to display
        
        Returns:
            Full text with all citations formatted
        """
        result_parts = []
        
        for cit in citations:
            citation = self.create_citation(
                page_title=cit['page_title'],
                field_or_text=cit['field_or_text'],
                paraphrased_text=cit['paraphrased_text'],
                include_attribution=False  # Skip attribution for batch
            )
            result_parts.append(citation['formatted'])
        
        return ' '.join(result_parts)


def create_citation_for_ai(
    page_title: str,
    field_or_text: str,
    your_text: str
) -> str:
    """
    Tool function that AI can call to create citations
    
    Args:
        page_title: The wiki page you're citing (e.g., "Zulrah", "Abyssal whip")
        field_or_text: The field name (e.g., "combat") or exact text you're referencing
        your_text: Your paraphrased version of the information
    
    Returns:
        Properly formatted citation string that you should include in your response
    
    Example:
        AI wants to say "Zulrah has 725 combat level"
        Call: create_citation_for_ai("Zulrah", "combat", "Zulrah has 725 combat level")
        Returns: "[CITE:source=\"Zulrah\"|text=\"combat: 725\"]Zulrah has 725 combat level[/CITE]"
    """
    tool = CitationTool()
    result = tool.create_citation(
        page_title=page_title,
        field_or_text=field_or_text,
        paraphrased_text=your_text,
        include_attribution=False  # Attribution is added later by the API
    )
    
    if 'error' in result:
        # If page not found, return unformatted text
        logger.warning(f"Citation error: {result['error']}")
        return your_text
    
    return result['formatted']


# Singleton instance
_citation_tool = None

def get_citation_tool(attribution_service=None) -> CitationTool:
    """Get or create the global citation tool instance"""
    global _citation_tool
    if _citation_tool is None:
        _citation_tool = CitationTool(attribution_service=attribution_service)
    return _citation_tool


if __name__ == "__main__":
    # Test the citation tool
    tool = CitationTool()
    
    print("=" * 80)
    print("Testing Citation Tool")
    print("=" * 80)
    
    # Test 1: Field citation
    print("\n1. Field citation (Zulrah combat level):")
    result = tool.create_citation(
        page_title="Zulrah",
        field_or_text="combat",
        paraphrased_text="Zulrah has a combat level of 725"
    )
    print(f"   Formatted: {result['formatted']}")
    print(f"   Source: {result['source_text']}")
    
    # Test 2: Text snippet citation
    print("\n2. Text snippet citation:")
    result = tool.create_citation(
        page_title="Abyssal whip",
        field_or_text="slash attack bonus",
        paraphrased_text="The abyssal whip provides excellent slash bonuses"
    )
    print(f"   Formatted: {result['formatted']}")
    print(f"   Source: {result['source_text']}")
    
    # Test 3: Using the AI function
    print("\n3. AI function test:")
    formatted = create_citation_for_ai(
        page_title="Dragon scimitar",
        field_or_text="attack",
        your_text="The dragon scimitar requires 60 Attack to wield"
    )
    print(f"   Result: {formatted}")

