#!/usr/bin/env python3
"""
Citation Injector - Post-processing system to inject proper citations

This system automatically injects citations into AI responses by:
1. Analyzing tool calls to identify facts
2. Parsing the AI's answer into sentences
3. Matching sentences to facts from tool calls
4. Using the citation tool to generate proper citations
5. Replacing sentences with cited versions

This is more reliable than asking the AI to format citations manually.
"""

import json
import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class CitationInjector:
    """Automatically inject citations into AI responses"""
    
    def __init__(self, citation_tool=None):
        """
        Initialize citation injector
        
        Args:
            citation_tool: CitationTool instance for generating citations
        """
        self.citation_tool = citation_tool
        if citation_tool is None:
            from citation_tool import get_citation_tool
            self.citation_tool = get_citation_tool()
    
    def extract_facts_from_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract facts from tool call results
        
        Args:
            tool_calls: List of tool calls with results
        
        Returns:
            List of facts with page_title, field, value, and context
        """
        facts = []
        
        for tc in tool_calls:
            tool_name = tc.get('tool')
            args = tc.get('args', {})
            result = tc.get('result')
            
            if not result:
                continue
            
            # Parse get_full_wiki_page results
            if tool_name == 'get_full_wiki_page':
                try:
                    page_data = json.loads(result) if isinstance(result, str) else result
                    if 'title' in page_data and 'content' in page_data:
                        # Extract key facts from the page
                        title = page_data['title']
                        content = page_data['content']
                        
                        # Look for common patterns in content
                        # Combat level, hitpoints, etc.
                        patterns = {
                            'combat': r'Combat Level:\s*(\d+)',
                            'hitpoints': r'Hitpoints:\s*(\d+)',
                            'attack': r'Attack:\s*(\d+)',
                            'strength': r'Strength:\s*(\d+)',
                            'defence': r'Defence:\s*(\d+)',
                            'magic': r'Magic:\s*(\d+)',
                            'ranged': r'Ranged:\s*(\d+)',
                        }
                        
                        for field, pattern in patterns.items():
                            match = re.search(pattern, content, re.IGNORECASE)
                            if match:
                                facts.append({
                                    'page_title': title,
                                    'field': field,
                                    'value': match.group(1),
                                    'source': 'wiki_page',
                                    'context': match.group(0)
                                })
                except:
                    pass
            
            # Parse get_item_price results
            elif tool_name == 'get_item_price':
                try:
                    price_data = json.loads(result) if isinstance(result, str) else result
                    if 'item' in price_data and 'prices' in price_data:
                        item_name = price_data['item']
                        prices = price_data['prices']
                        
                        if prices.get('high'):
                            facts.append({
                                'page_title': item_name,
                                'field': 'price',
                                'value': prices['high'],
                                'source': 'price_api',
                                'context': f"Grand Exchange price: {prices['high']} GP"
                            })
                except:
                    pass
        
        return facts
    
    def parse_sentences(self, text: str) -> List[str]:
        """
        Parse text into sentences
        
        Args:
            text: Text to parse
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, question marks, exclamation marks
        # But preserve decimal numbers
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def find_matching_facts(self, sentence: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find facts that match a sentence

        Args:
            sentence: Sentence to match
            facts: List of facts

        Returns:
            List of matching facts
        """
        matches = []
        sentence_lower = sentence.lower()

        for fact in facts:
            page_title = fact['page_title'].lower()
            field = fact['field'].lower()
            value = str(fact['value']).lower()

            # For price facts, be more flexible with matching
            if fact['field'] == 'price':
                # Check if sentence mentions the item and price-related keywords
                price_keywords = ['cost', 'price', 'worth', 'gp', 'gold', 'sell', 'buy']
                if page_title in sentence_lower and any(kw in sentence_lower for kw in price_keywords):
                    matches.append(fact)
                    continue

            # Check if sentence mentions the page and the value
            if page_title in sentence_lower and value in sentence_lower:
                matches.append(fact)
            # Or if it mentions the field and value
            elif field in sentence_lower and value in sentence_lower:
                matches.append(fact)
            # Or if it mentions the page and the field
            elif page_title in sentence_lower and field in sentence_lower:
                matches.append(fact)

        return matches
    
    def inject_citations(
        self,
        answer: str,
        tool_calls: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Inject citations into an answer based on tool calls
        
        Args:
            answer: AI's answer text
            tool_calls: List of tool calls with results
        
        Returns:
            Tuple of (cited_answer, citations_list)
        """
        # Extract facts from tool calls
        facts = self.extract_facts_from_tools(tool_calls)
        
        if not facts:
            logger.info("No facts extracted from tool calls, returning answer as-is")
            return answer, []
        
        logger.info(f"Extracted {len(facts)} facts from tool calls")
        
        # Parse answer into sentences
        sentences = self.parse_sentences(answer)
        
        # Process each sentence
        cited_sentences = []
        all_citations = []
        
        for sentence in sentences:
            # Find matching facts
            matching_facts = self.find_matching_facts(sentence, facts)
            
            if matching_facts:
                # Use the first matching fact to generate citation
                fact = matching_facts[0]
                
                try:
                    # Generate citation using the citation tool
                    citation_result = self.citation_tool.create_citation(
                        page_title=fact['page_title'],
                        field_or_text=fact['field'],
                        paraphrased_text=sentence,
                        include_attribution=False  # Attribution added later by API
                    )
                    
                    if 'formatted' in citation_result:
                        cited_sentences.append(citation_result['formatted'])
                        
                        # Track citation for attribution
                        all_citations.append({
                            'text': sentence,
                            'source_title': fact['page_title'],
                            'source_text': citation_result.get('source_text', fact['context']),
                            'start': len(' '.join(cited_sentences[:-1])) + (1 if cited_sentences[:-1] else 0),
                            'end': len(' '.join(cited_sentences))
                        })
                        
                        logger.info(f"✅ Injected citation for: {sentence[:50]}...")
                    else:
                        # Citation failed, use original sentence
                        cited_sentences.append(sentence)
                except Exception as e:
                    logger.warning(f"Failed to generate citation: {e}")
                    cited_sentences.append(sentence)
            else:
                # No matching facts, keep sentence as-is
                cited_sentences.append(sentence)
        
        # Combine sentences
        cited_answer = ' '.join(cited_sentences)
        
        logger.info(f"✅ Citation injection complete: {len(all_citations)} citations added")
        
        return cited_answer, all_citations


# Singleton instance
_citation_injector = None

def get_citation_injector() -> CitationInjector:
    """Get or create the global citation injector instance"""
    global _citation_injector
    if _citation_injector is None:
        _citation_injector = CitationInjector()
    return _citation_injector


if __name__ == "__main__":
    # Test the citation injector
    injector = CitationInjector()
    
    # Simulate tool calls
    tool_calls = [
        {
            'tool': 'get_full_wiki_page',
            'args': {'title': 'Zulrah'},
            'result': json.dumps({
                'title': 'Zulrah',
                'content': 'Combat Level: 725\nHitpoints: 500\nZulrah is a snake boss...'
            })
        }
    ]
    
    # Test answer
    answer = "Zulrah has a combat level of 725. It has 500 hitpoints."
    
    print("=" * 80)
    print("Testing Citation Injector")
    print("=" * 80)
    print(f"\nOriginal answer:\n{answer}")
    
    cited_answer, citations = injector.inject_citations(answer, tool_calls)
    
    print(f"\nCited answer:\n{cited_answer}")
    print(f"\nCitations: {len(citations)}")
    for i, cit in enumerate(citations):
        print(f"  {i+1}. {cit['text'][:50]}... -> {cit['source_title']}")

