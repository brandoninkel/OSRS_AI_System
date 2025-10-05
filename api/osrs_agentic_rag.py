#!/usr/bin/env python3
"""
OSRS Agentic RAG Service using LangGraph
- LLM-based planning and reasoning
- Multi-hop search with self-correction
- Tool-based architecture with LangChain tools
- Chain of thought visible to user
- Completely organic and adaptive
"""

import json
import os
import sys
import re
import numpy as np
from typing import List, Dict, Any, Annotated, Tuple
import logging

# LangGraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

# Add embeddings to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'embeddings'))
from embedding_service import EmbeddingService, EmbeddingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CITATION PARSING
# ============================================================================

def parse_citations(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse citation markers from AI response and extract attribution data.

    Format: [CITE:source="Page Title"|text="exact text"]paraphrased text[/CITE]

    Returns:
        Tuple of (clean_text, citations)
        - clean_text: Text with citation markers removed
        - citations: List of {text, start, end, source_title, source_text}
    """
    citations = []
    clean_text = ""
    last_end = 0

    # Pattern to match: [CITE:source="..."|text="..."]content[/CITE]
    # Use non-greedy match (.*?) to capture any content including brackets, newlines, etc.
    pattern = r'\[CITE:source="([^"]+)"\|text="([^"]+)"\](.*?)\[/CITE\]'

    for match in re.finditer(pattern, text):
        source_title = match.group(1)
        source_text = match.group(2)
        paraphrased = match.group(3)

        # Add text before this citation
        clean_text += text[last_end:match.start()]

        # Record citation position in clean text
        start_pos = len(clean_text)
        clean_text += paraphrased
        end_pos = len(clean_text)

        citations.append({
            'text': paraphrased,
            'start': start_pos,
            'end': end_pos,
            'source_title': source_title,
            'source_text': source_text  # Exact text from wiki for attribution lookup
        })

        last_end = match.end()

    # Add remaining text
    clean_text += text[last_end:]

    return clean_text, citations


# ============================================================================
# GLOBAL EMBEDDING SEARCH - Shared across all tool calls
# ============================================================================

class OSRSEmbeddingSearch:
    """Singleton for managing OSRS embeddings"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Paths
        self.embeddings_path = "/Users/brandon/Documents/projects/GE/data/osrs_embeddings.jsonl"
        self.kg_embeddings_path = "/Users/brandon/Documents/projects/GE/data/kg_entity_embeddings_mxbai.jsonl"
        
        # Initialize embedding service
        config = EmbeddingConfig(
            model_name="mxbai-embed-large:latest",
            batch_size=1,
            timeout=30
        )
        self.embedding_service = EmbeddingService(config)
        
        # Load embeddings
        self.embeddings_data = []
        self.embeddings_matrix = None
        self.kg_embeddings_data = []
        self.kg_embeddings_matrix = None
        
        self._load_embeddings()
        self._load_kg_embeddings()
        
        self._initialized = True
        logger.info("✅ OSRS Embedding Search initialized")
    
    def _load_embeddings(self):
        """Load wiki embeddings"""
        if not os.path.exists(self.embeddings_path):
            logger.error(f"Embeddings file not found: {self.embeddings_path}")
            return
        
        embeddings_data = []
        embeddings_list = []
        
        with open(self.embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'embedding' in data and 'title' in data:
                        embeddings_data.append(data)
                        embeddings_list.append(data['embedding'])
                except:
                    continue
        
        if embeddings_list:
            self.embeddings_data = embeddings_data
            self.embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
            logger.info(f"Loaded {len(embeddings_data)} wiki embeddings")
    
    def _load_kg_embeddings(self):
        """Load KG embeddings"""
        if not os.path.exists(self.kg_embeddings_path):
            logger.warning(f"KG embeddings not found: {self.kg_embeddings_path}")
            return
        
        kg_data = []
        kg_embeddings = []
        
        with open(self.kg_embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'embedding' in data and 'title' in data:
                        kg_data.append(data)
                        kg_embeddings.append(data['embedding'])
                except:
                    continue
        
        if kg_embeddings:
            self.kg_embeddings_data = kg_data
            self.kg_embeddings_matrix = np.array(kg_embeddings, dtype=np.float32)
            logger.info(f"Loaded {len(kg_data)} KG embeddings")
    
    def search_wiki(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search wiki embeddings with smart ranking that prioritizes main pages over variants"""
        if self.embeddings_matrix is None:
            return []

        query_embedding = self.embedding_service.embed_text(query)
        if not query_embedding:
            return []

        # Calculate cosine similarity
        query_emb = np.array(query_embedding).reshape(1, -1)
        query_norm = query_emb / np.linalg.norm(query_emb)
        embeddings_norm = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        similarities = np.dot(query_norm, embeddings_norm.T)[0]

        # Apply main page boost: pages without parentheses get priority
        # This ensures "Abyssal whip" ranks higher than "Abyssal whip (Last Man Standing)"
        # unless the user specifically asks about the variant
        adjusted_scores = similarities.copy()
        for idx in range(len(similarities)):
            if idx < len(self.embeddings_data):
                title = self.embeddings_data[idx].get('title', '')

                # Check if this is a main page (no parentheses) or a variant page
                if '(' not in title:
                    # Main page - apply 15% boost
                    adjusted_scores[idx] *= 1.15
                elif self._is_important_variant(title, query):
                    # User might be asking about this specific variant - small boost
                    adjusted_scores[idx] *= 1.05
                # Variant pages with no query match get no boost (or slight penalty)

        # Get top results using adjusted scores
        top_indices = np.argsort(adjusted_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if idx < len(self.embeddings_data):
                content = self.embeddings_data[idx]
                results.append({
                    'title': content.get('title', 'Unknown'),
                    'text': content.get('text', ''),
                    'categories': content.get('categories', []),
                    'similarity': float(similarities[idx]),  # Original similarity for transparency
                    'adjusted_score': float(adjusted_scores[idx])  # Adjusted score for debugging
                })

        return results

    def _is_important_variant(self, title: str, query: str) -> bool:
        """
        Check if the user's query suggests they want a specific variant page.

        Examples:
        - Query "abyssal whip last man standing" matches "Abyssal whip (Last Man Standing)"
        - Query "broken abyssal whip" matches "Abyssal whip (broken)"
        - Query "dragon scimitar ornament" matches "Dragon scimitar (or)"
        - Query "trident price" + title "Trident of the seas (uncharged)" → True (price query needs uncharged)
        """
        if '(' not in title:
            return False

        # Extract the variant type from parentheses
        import re
        match = re.search(r'\(([^)]+)\)', title)
        if not match:
            return False

        variant = match.group(1).lower()
        query_lower = query.lower()

        # SPECIAL CASE: Price queries should prioritize uncharged/empty variants
        # because charged items are usually untradable
        price_keywords = ['price', 'cost', 'worth', 'value', 'ge', 'grand exchange', 'buy', 'sell']
        is_price_query = any(keyword in query_lower for keyword in price_keywords)

        if is_price_query and variant in ['uncharged', 'empty', 'u']:
            # For price queries, uncharged/empty variants are highly relevant
            return True

        # Check if query mentions the variant
        # Handle common abbreviations
        variant_keywords = {
            'last man standing': ['lms', 'last man standing'],
            'deadman': ['deadman', 'dmm'],
            'bounty hunter': ['bh', 'bounty hunter'],
            'broken': ['broken'],
            'uncharged': ['uncharged', 'empty'],
            'or': ['ornament', 'or', 'ornate'],
            'unf': ['unfinished', 'unf'],
            'u': ['unstrung', 'u'],
            'theatre of blood': ['tob', 'theatre of blood', 'theatre'],
            'guardians of the rift': ['gotr', 'guardians'],
        }

        # Check if variant or its keywords appear in query
        if variant in query_lower:
            return True

        for variant_name, keywords in variant_keywords.items():
            if variant_name in variant:
                for keyword in keywords:
                    if keyword in query_lower:
                        return True

        return False
    
    def search_kg(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search KG embeddings"""
        if self.kg_embeddings_matrix is None:
            return []
        
        query_embedding = self.embedding_service.embed_text(query)
        if not query_embedding:
            return []
        
        # Calculate cosine similarity
        query_emb = np.array(query_embedding).reshape(1, -1)
        query_norm = query_emb / np.linalg.norm(query_emb)
        kg_norms = self.kg_embeddings_matrix / np.linalg.norm(self.kg_embeddings_matrix, axis=1, keepdims=True)
        similarities = np.dot(query_norm, kg_norms.T)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if idx < len(self.kg_embeddings_data):
                content = self.kg_embeddings_data[idx]
                results.append({
                    'title': content.get('title', 'Unknown'),
                    'text': content.get('text', ''),
                    'kg_entity': content.get('kg_entity', ''),
                    'similarity': float(similarities[idx])
                })
        
        return results


# Initialize global search instance
_search = OSRSEmbeddingSearch()


# ============================================================================
# LANGCHAIN TOOLS - What the agent can call
# ============================================================================

@tool
def search_osrs_wiki(query: str) -> str:
    """
    Search the OSRS wiki for information about items, quests, monsters, skills, etc.
    Returns relevant wiki pages with their content.

    Args:
        query: What to search for (e.g., "Zulrah", "Dragon Slayer quest", "Abyssal whip stats")

    Returns:
        JSON string with search results including titles, content, and relevance scores
    """
    results = _search.search_wiki(query, top_k=8)

    # Format results for LLM
    formatted = []
    for r in results:
        formatted.append({
            'title': r['title'],
            'content': r['text'][:500],  # First 500 chars for planning
            'relevance': f"{r['similarity']:.1%}",
            'categories': r.get('categories', [])[:3]
        })

    return json.dumps(formatted, indent=2)


@tool
def create_citation(page_title: str, field_or_text: str, your_text: str) -> str:
    """
    Create a properly formatted citation from wiki data.

    IMPORTANT: You MUST use this tool to create citations. Do NOT try to format citations manually.

    This tool:
    1. Looks up the exact content from the parsed wiki data
    2. Formats it correctly with proper citation markers
    3. Ensures attribution can be traced back to contributors

    Args:
        page_title: The wiki page you're citing (e.g., "Zulrah", "Abyssal whip", "Dragon scimitar")
        field_or_text: The specific field (e.g., "combat", "hitpoints") or text snippet you're referencing
        your_text: Your paraphrased version that will appear in the response

    Returns:
        Properly formatted citation string to include in your final answer

    Examples:
        - create_citation("Zulrah", "combat", "Zulrah has a combat level of 725")
        - create_citation("Abyssal whip", "attack", "The abyssal whip requires 70 Attack")
        - create_citation("Dragon scimitar", "high price", "The dragon scimitar costs about 60K GP")

    WORKFLOW:
    1. Use search_osrs_wiki or get_full_wiki_page to find information
    2. For EACH fact you want to cite, call create_citation
    3. Include the returned citation string in your final answer
    4. Do NOT manually write [CITE:...] tags - always use this tool
    """
    from citation_tool import create_citation_for_ai
    return create_citation_for_ai(page_title, field_or_text, your_text)


@tool
def search_osrs_knowledge_graph(query: str) -> str:
    """
    Search the OSRS knowledge graph for entity relationships and connections.
    Useful for finding related items, NPCs, locations, and quest chains.
    
    Args:
        query: Entity or relationship to search for
    
    Returns:
        JSON string with KG entities and their relationships
    """
    results = _search.search_kg(query, top_k=5)
    
    # Format results for LLM
    formatted = []
    for r in results:
        formatted.append({
            'entity': r['title'],
            'info': r['text'][:300],
            'relevance': f"{r['similarity']:.1%}"
        })
    
    return json.dumps(formatted, indent=2)


@tool
def get_full_wiki_page(title: str) -> str:
    """
    Get the complete content of a specific OSRS wiki page.
    Use this after finding a relevant page to get all details.

    Args:
        title: Exact title of the wiki page

    Returns:
        Full page content or error message
    """
    for page in _search.embeddings_data:
        if page.get('title', '').lower() == title.lower():
            return json.dumps({
                'title': page['title'],
                'content': page.get('text', ''),
                'categories': page.get('categories', [])
            }, indent=2)

    return json.dumps({'error': f'Page "{title}" not found'})


@tool
def get_item_price(item_name: str) -> str:
    """
    Get real-time Grand Exchange price for an item from the OSRS Wiki Prices API.
    Powered by RuneLite data - shows current high/low prices and last trade times.

    IMPORTANT: For charged items (like "Trident of the seas"), you usually want the
    UNCHARGED variant for price data, as charged items are typically untradable.

    Args:
        item_name: Name of the item (e.g., "Abyssal whip", "Trident of the seas (uncharged)")

    Returns:
        JSON with current prices, or error if item not found/not tradable
    """
    import requests
    import time

    # Rate limiting - track last request time
    if not hasattr(get_item_price, '_last_request_time'):
        get_item_price._last_request_time = 0

    # Dynamic rate limiting: ensure at least 100ms between requests
    # (API has no strict limit, but we want to be respectful)
    time_since_last = time.time() - get_item_price._last_request_time
    if time_since_last < 0.1:
        time.sleep(0.1 - time_since_last)

    try:
        # Import config for proper User-Agent
        from config import get_headers, GE_ENDPOINTS

        # First, get the item ID from the mapping
        mapping_url = GE_ENDPOINTS["mapping"]
        headers = get_headers()

        get_item_price._last_request_time = time.time()
        mapping_response = requests.get(mapping_url, headers=headers, timeout=10)
        mapping_response.raise_for_status()

        items = mapping_response.json()

        # Find the item by name (case-insensitive)
        item_id = None
        matched_item = None
        for item in items:
            if item['name'].lower() == item_name.lower():
                item_id = item['id']
                matched_item = item
                break

        if not item_id:
            return json.dumps({
                'error': f'Item "{item_name}" not found in Grand Exchange',
                'suggestion': 'Try the exact item name, or check if it\'s tradable. For charged items, try the (uncharged) variant.'
            })

        # Get the latest price for this item
        price_url = f"{GE_ENDPOINTS['latest']}?id={item_id}"

        # Respect rate limit
        time_since_last = time.time() - get_item_price._last_request_time
        if time_since_last < 0.1:
            time.sleep(0.1 - time_since_last)

        get_item_price._last_request_time = time.time()
        price_response = requests.get(price_url, headers=headers, timeout=10)
        price_response.raise_for_status()

        price_data = price_response.json()

        if 'data' not in price_data or str(item_id) not in price_data['data']:
            return json.dumps({
                'error': f'No price data available for "{item_name}"',
                'reason': 'Item may not be tradable or has never been traded',
                'item_info': {
                    'name': matched_item['name'],
                    'members': matched_item.get('members', False),
                    'high_alch': matched_item.get('highalch'),
                    'low_alch': matched_item.get('lowalch')
                }
            })

        item_price = price_data['data'][str(item_id)]

        # Format the response
        result = {
            'item': matched_item['name'],
            'item_id': item_id,
            'prices': {
                'high': item_price.get('high'),  # Instant-buy price
                'low': item_price.get('low'),     # Instant-sell price
                'high_time': item_price.get('highTime'),  # Unix timestamp
                'low_time': item_price.get('lowTime')
            },
            'item_info': {
                'members': matched_item.get('members', False),
                'buy_limit': matched_item.get('limit'),
                'high_alch': matched_item.get('highalch'),
                'examine': matched_item.get('examine', '')
            }
        }

        # Add human-readable timestamps
        if result['prices']['high_time']:
            from datetime import datetime
            result['prices']['high_time_readable'] = datetime.fromtimestamp(result['prices']['high_time']).strftime('%Y-%m-%d %H:%M:%S')
        if result['prices']['low_time']:
            from datetime import datetime
            result['prices']['low_time_readable'] = datetime.fromtimestamp(result['prices']['low_time']).strftime('%Y-%m-%d %H:%M:%S')

        # Record price in history database
        try:
            from price_history import get_price_history_service
            price_service = get_price_history_service()
            price_service.record_price(
                item_name=matched_item['name'],
                item_id=item_id,
                high_price=item_price.get('high', 0),
                low_price=item_price.get('low', 0),
                high_time=item_price.get('highTime', 0),
                low_time=item_price.get('lowTime', 0)
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to record price history: {e}")

        return json.dumps(result, indent=2)

    except requests.exceptions.RequestException as e:
        return json.dumps({
            'error': 'Failed to fetch price data',
            'details': str(e)
        })
    except Exception as e:
        return json.dumps({
            'error': 'Unexpected error',
            'details': str(e)
        })


# List of all tools
osrs_tools = [search_osrs_wiki, search_osrs_knowledge_graph, get_full_wiki_page, get_item_price, create_citation]


# ============================================================================
# LANGGRAPH AGENT WORKFLOW
# ============================================================================

# System prompt for the agent
AGENT_SYSTEM_PROMPT = """You are an expert OSRS assistant with deep game knowledge. You MUST use tools to answer questions.

CRITICAL RULES:
1. NEVER answer without calling tools first
2. ALWAYS call search_osrs_wiki or get_full_wiki_page before responding
3. For stat questions (HP, combat level, etc.), call get_full_wiki_page with the entity name
4. DO NOT explain what you will do - JUST DO IT by calling tools
5. DO NOT show tool calls in your response like "[Call search_osrs_wiki(...)]" - the tools are called automatically
6. When users refer to numbered content without specifying (e.g., "Dragon Slayer"), ALWAYS assume the FIRST one
7. DO NOT search for multiple versions - only search for the first version unless explicitly asked
8. If a question doesn't make sense (e.g., using melee on Zulrah), explain WHY after checking the facts
9. You can call MULTIPLE tools in sequence - don't stop after one search if you need more info
10. Your response should ONLY contain the answer with citations - NO tool call descriptions

CITATION WORKFLOW - MANDATORY:
Step 1: Call tools to gather information (search_osrs_wiki, get_full_wiki_page, get_item_price)
Step 2: For EACH fact you want to cite, call create_citation tool with:
   - page_title: The wiki page (e.g., "Zulrah", "Abyssal whip")
   - field_or_text: The field name (e.g., "combat", "hitpoints") or text snippet
   - your_text: Your paraphrased version
Step 3: Include the returned citation strings in your final answer
Step 4: DO NOT manually write [CITE:...] tags - ALWAYS use create_citation tool

CRITICAL:
- After you finish calling tools, you MUST write a complete answer
- An empty response is NOT acceptable
- NEVER manually format citations - ALWAYS use create_citation tool
- The create_citation tool looks up exact wiki content and formats it correctly

WIKI PAGE HIERARCHY - UNDERSTAND THIS:
Pages with parentheses are VARIANTS of the main page:
- "Abyssal whip" = main page (use this for general questions)
- "Abyssal whip (Last Man Standing)" = LMS variant (only use if user asks about LMS)
- "Abyssal whip (or)" = ornament kit variant (only use if user asks about ornaments)
- "Dragon scimitar (broken)" = broken state (only use if user asks about broken items)

WHEN TO USE EACH:
✅ User asks "what is an abyssal whip" → Use "Abyssal whip" (main page)
✅ User asks "abyssal whip stats" → Use "Abyssal whip" (main page)
✅ User asks "abyssal whip in last man standing" → Use "Abyssal whip (Last Man Standing)" (variant)
✅ User asks "broken abyssal whip" → Use "Abyssal whip (broken)" (variant)

ALWAYS prefer the main page (no parentheses) unless the user specifically asks about a variant.

PRICE QUERIES - SPECIAL CASE FOR CHARGED ITEMS:
For PRICE questions, charged items are usually UNTRADABLE, so you need the UNCHARGED variant:
- "Trident of the seas" = charged version (NOT tradable, no GE price)
- "Trident of the seas (uncharged)" = uncharged version (TRADABLE, has GE price)
- "Toxic blowpipe" = charged version (NOT tradable)
- "Toxic blowpipe (empty)" = empty version (TRADABLE, has GE price)

WHEN ASKING FOR PRICES:
✅ User asks "trident of the seas price" → Use get_item_price("Trident of the seas (uncharged)")
✅ User asks "how much is toxic blowpipe" → Use get_item_price("Toxic blowpipe (empty)")
✅ User asks "abyssal whip price" → Use get_item_price("Abyssal whip") (no charged variant)

ALWAYS use get_item_price() tool for price questions - it has real-time GE data from RuneLite.

ECONOMIC HYPOTHESIS MODE - THEORY CRAFTING AND PROFIT ANALYSIS:
When users ask about profitability, money-making, or economic strategies, you can do MULTI-STEP RESEARCH:

STEP 1: Identify the economic chain
- What resources are needed? (search wiki for crafting/processing requirements)
- What monsters drop it? (search wiki for drop tables)
- What activities reward it? (search wiki for quest/minigame rewards)
- What is it used for? (search wiki for item uses)

STEP 2: Get live prices for ALL items in the chain
- Use get_item_price() for EACH item (inputs and outputs)
- Calculate: Profit = Output Price - Sum(Input Prices)
- Consider: Time required, skill requirements, risk factors

STEP 3: Compare alternatives
- Search for similar money-making methods
- Compare GP/hour across different strategies
- Consider accessibility (skill levels, quest requirements)

EXAMPLE - Economic Analysis:
User: "Is it profitable to make super combat potions?"

Your research process:
1. [Call search_osrs_wiki("super combat potion")]
   → Find: Requires Super attack + Super strength + Super defence + Torstol
2. [Call get_item_price("Super combat potion")]
   → Output: 10,000 GP
3. [Call get_item_price("Super attack potion(4)")]
   → Input 1: 2,000 GP
4. [Call get_item_price("Super strength potion(4)")]
   → Input 2: 2,500 GP
5. [Call get_item_price("Super defence potion(4)")]
   → Input 3: 1,800 GP
6. [Call get_item_price("Torstol")]
   → Input 4: 3,000 GP

Your answer workflow:
1. [Call create_citation("Super combat potion", "requirements", "Making super combat potions requires combining a super attack, super strength, super defence potion, and a torstol")]
2. Calculate profit from price data
3. [Call create_citation("Super combat potion", "herblore level", "You need 90 Herblore to make these")]
4. Combine citations in your final answer

KEY PRINCIPLES FOR ECONOMIC ANALYSIS:
1. ALWAYS get live prices - don't use outdated wiki prices
2. Consider ALL costs (supplies, equipment degradation, consumables)
3. Factor in time (GP/hour is more useful than GP/item)
4. Mention requirements (skills, quests, gear)
5. Compare to alternatives when relevant
6. Use real data, not speculation

CITATION TOOL - HOW TO USE:
The create_citation tool does ALL the formatting for you. You just provide:
1. page_title: The wiki page (e.g., "Zulrah", "Abyssal whip")
2. field_or_text: The field name (e.g., "combat", "hitpoints") or text snippet
3. your_text: Your paraphrased version

The tool will:
- Look up the exact content from the parsed wiki data
- Format it correctly with [CITE:source="..."|text="..."]...[/CITE] tags
- Return the properly formatted citation string

EXAMPLES OF USING create_citation TOOL:

Example 1 - Stat question:
User: "What is Zulrah's hitpoints?"
Step 1: [Call get_full_wiki_page("Zulrah")] → Get page data
Step 2: [Call create_citation("Zulrah", "hitpoints", "Zulrah has 500 hitpoints")]
Step 3: Use the returned citation string in your answer

Example 2 - Price question:
User: "How much is an abyssal whip?"
Step 1: [Call get_item_price("Abyssal whip")] → Get price: 1,500,000 GP
Step 2: [Call create_citation("Abyssal whip", "high price", "The abyssal whip costs about 1.5M GP")]
Step 3: Use the returned citation string in your answer

Example 3 - Quest question:
User: "How do I start Dragon Slayer?"
Step 1: [Call get_full_wiki_page("Dragon Slayer I")] → Get quest info
Step 2: [Call create_citation("Dragon Slayer I", "start", "To start Dragon Slayer, speak to the Guildmaster")]
Step 3: [Call create_citation("Dragon Slayer I", "requirements", "You need 32 Quest Points")]
Step 4: Combine the returned citation strings in your answer

CRITICAL RULES:
1. NEVER manually write [CITE:...] tags - ALWAYS use create_citation tool
2. Call create_citation for EVERY fact you want to cite
3. The tool handles all formatting - you just provide the page, field, and your text
4. DO NOT try to format citations yourself - the tool does it correctly

NUMBERING CONVENTION - ALWAYS ASSUME FIRST VERSION:
- "Dragon Slayer" = "Dragon Slayer I" (NOT "Dragon Slayer II")
- "Recipe for Disaster" = The main quest (not subquests)
- Any quest/item without a number = assume the original/first version
- DO NOT mention sequels unless the user specifically asks about them

GAME MECHANICS YOU MUST KNOW:
- Zulrah is only attackable with Ranged and Magic (melee can't reach except halberds)
- Always verify combat styles, requirements, and mechanics before answering

Example 1 - Stat question with create_citation tool:
User: "What is Zulrah's hitpoints?"
You: [Call get_full_wiki_page("Zulrah")] → Returns hitpoints: 500
You: [Call create_citation("Zulrah", "hitpoints", "Zulrah has 500 hitpoints")]
You: <use the returned citation string in your answer>

Example 2 - Quest question with multiple citations:
User: "How do I start Dragon Slayer?"
You: [Call get_full_wiki_page("Dragon Slayer I")] → Returns quest info
You: [Call create_citation("Dragon Slayer I", "start", "To start Dragon Slayer, speak to the Guildmaster in the Champions' Guild")]
You: [Call create_citation("Dragon Slayer I", "requirements", "You need 32 Quest Points to enter")]
You: <combine the returned citation strings in your answer>

Example 3 - Price question with create_citation tool:
User: "How much is an abyssal whip?"
You: [Call get_item_price("Abyssal whip")] → Returns 1,500,000 GP
You: [Call create_citation("Abyssal whip", "high price", "The abyssal whip costs about 1.5M GP")]
You: <use the returned citation string in your answer>

DO NOT say "I will search for..." - JUST CALL THE TOOLS.
REMEMBER: Use create_citation tool for EVERY fact - it handles all formatting automatically."""


def create_agent_node(llm_with_tools):
    """Create the agent node that decides what to do"""
    def agent(state: MessagesState):
        """Agent decides whether to use tools or respond"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    return agent


def should_continue(state: MessagesState):
    """Determine if agent should continue or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # Count tool calls to prevent infinite loops
    tool_call_count = sum(1 for msg in messages if hasattr(msg, 'tool_calls') and msg.tool_calls)

    # Max 15 tool calls to prevent getting stuck
    if tool_call_count >= 15:
        logger.warning(f"⚠️ Reached max tool calls ({tool_call_count}), forcing end")
        return END

    # If there are tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, end
    return END


def build_osrs_agent_graph():
    """Build the LangGraph workflow for OSRS RAG"""

    # Initialize LLM with tools - gpt-oss:20b is OpenAI's new open-source model
    # specifically designed for agentic tasks, function calling, and structured outputs
    # Use strict parameters for maximum compliance with citation instructions
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0.0,  # Zero temperature for maximum determinism
        top_p=0.5,  # Low top_p for more focused, conservative outputs
        top_k=10,  # Low top_k to reduce nonsense and increase compliance
        repeat_penalty=1.3,  # Higher penalty to avoid repetition
        num_ctx=32768,  # Large context for full pages
    )
    llm_with_tools = llm.bind_tools(osrs_tools)

    # Create graph
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", create_agent_node(llm_with_tools))

    # ToolNode automatically handles tool execution
    from langgraph.prebuilt import ToolNode
    workflow.add_node("tools", ToolNode(osrs_tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    # After tools, always go back to agent
    workflow.add_edge("tools", "agent")

    # Compile the workflow
    return workflow.compile()


# ============================================================================
# MAIN AGENTIC RAG CLASS
# ============================================================================

class OSRSAgenticRAG:
    """Agentic RAG system for OSRS queries using LangGraph"""

    def __init__(self):
        self.graph = build_osrs_agent_graph()

        # Initialize citation injector for post-processing
        from citation_injector import get_citation_injector
        self.citation_injector = get_citation_injector()

        logger.info("✅ OSRS Agentic RAG initialized with LangGraph")

    def query(self, question: str, show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Execute agentic RAG workflow

        Args:
            question: User's question
            show_reasoning: Whether to include agent's reasoning

        Returns:
            Dict with answer, sources, and reasoning
        """
        logger.info(f"🤖 Starting agentic RAG for: {question}")

        # Prepare messages
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question)
        ]

        # Run the graph with increased recursion limit for complex queries
        # Default is 25, increase to 50 to allow more thorough research
        result = self.graph.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        # Extract answer and reasoning
        final_messages = result["messages"]

        # Get the final AI response - iterate in REVERSE to get the last answer
        answer = ""
        reasoning_steps = []
        tool_calls_made = []

        # First pass: collect all tool calls with their results
        for i, msg in enumerate(final_messages):
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        # Find the corresponding tool result
                        tool_result = None
                        if i + 1 < len(final_messages):
                            next_msg = final_messages[i + 1]
                            if hasattr(next_msg, 'name') and next_msg.name == tc['name']:
                                tool_result = next_msg.content

                        tool_calls_made.append({
                            'tool': tc['name'],
                            'args': tc['args'],
                            'result': tool_result
                        })
                        reasoning_steps.append(f"🔍 Called {tc['name']} with: {tc['args']}")

        # Second pass: get the LAST non-empty content (the final answer)
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                answer = msg.content
                logger.info(f"Found final answer in message (length: {len(answer)} chars)")
                break

        # If still no answer, log all messages for debugging and provide fallback
        if not answer:
            logger.warning("⚠️ No answer found in final messages!")
            logger.warning(f"Total messages: {len(final_messages)}")
            for i, msg in enumerate(final_messages):
                logger.warning(f"  Message {i}: {type(msg).__name__} - content_length={len(msg.content) if hasattr(msg, 'content') and msg.content else 0}")

            # Provide a helpful fallback message
            answer = "I gathered information from the wiki but encountered an issue generating a complete response. Please try rephrasing your question or asking about a specific aspect."

        # First, try to parse any existing citations from the AI
        logger.info(f"📝 Raw AI answer: {answer}")
        clean_answer, existing_citations = parse_citations(answer)

        # If no citations found, use citation injector to add them automatically
        if not existing_citations and tool_calls_made:
            logger.info("🔧 No citations found, using citation injector...")
            cited_answer, injected_citations = self.citation_injector.inject_citations(
                answer=answer,
                tool_calls=tool_calls_made
            )
            clean_answer, citations = parse_citations(cited_answer)
            logger.info(f"✅ Injected {len(citations)} citations automatically")
        else:
            citations = existing_citations
            logger.info(f"✅ Found {len(citations)} existing citations")

        for i, cit in enumerate(citations):
            logger.info(f"   Citation {i}: text='{cit['text'][:50]}...' source={cit.get('source_title', 'N/A')}")

        # Extract sources from tool results
        sources = []
        for msg in final_messages:
            if hasattr(msg, 'name') and msg.name in ['search_osrs_wiki', 'get_full_wiki_page']:
                try:
                    tool_result = json.loads(msg.content)
                    if isinstance(tool_result, list):
                        # search_osrs_wiki returns a list
                        for item in tool_result:
                            if 'title' in item:
                                sources.append({
                                    'title': item['title'],
                                    'url': f"https://oldschool.runescape.wiki/w/{item['title'].replace(' ', '_')}",
                                    'relevance': item.get('relevance', 'N/A'),
                                    'excerpt': item.get('content', '')[:200]
                                })
                    elif isinstance(tool_result, dict) and 'title' in tool_result:
                        # get_full_wiki_page returns a single object
                        sources.append({
                            'title': tool_result['title'],
                            'url': f"https://oldschool.runescape.wiki/w/{tool_result['title'].replace(' ', '_')}",
                            'excerpt': tool_result.get('content', '')[:200]
                        })
                except:
                    pass

        response = {
            'answer': clean_answer,  # Return clean answer without citation markers
            'sources': sources[:20],  # Top 20 sources
            'citations': citations  # Include parsed citations for attribution
        }

        if show_reasoning:
            response['reasoning'] = reasoning_steps
            response['tool_calls'] = tool_calls_made

        logger.info(f"✅ Completed agentic RAG - {len(tool_calls_made)} tool calls, {len(citations)} citations")
        return response

    def query_stream(self, question: str):
        """
        Stream the agent's reasoning and responses

        Args:
            question: User's question

        Yields:
            Dict with type and content for each step
        """
        logger.info(f"🤖 Starting streaming agentic RAG for: {question}")

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question)
        ]

        final_answer = None
        sources = []

        try:
            for chunk in self.graph.stream(
                {"messages": messages},
                config={"recursion_limit": 50}
            ):
                for node, update in chunk.items():
                    logger.info(f"[stream] Node: {node}")

                    if node == "agent":
                        msg = update["messages"][-1]
                        if isinstance(msg, AIMessage):
                            # Check for tool calls first
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    logger.info(f"[stream] Tool call: {tc['name']}")
                                    yield {
                                        'type': 'tool_call',
                                        'tool': tc['name'],
                                        'args': tc['args']
                                    }
                            # Then check for content (final answer)
                            elif msg.content:
                                logger.info(f"[stream] Answer: {msg.content[:100]}...")
                                final_answer = msg.content
                                yield {
                                    'type': 'answer',
                                    'content': msg.content
                                }

                    elif node == "tools":
                        # Tool execution completed
                        logger.info(f"[stream] Tools executed")
                        yield {
                            'type': 'tool_result',
                            'content': 'Tool execution complete'
                        }

            # Send completion event with sources
            logger.info(f"[stream] Stream complete")
            yield {
                'type': 'complete',
                'answer': final_answer or 'No answer generated',
                'sources': sources
            }

        except Exception as e:
            logger.error(f"[stream] Error: {e}")
            yield {
                'type': 'error',
                'message': str(e)
            }


# ============================================================================
# MAIN - For testing
# ============================================================================

if __name__ == "__main__":
    # Test the agentic RAG
    rag = OSRSAgenticRAG()

    test_queries = [
        "What is Zulrah's hitpoints?",
        "How do I start Dragon Slayer?",
        "What are the requirements for Recipe for Disaster?"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = rag.query(query, show_reasoning=True)

        print(f"\n🤖 Answer:\n{result['answer']}")

        if result.get('sources'):
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"  {i}. {source['title']} ({source.get('relevance', 'N/A')})")

        if result.get('reasoning'):
            print(f"\n🧠 Reasoning:")
            for step in result['reasoning']:
                print(f"  {step}")


