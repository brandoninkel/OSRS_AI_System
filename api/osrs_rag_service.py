#!/usr/bin/env python3
"""
OSRS RAG Service - Query OSRS Knowledge with Embeddings + LLaMA 3.1
Combines embeddings for retrieval with LLaMA 3.1 for generation
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import logging
from datetime import datetime, timezone
import requests
import threading
import signal
import fcntl
import csv

# Add the embeddings to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'embeddings'))
from embedding_service import EmbeddingService, EmbeddingConfig
from attribution_service import WikiAttributionService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OSRSRAGService:
    def __init__(self):
        # Paths
        self.embeddings_path = "/Users/brandon/Documents/projects/GE/OSRS_AI_SYSTEM/data/osrs_embeddings.jsonl"

        # Initialize embedding service for queries
        config = EmbeddingConfig(
            model_name="mxbai-embed-large:latest",
            batch_size=1,
            timeout=30
        )
        self.embedding_service = EmbeddingService(config)

        # LLaMA 3.1 configuration
        self.llama_url = "http://localhost:11434"
        self.llama_model = "llama3.1:8b"

        # Chat session management - each chat has its own context
        self.chat_sessions = {}  # chat_id -> {'conversation_history': [], 'entity_context': {}}
        self.default_chat_id = "default"

        # Concurrency + reload coordination
        self._reload_lock = threading.RLock()
        self._reload_event = threading.Event()

        # PID file for signal-based reload notifications from the embedding writer
        self.pid_file = "/Users/brandon/Documents/projects/GE/OSRS_AI_SYSTEM/data/rag_service.pid"
        try:
            os.makedirs(os.path.dirname(self.pid_file), exist_ok=True)
            with open(self.pid_file, 'w') as pf:
                pf.write(str(os.getpid()))
            logger.info(f"Wrote RAG PID to {self.pid_file}")
        except Exception as e:
            logger.warning(f"Could not write PID file: {e}")

        # Register SIGUSR1 handler for on-change reloads
        try:
            signal.signal(signal.SIGUSR1, self._handle_sigusr1)
            logger.info("Registered SIGUSR1 handler for embeddings reload")
        except Exception as e:
            logger.warning(f"Could not register SIGUSR1 handler: {e}")

        # Load embeddings into memory
        self.embeddings_data = []
        self.embeddings_matrix = None
        # Optional reranker (BGE cross-encoder) — initialize asynchronously so API can start immediately
        self.reranker = None
        if os.getenv('OSRS_USE_RERANKER', '1') == '1':
            reranker_model = os.getenv('OSRS_RERANKER_MODEL', 'BAAI/bge-reranker-large')
            # First run may download the model; do it in background
            threading.Thread(target=self._init_reranker_async, args=(reranker_model,), daemon=True).start()

        # Initialize wiki attribution helper
        self.attribution = WikiAttributionService()

        # Spelling/term lexicon built from titles (no hardcoded topic data)
        self._title_token_freq = {}
        self._title_token_set = set()
        self._spellcorr_enabled = os.getenv('OSRS_SPELLCORR', '1') == '1'

        self.load_embeddings()

        # Optional: load lightweight knowledge graph adjacency if available
        try:
            self._load_kg()
        except Exception as e:
            logger.warning(f"KG load skipped: {e}")

        # Optional: load KG embeddings if trained artifacts exist
        try:
            self._load_kg_embeddings()
        except Exception as e:
            logger.warning(f"KG embeddings load skipped: {e}")

    def load_embeddings(self):
        """Load all embeddings into memory for fast similarity search with read lock + partial-line safety"""
        logger.info(f"Loading embeddings from: {self.embeddings_path}")

        if not os.path.exists(self.embeddings_path):
            logger.warning(f"Embeddings file not found: {self.embeddings_path}")
            with self._reload_lock:
                self.embeddings_data = []
                self.embeddings_matrix = np.zeros((0, 0), dtype=float)
            return

        new_data: List[Dict[str, Any]] = []
        embeddings_list: List[List[float]] = []

        with open(self.embeddings_path, 'r', encoding='utf-8') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH)
            except Exception:
                pass
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    emb = data.get('embedding')
                    if emb is None:
                        continue
                    new_data.append(data)
                    embeddings_list.append(emb)
                except json.JSONDecodeError:
                    # Skip a potentially half-written trailing line during concurrent appends
                    continue
                except Exception:
                    continue

        if embeddings_list:
            try:
                matrix = np.array(embeddings_list)
                dim = matrix.shape[1] if matrix.ndim == 2 else len(embeddings_list[0])
            except Exception:
                matrix = np.array(embeddings_list)
                dim = matrix.shape[1] if matrix.ndim == 2 else 0
        else:
            matrix = np.zeros((0, 0), dtype=float)
            dim = 0

        # Prepare normalized matrix for fast cosine similarity and annotate row indices
        if matrix.size > 0:
            try:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                norm_matrix = matrix / norms
            except Exception:
                norm_matrix = matrix
        else:
            norm_matrix = matrix

        with self._reload_lock:
            for i, d in enumerate(new_data):
                try:
                    d['_row'] = i
                except Exception:
                    pass
            self.embeddings_data = new_data
            self.embeddings_matrix = matrix
            self.embeddings_norm = norm_matrix
            # Build title indices for fast lookups
            try:
                self._title_to_doc = {}
                self._title_to_row = {}
                for i, d in enumerate(new_data):
                    t = (d.get('title') or '').strip()
                    if not t:
                        continue
                    if t not in self._title_to_doc:
                        self._title_to_doc[t] = d
                        self._title_to_row[t] = i
            except Exception:
                self._title_to_doc = {}
                self._title_to_row = {}

        # Build/update lightweight title token lexicon for spelling correction (outside lock)
        try:
            tok_freq = {}
            tok_set = set()
            def _tok(text: str):
                import re
                return [t for t in re.split(r"[^A-Za-z0-9]+", (text or "").lower()) if t]
            for d in new_data:
                title = d.get('title') or ''
                for t in _tok(title):
                    tok_set.add(t)
                    tok_freq[t] = tok_freq.get(t, 0) + 1
            with self._reload_lock:
                self._title_token_freq = tok_freq
                self._title_token_set = tok_set
            logger.info(f"Title lexicon size: {len(tok_set)} tokens")
        except Exception as e:
            logger.warning(f"Failed building title lexicon: {e}")

        logger.info(f"Loaded {len(new_data)} embeddings ({dim}D)")

    def _load_kg(self):
        """Load lightweight KG adjacency if osrs_kg_triples.csv exists next to embeddings.
        Builds:
          - self.kg_links: Dict[title, Set[neighbor_title]] from links_to edges
          - self.kg_categories: Dict[title, Set[category]] from is_a Category: edges
        """
        try:
            data_dir = os.path.dirname(self.embeddings_path)
            kg_path = os.path.join(data_dir, 'osrs_kg_triples.csv')
            self.kg_links = {}
            self.kg_categories = {}
            if not os.path.exists(kg_path):
                logger.info(f"KG triples not found (optional): {kg_path}")
                return
            with open(kg_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = (row.get('head') or '').strip()
                    r = (row.get('relation') or '').strip()
                    t = (row.get('tail') or '').strip()
                    if not h or not r or not t:
                        continue
                    if r == 'links_to':
                        s = self.kg_links.get(h)
                        if s is None:
                            s = set()
                            self.kg_links[h] = s
                        s.add(t)
                    elif r == 'is_a' and t.startswith('Category:'):
                        cat = t[len('Category:'):]
                        s2 = self.kg_categories.get(h)
                        if s2 is None:
                            s2 = set()
                            self.kg_categories[h] = s2
                        s2.add(cat)
            logger.info(f"Loaded KG edges: links={sum(len(v) for v in self.kg_links.values())}, nodes={len(self.kg_links)}")
        except Exception as e:
            logger.warning(f"Failed to load KG: {e}")


    def _load_kg_embeddings(self):
        """Load unified mxbai KG embeddings if present.
        Tries mxbai embeddings first (unified space), falls back to PyKEEN embeddings.
        Sets:
          - self.use_kg_embeddings: bool (from OSRS_USE_KG_EMBEDDINGS env)
          - self.kg_embeddings_data: List[Dict] (mxbai format)
          - self.kg_embeddings_matrix: np.ndarray (for similarity search)
        """
        try:
            self.use_kg_embeddings = os.getenv('OSRS_USE_KG_EMBEDDINGS', '1') == '1'
            if not self.use_kg_embeddings:
                logger.info("KG embeddings disabled by environment variable")
                self.kg_embeddings_data = []
                self.kg_embeddings_matrix = None
                return

            data_dir = os.path.dirname(self.embeddings_path)
            mxbai_kg_path = os.path.join(data_dir, 'kg_entity_embeddings_mxbai.jsonl')
            mxbai_sample_path = os.path.join(data_dir, 'kg_entity_embeddings_mxbai_sample.jsonl')

            # Try to load unified mxbai KG embeddings (full first, then sample, then PyKEEN)
            if os.path.exists(mxbai_kg_path):
                logger.info("Loading unified mxbai KG embeddings (full)...")
                self._load_mxbai_kg_embeddings(mxbai_kg_path)
            elif os.path.exists(mxbai_sample_path):
                logger.info("Loading unified mxbai KG embeddings (sample)...")
                self._load_mxbai_kg_embeddings(mxbai_sample_path)
            else:
                logger.info("Unified mxbai KG embeddings not found, falling back to PyKEEN embeddings")
                self._load_pykeen_kg_embeddings()

        except Exception as e:
            logger.warning(f"Failed to load KG embeddings: {e}")
            self.kg_embeddings_data = []
            self.kg_embeddings_matrix = None

    def _load_mxbai_kg_embeddings(self, mxbai_kg_path: str):
        """Load unified mxbai KG embeddings (same format as wiki embeddings)"""
        kg_data = []
        kg_embeddings = []

        with open(mxbai_kg_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    emb = data.get('embedding')
                    if emb is None:
                        continue
                    kg_data.append(data)
                    kg_embeddings.append(emb)
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue

        if kg_embeddings:
            self.kg_embeddings_data = kg_data
            self.kg_embeddings_matrix = np.array(kg_embeddings, dtype=float)
            logger.info(f"Loaded unified mxbai KG embeddings: {self.kg_embeddings_matrix.shape}")
        else:
            logger.warning("No valid mxbai KG embeddings found")
            self.kg_embeddings_data = []
            self.kg_embeddings_matrix = None

    def _load_pykeen_kg_embeddings(self):
        """Fallback: Load legacy PyKEEN KG embeddings"""
        data_dir = os.path.dirname(self.embeddings_path)
        model_dir = os.path.join(data_dir, 'kg_model')
        ent_path = os.path.join(model_dir, 'entity_embeddings.npy')
        map_path = os.path.join(model_dir, 'entity_to_id.json')

        if not (os.path.exists(ent_path) and os.path.exists(map_path)):
            logger.info(f"PyKEEN KG embeddings not found: {ent_path}")
            self.kg_embeddings_data = []
            self.kg_embeddings_matrix = None
            return

        # Load legacy PyKEEN format (for backward compatibility)
        with open(map_path, 'r', encoding='utf-8') as f:
            entity_to_id = json.load(f)

        entity_emb = np.load(ent_path)

        # Convert to unified format
        kg_data = []
        for entity_name, entity_id in entity_to_id.items():
            if entity_id < len(entity_emb):
                kg_data.append({
                    'title': entity_name,
                    'text': f"OSRS entity: {entity_name}",
                    'source': 'knowledge_graph_pykeen',
                    'kg_entity': True,
                    'entity_id': entity_id,
                    'url': f"https://oldschool.runescape.wiki/w/{entity_name.replace(' ', '_')}"
                })

        self.kg_embeddings_data = kg_data
        self.kg_embeddings_matrix = entity_emb
        logger.info(f"Loaded PyKEEN KG embeddings: {self.kg_embeddings_matrix.shape}")
        logger.warning("Using PyKEEN embeddings (different space). Consider creating unified mxbai embeddings.")

    def _init_reranker_async(self, model_name: str):
        """Initialize the cross-encoder reranker in a background thread.
        First run may download the model and take several minutes.
        """
        try:
            logger.info(f"Initializing reranker: {model_name} (first run may download the model; this can take several minutes)...")
            from reranker_service import RerankerService
            r = RerankerService(model_name=model_name)
            if getattr(r, 'available', False):
                self.reranker = r
                logger.info(f"Reranker enabled: {model_name}")
            else:
                logger.warning("Reranker requested but not available after init; proceeding without reranking")
        except Exception as e:
            logger.warning(f"Could not initialize reranker: {e}")


    def _handle_sigusr1(self, signum, frame):
        """Signal handler to trigger async reload when embeddings file changes."""
        try:
            # Use a background thread to avoid heavy work in signal handler
            threading.Thread(target=self._reload_async, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to spawn reload thread: {e}")

    def _reload_async(self):
        try:
            logger.info("🔁 SIGUSR1 received: reloading embeddings...")
            self.load_embeddings()
            logger.info("✅ Embeddings reload complete")
        except Exception as e:
            logger.error(f"Reload failed: {e}")

    def get_or_create_chat_session(self, chat_id: str = None) -> Dict:
        """Get or create a chat session with isolated context"""
        if chat_id is None:
            chat_id = self.default_chat_id

        if chat_id not in self.chat_sessions:
            self.chat_sessions[chat_id] = {
                'conversation_history': [],
                'entity_context': {}
            }
            logger.info(f"Created new chat session: {chat_id}")

        return self.chat_sessions[chat_id]

    def get_chat_context(self, chat_id: str = None) -> Dict:
        """Get context information for a specific chat"""
        session = self.get_or_create_chat_session(chat_id)

        # Calculate context window usage
        total_context_chars = 0
        for entry in session['conversation_history']:
            total_context_chars += len(entry.get('query', ''))
            total_context_chars += len(entry.get('response', ''))


        # Estimate token usage (rough approximation: 4 chars per token)
        estimated_tokens = total_context_chars // 4
        max_context_tokens = 8192  # LLaMA 3.1 context window
        context_usage_percent = min((estimated_tokens / max_context_tokens) * 100, 100)

        return {
            'chat_id': chat_id or self.default_chat_id,
            'conversation_history': {
                'count': len(session['conversation_history']),
                'max_stored': 10,
                'usage_percent': (len(session['conversation_history']) / 10) * 100
            },
            'entity_context': {
                'count': len(session['entity_context']),
                'entities': list(session['entity_context'].keys())[:10]  # Show first 10
            },
            'context_window': {
                'estimated_tokens': estimated_tokens,
                'max_tokens': max_context_tokens,
                'usage_percent': context_usage_percent,
                'remaining_tokens': max_context_tokens - estimated_tokens
            }
        }

    def extract_entities_from_response(self, query: str, response: str) -> Dict[str, str]:
        """
        Extract key entities mentioned in the query and response for context tracking

        Args:
            query: User's question
            response: AI's response

        Returns:
            Dictionary of entities and their details
        """
        entities = {}
        logger.info(f"Extracting entities from response: '{response[:100]}...'")

        # Common OSRS entity patterns
        entity_patterns = {
            # Barrows brothers with ordinal numbers
            r'(\d+(?:st|nd|rd|th))\s+barrows\s+brother.*?is\s+([A-Z][a-z]+(?:\s+the\s+[A-Z][a-z]+)?)',
            # Direct entity identification
            r'([A-Z][a-z]+(?:\s+the\s+[A-Z][a-z]+)?)\s+(?:is|are)\s+(?:the|a)\s+([^.]+)',
            # Boss/monster mentions
            r'(Zulrah|Vorkath|Abyssal\s+Sire|Kraken|Cerberus|Thermonuclear\s+Smoke\s+Devil)',
            # Item mentions
            r'(Dragon\s+scimitar|Abyssal\s+whip|Barrows\s+(?:armor|armour)|Prayer\s+scroll)',
        }

        import re

        # Extract Barrows brother ordinal references - more flexible patterns
        barrows_patterns = [
            # "3rd barrows brother is Guthan"
            r'(\d+)(?:st|nd|rd|th)\s+barrows\s+brother.*?is\s+([A-Z][a-z]+(?:\s+the\s+[A-Z][a-z]+)?)',
            # "3rd brother is Guthan" (when context is about Barrows)
            r'(\d+)(?:st|nd|rd|th)\s+brother\s+is\s+([A-Z][a-z]+(?:\s+(?:the\s+)?[A-Z][a-z]+)?)',
            # "The 3rd brother is Guthan"
            r'[Tt]he\s+(\d+)(?:st|nd|rd|th)\s+brother\s+is\s+([A-Z][a-z]+(?:\s+(?:the\s+)?[A-Z][a-z]+)?)',
            # "The third Barrows brother is Guthan the Infested"
            r'[Tt]he\s+(first|second|third|fourth|fifth|sixth)\s+[Bb]arrows\s+brother\s+is\s+([A-Z][a-z]+(?:\s+the\s+[A-Z][a-z]+)?)',
            # "third brother is Guthan"
            r'(first|second|third|fourth|fifth|sixth)\s+brother\s+is\s+([A-Z][a-z]+(?:\s+the\s+[A-Z][a-z]+)?)',
            # "Guthan (the Infested)" format
            r'([A-Z][a-z]+)\s+\(the\s+([A-Z][a-z]+)\)'
        ]

        # Word to number mapping
        word_to_num = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6"}

        for pattern in barrows_patterns:
            barrows_match = re.search(pattern, response, re.IGNORECASE)
            if barrows_match:
                logger.info(f"Matched pattern: {pattern} with groups: {barrows_match.groups()}")

                if pattern.endswith(r'\(the\s+([A-Z][a-z]+)\)'):
                    # Handle "Guthan (the Infested)" format
                    brother_name = f"{barrows_match.group(1)} the {barrows_match.group(2)}"
                    # Try to determine ordinal from known Barrows brothers
                    barrows_order = {"ahrim": "1", "dharok": "2", "guthan": "3", "karil": "4", "torag": "5", "verac": "6"}
                    ordinal = barrows_order.get(barrows_match.group(1).lower(), "unknown")
                    if ordinal != "unknown":
                        entities[f"{ordinal}_barrows_brother"] = brother_name
                        entities["barrows_brother_context"] = f"The {ordinal} Barrows brother is {brother_name}"
                else:
                    # Handle ordinal patterns (both numeric and word)
                    ordinal_raw = barrows_match.group(1)
                    brother_name = barrows_match.group(2)

                    # Convert word ordinals to numbers
                    if ordinal_raw.lower() in word_to_num:
                        ordinal = word_to_num[ordinal_raw.lower()]
                    else:
                        ordinal = ordinal_raw

                    entities[f"{ordinal}_barrows_brother"] = brother_name
                    entities["barrows_brother_context"] = f"The {ordinal} Barrows brother is {brother_name}"
                    logger.info(f"Extracted: {ordinal}_barrows_brother = {brother_name}")
                break  # Only use first match

        # Extract general entity mentions
        for pattern in entity_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    key = match.group(1).lower().replace(' ', '_')
                    value = match.group(2)
                    entities[key] = value

        return entities

    def resolve_contextual_references(self, query: str, chat_id: str = None) -> str:
        """
        Resolve references like "the 3rd barrows brother" using conversation context

        Args:

            query: User's question that may contain contextual references
            chat_id: Chat session ID for context isolation

        Returns:
            Enhanced query with resolved references
        """
        enhanced_query = query
        session = self.get_or_create_chat_session(chat_id)
        entity_context = session['entity_context']

        # Check for ordinal Barrows brother references - multiple patterns
        import re

        # Pattern 1: "3rd barrows brother"
        barrows_ref = re.search(r'(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+barrows\s+brother', query, re.IGNORECASE)
        if barrows_ref:
            ordinal = barrows_ref.group(1)
            context_key = f"{ordinal}_barrows_brother"

            if context_key in entity_context:
                brother_name = entity_context[context_key]
                enhanced_query = re.sub(
                    r'(?:the\s+)?\d+(?:st|nd|rd|th)\s+barrows\s+brother',
                    brother_name,
                    query,
                    flags=re.IGNORECASE
                )
                logger.info(f"Resolved '{ordinal} barrows brother' to '{brother_name}' from context")

        # Pattern 2: "3rd brother" (when we have Barrows context)
        else:
            brother_ref = re.search(r'(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+brother', query, re.IGNORECASE)
            if brother_ref:
                ordinal = brother_ref.group(1)
                context_key = f"{ordinal}_barrows_brother"

                # Check if we have this brother in context
                if context_key in entity_context:
                    brother_name = entity_context[context_key]
                    enhanced_query = re.sub(
                        r'(?:the\s+)?\d+(?:st|nd|rd|th)\s+brother',
                        brother_name,
                        query,
                        flags=re.IGNORECASE
                    )
                    logger.info(f"Resolved '{ordinal} brother' to '{brother_name}' from Barrows context")

        # Check for other contextual references (only if they exist as whole words)
        pronouns = ['it', 'he', 'she', 'they', 'this', 'that']
        import re
        for pronoun in pronouns:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + pronoun + r'\b', query.lower()):
                # Look for the most recent entity mentioned
                if session['conversation_history']:
                    last_entities = session['conversation_history'][-1].get('entities', {})
                    if last_entities:
                        # Use the first entity as context (could be improved)
                        first_entity = list(last_entities.values())[0]
                        enhanced_query = re.sub(r'\b' + pronoun + r'\b', first_entity, enhanced_query, count=1, flags=re.IGNORECASE)
                        logger.info(f"Resolved pronoun '{pronoun}' to '{first_entity}' from context")
                        break

        return enhanced_query

    def find_similar_content(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find the most similar OSRS content to the query with canonical page inclusion
        Now includes hybrid retrieval using both wiki embeddings and KG embeddings
        Uses multi-faceted search for better contextual retrieval

        Args:
            query: User's question
            top_k: Number of similar results to return

        Returns:
            List of (content_data, similarity_score) tuples
        """
        # Ensure we have embeddings loaded
        if self.embeddings_matrix is None or getattr(self.embeddings_matrix, "size", 0) == 0 or not self.embeddings_data:
            logger.warning("No embeddings loaded; cannot retrieve context")
            return []

        # Use simple search
        return self._simple_search(query, top_k)

    def _simple_search(self, query: str, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Simple single-query search using embeddings
        """
        try:
            query_embedding = self.embedding_service.embed_text(query)
            wiki_results = self._find_similar_wiki_content(query, query_embedding, top_k)
            return wiki_results
        except Exception as e:
            logger.error(f"Error in simple search: {e}")
            return []

    def _extract_query_components(self, query: str) -> List[Tuple[str, float]]:
        """
        Extract key components from a query with weights based on importance
        Returns list of (component, weight) tuples
        """
        # Just return the original query - let the embeddings do the work
        return [(query, 1.0)]

    def _find_similar_wiki_content(self, query: str, query_embedding: List[float], top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar content using traditional wiki embeddings"""
        if not query_embedding:
            logger.error("Failed to create query embedding")
            return []

        # Calculate cosine similarity with all embeddings using numpy
        query_emb_array = np.array(query_embedding).reshape(1, -1)

        # Normalize vectors for cosine similarity
        query_norm = query_emb_array / np.linalg.norm(query_emb_array)
        embeddings_norm = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(query_norm, embeddings_norm.T)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for filtering

        results = []
        for idx in top_indices:
            if idx < len(self.embeddings_data):
                content = self.embeddings_data[idx]
                score = float(similarities[idx])
                results.append((content, score))

        return results[:top_k]

    def _find_similar_kg_entities(self, query: str, query_embedding: List[float], top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar entities using unified mxbai KG embeddings"""
        if not self.use_kg_embeddings or not hasattr(self, 'kg_embeddings_matrix') or self.kg_embeddings_matrix is None or not query_embedding:
            return []

        try:
            # Convert query embedding to numpy array
            query_emb = np.array(query_embedding).reshape(1, -1)

            # For unified mxbai embeddings, we're in the same space!
            # Normalize for cosine similarity
            query_norm = query_emb / np.linalg.norm(query_emb)
            kg_norms = self.kg_embeddings_matrix / np.linalg.norm(self.kg_embeddings_matrix, axis=1, keepdims=True)

            # Compute similarities with KG entities
            similarities = np.dot(query_norm, kg_norms.T)[0]

            # Get top KG entities
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if idx < len(self.kg_embeddings_data):
                    kg_content = self.kg_embeddings_data[idx]
                    score = float(similarities[idx])
                    results.append((kg_content, score))

            return results

        except Exception as e:
            logger.warning(f"KG entity search failed: {e}")
            return []

    def _combine_wiki_and_kg_results(self, wiki_results: List[Tuple[Dict[str, Any], float]],
                                   kg_results: List[Tuple[Dict[str, Any], float]],
                                   top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Combine and rerank wiki and KG results"""
        # Simple combination: interleave results with slight preference for wiki content
        combined = []

        # Add wiki results with slight boost
        for content, score in wiki_results:
            combined.append((content, score * 1.1))  # Slight boost for wiki content

        # Add KG results
        for content, score in kg_results:
            combined.append((content, score))

        # Sort by score and return top_k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

        # Intent-aware neutral query expansion (no entity hints; generic synonyms to improve recall)
        intents = self.detect_query_intent(query)
        if intents:
            expansion_terms = []
            if 'requirements' in intents:
                expansion_terms += ["requirements", "quest requirements", "skill requirements", "unlock", "completion"]
            if 'drops' in intents:
                expansion_terms += ["drop table", "unique drops", "loot"]
            if 'location' in intents:
                expansion_terms += ["located at", "located in", "near", "north of", "south of"]
            if 'stats' in intents:
                expansion_terms += ["max hit", "hitpoints", "combat level"]
            if 'strategy' in intents:
                expansion_terms += ["mechanics", "strategy", "safe spot"]
            if 'economy' in intents:
                expansion_terms += ["price", "grand exchange", "market"]
            if expansion_terms:
                try:
                    aug_query = f"{query} | " + ", ".join(expansion_terms)
                    aug_emb = self.embedding_service.embed_text(aug_query)
                    if aug_emb and query_embedding:
                        # Blend original and augmented embeddings to broaden recall slightly
                        qe = np.array(query_embedding)
                        ae = np.array(aug_emb)
                        if qe.shape == ae.shape:
                            query_embedding = (0.7 * qe + 0.3 * ae).tolist()
                except Exception:
                    pass

        if not query_embedding:
            logger.error("Failed to create query embedding")
            return []

        # Calculate cosine similarity with all embeddings using numpy
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Normalize vectors for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(query_norm, embeddings_norm.T)[0]

        # Apply organic, domain-agnostic boosts (no hardcoded entities)
        query_lower = query.lower().strip()

        import re
        item_patterns = re.findall(r'\b[a-z]+\s+[a-z]+(?:\s+[a-z]+)?\b', query_lower)
        item_patterns.extend(re.findall(r'\b[a-z]+\b', query_lower))

        def _base_title(t: str) -> str:
            t = re.sub(r"\s*\(.*?\)", "", t or "").strip()
            t = t.split('/') [0].strip()
            return t

        # Precompute base title frequency to identify canonical vs variants
        base_title_counts = {}
        for d in self.embeddings_data:
            try:
                bt = _base_title(d.get('title', ''))
                if bt:
                    base_title_counts[bt] = base_title_counts.get(bt, 0) + 1
            except Exception:
                continue

        for i, content_data in enumerate(self.embeddings_data):
            if not isinstance(content_data, dict):
                continue
            title = (content_data.get('title') or '').strip()
            title_lower = title.lower()
            text_val = (content_data.get('text') or '')

            # Title/query lexical alignment (generic)
            if title_lower == query_lower:
                similarities[i] = min(similarities[i] + 0.4, 1.0)
            elif title_lower in query_lower:
                similarities[i] = min(similarities[i] + 0.25, 1.0)
            elif query_lower in title_lower:
                similarities[i] = min(similarities[i] + 0.15, 1.0)
            else:
                for pattern in item_patterns:
                    if len(pattern) > 3 and pattern in title_lower:
                        similarities[i] = min(similarities[i] + 0.15, 1.0)
                        break

            # Structural priors (organic, content-derived)
            bt = _base_title(title)
            if bt and base_title_counts.get(bt, 0) > 1 and title != bt:
                similarities[i] = max(similarities[i] - 0.06, -1.0)  # prefer canonical/short titles over variants

            # Penalize extremely short pages; reward section-rich pages slightly
            L = len(text_val)
            if L < 400:
                similarities[i] = max(similarities[i] - 0.08, -1.0)
            elif L > 3000:
                similarities[i] = min(similarities[i] + 0.02, 1.0)
            try:
                headings = sum(1 for ln in text_val.split('\n') if ln.strip().startswith('='))
                if headings >= 3:
                    similarities[i] = min(similarities[i] + 0.03, 1.0)
            except Exception:
                pass


            # Intent-aware structural adjustments to title namespaces/subpages (domain-agnostic)
            intents = self.detect_query_intent(query)
            title_ns = title_lower
            # Penalize community/guide namespaces across the board
            if any(title_ns.startswith(ns) for ns in ["guide:", "user:", "talk:", "blog:", "old school runescape wiki:", "oldschool runescape wiki:"]):
                similarities[i] = max(similarities[i] - 0.12, -1.0)
            # Money making guides: helpful for economy, distracting otherwise
            if "money making guide/" in title_ns:
                if 'economy' in intents:
                    similarities[i] = min(similarities[i] + 0.08, 1.0)
                else:
                    similarities[i] = max(similarities[i] - 0.10, -1.0)
            # Strategy pages: boost only for strategy intent
            if "/strateg" in title_ns:  # matches /Strategies, /Strategy
                if 'strategy' in intents or 'damage_modality' in intents:
                    similarities[i] = min(similarities[i] + 0.05, 1.0)
                else:
                    similarities[i] = max(similarities[i] - 0.03, -1.0)
            # Drop-related subpages: boost for drops intent
            if any(s in title_ns for s in ["/drops", "/drop rates", "/drop table"]):
                if 'drops' in intents:
                    similarities[i] = min(similarities[i] + 0.07, 1.0)
                else:
                    similarities[i] = max(similarities[i] - 0.02, -1.0)
            # Location subpages: boost for location intent
            if any(s in title_ns for s in ["/location", "/locations"]):
                if 'location' in intents:
                    similarities[i] = min(similarities[i] + 0.06, 1.0)
            # Stats/Bestiary pages: boost for stats intent
            if any(s in title_ns for s in ["/bestiary", "/stats", "/combat info"]):
                if 'stats' in intents:
                    similarities[i] = min(similarities[i] + 0.06, 1.0)

            # Content-based intent presence adjustment (tilt retrieval toward evidence-rich docs)
            try:
                text_lower = (text_val or "").lower()
                intents2 = self.detect_query_intent(query)
                if 'requirements' in intents2:
                    if any(tok in text_lower for tok in ["requirement", "requirements", "level req", "quest requirement", "completion"]):
                        similarities[i] = min(similarities[i] + 0.06, 1.0)
                    else:
                        similarities[i] = max(similarities[i] - 0.03, -1.0)
                if 'drops' in intents2:
                    if any(tok in text_lower for tok in ["drop table", "drops", "unique", "rare drop", "loot"]):
                        similarities[i] = min(similarities[i] + 0.05, 1.0)
                if 'location' in intents2:
                    if any(tok in text_lower for tok in ["located in", "located at", "found in", "near ", "north of", "south of", "east of", "west of", "island", "dungeon", "cave"]):
                        similarities[i] = min(similarities[i] + 0.05, 1.0)
                if 'stats' in intents2:
                    if any(tok in text_lower for tok in ["hitpoints", "hp", "max hit", "combat level", "attack speed", "defence level", "defense level", "accuracy"]):
                        similarities[i] = min(similarities[i] + 0.05, 1.0)
                if 'strategy' in intents2:
                    if any(tok in text_lower for tok in ["mechanics", "strategy", "phase", "attack cycle", "avoid", "dodge", "safe spot"]):
                        similarities[i] = min(similarities[i] + 0.04, 1.0)
                if 'economy' in intents2:
                    if any(tok in text_lower for tok in ["price", "grand exchange", "market", "value"]):
                        similarities[i] = min(similarities[i] + 0.05, 1.0)
            except Exception:
                pass


        # Check if user is asking about temporary content
        include_temporary = self.query_mentions_temporary_content(query)

        # Rank and select top_k after filtering (no hardcoded inclusions)
        top_indices = np.argsort(similarities)[::-1]

        results = []
        used_bases = set()

        for idx in top_indices:
            if len(results) >= top_k:
                break
            content_data = self.embeddings_data[idx]
            if not isinstance(content_data, dict):
                continue
            title_val = content_data.get('title', '')
            text_val = (content_data.get('text', '') or '')
            categories = content_data.get('categories', []) or []

            # Filter out temporary content unless specifically requested
            if not include_temporary and self.is_temporary_content(title_val, text_val):
                continue

            # Filter out disambiguation/interface-item style pages which are poor answers
            tl = (title_val or '').lower()
            cats_lower = [str(c).lower() for c in categories]
            if 'disambiguation' in tl or any('disambiguation' in c for c in cats_lower):
                continue
            if '(interface item' in tl or any('interface item' in c for c in cats_lower):
                continue

            # Deduplicate by base title to avoid flooding with variants
            bt = _base_title(title_val)
            if bt in used_bases:
                continue
            used_bases.add(bt)

            similarity_score = similarities[idx]
            results.append((content_data, similarity_score))

        return results


    # ---- Spelling-aware query augmentation (domain-agnostic; built from title tokens) ----
    def _tokenize_simple(self, text: str):
        import re
        return [t for t in re.split(r"[^A-Za-z0-9]+", (text or "").lower()) if t]

    def _levenshtein(self, a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0: return lb
        if lb == 0: return la
        # ensure a is shorter
        if la > lb:
            a, b = b, a
            la, lb = lb, la
        prev = list(range(la + 1))
        for j in range(1, lb + 1):
            curr = [j] + [0] * la
            bj = b[j - 1]
            for i in range(1, la + 1):
                cost = 0 if a[i - 1] == bj else 1
                curr[i] = min(
                    prev[i] + 1,      # deletion
                    curr[i - 1] + 1,  # insertion
                    prev[i - 1] + cost  # substitution
                )
            prev = curr
        return prev[la]

    def _suggest_corrections(self, token: str, max_candidates: int = 3):
        if not self._spellcorr_enabled:
            return []
        if not token or token.isdigit():
            return []
        toks = self._title_token_set
        if not toks:
            return []
        # Restrict search space: same first letter (if alpha) and similar length
        first = token[0]
        min_len = max(1, len(token) - 2)
        max_len = len(token) + 2
        candidates = []
        for t in toks:
            if len(t) < min_len or len(t) > max_len:
                continue
            if first.isalpha() and t[:1] != first.lower():
                continue
            dist = self._levenshtein(token, t)
            # Dynamic threshold: allow up to 2 edits for length>=5 else 1
            thr = 2 if len(token) >= 5 else 1
            if dist <= thr:
                # Confidence: 1 - normalized distance
                conf = 1.0 - (dist / max(len(token), len(t)))
                freq = self._title_token_freq.get(t, 1)
                candidates.append((t, conf, freq))
        # sort by confidence then frequency
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[:max_candidates]

    def _augment_query_variants(self, question: str):
        if not self._spellcorr_enabled:
            return [question], {}
        toks = self._tokenize_simple(question)
        if not toks:
            return [question], {}
        suggestions = {}
        corrected = toks[:]
        for idx, tok in enumerate(toks):
            if tok in self._title_token_set:
                continue
            cands = self._suggest_corrections(tok, max_candidates=2)
            if cands:
                suggestions[tok] = [{'term': c[0], 'confidence': round(c[1], 3)} for c in cands]
                # choose the top candidate if confident enough
                if cands[0][1] >= float(os.getenv('OSRS_SPELLCORR_MIN_CONF', '0.6')):
                    corrected[idx] = cands[0][0]
        if suggestions:
            corrected_q = ' '.join(corrected)
            variants = [question]
            if corrected_q != question:
                variants.append(corrected_q)
            # optional: inline ambiguity hint variant
            amb = None
            amb_score = 0.0
            for orig, arr in suggestions.items():
                if len(arr) >= 2 and arr[1]['confidence'] >= arr[0]['confidence'] - 0.1:
                    if arr[0]['confidence'] > amb_score:
                        amb = (orig, arr[0]['term'])
                        amb_score = arr[0]['confidence']
            if amb:
                orig, corr = amb
                variants.append(question.replace(orig, f"{orig} ({corr})", 1))
            return variants, suggestions
        return [question], {}



    def identify_relevant_sections(self, query: str, context_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phase 1: Quickly identify which sections of pages are most relevant to the query

        Args:
            query: User's question
            context_data: Retrieved similar content

        Returns:
            List of relevant sections with targeted content
        """
        # Build lightweight context with just headers and key sections
        section_previews = []

        for i, data in enumerate(context_data[:3], 1):  # Limit to top 3 pages for speed
            title = data['title']
            text = data['text']

            # Extract first 2000 chars (headers, infobox, table of contents)
            preview = text[:2000]
            section_previews.append(f"[{i}] {title}:\n{preview}...")

        preview_context = "\n\n".join(section_previews)

        # Quick section identification prompt (intent-aware focus hints)
        intents = self.detect_query_intent(query)
        focus = []
        if 'damage_modality' in intents:
            focus.append("Mechanics, Strategy, Immunities, Attacks")
        if 'drops' in intents:
            focus.append("Drops, Drop table, Unique drops")
        if 'requirements' in intents:
            focus.append("Requirements, Needed levels, Quest requirements")
        if 'location' in intents:
            focus.append("Location, Where to find, Area")
        if 'stats' in intents:
            focus.append("Stats, Infobox, Combat info")
        if 'strategy' in intents:
            focus.append("Strategy, Mechanics, Phases")
        if 'economy' in intents:
            focus.append("Economy, Price, Market")
        focus_hint = ("\nFocus on sections like: " + "; ".join(focus)) if focus else ""

        section_prompt = f"""Based on this OSRS wiki preview, identify which sections would contain the answer to: "{query}"
{focus_hint}

Wiki Previews:
{preview_context}

Respond with ONLY the section names that would contain the answer (e.g., "Combat info", "Requirements", "Drops", "Stats table").
If the answer is likely in the preview already, respond with "PREVIEW".

Relevant sections:"""

        try:
            response = requests.post(
                f"{self.llama_url}/api/generate",
                json={
                    "model": self.llama_model,
                    "prompt": section_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.8,
                        "num_ctx": 8192,
                        "num_predict": 100
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception:
                    data = None
                sections = ""
                if isinstance(data, dict):
                    sections = (data.get("response") or "").strip()
                if not isinstance(sections, str):
                    sections = ""
                if sections:
                    logger.info(f"Identified relevant sections: {sections}")

                    if "PREVIEW" in sections.upper():
                        return [{"title": d.get("title",""), "text": (d.get("text","") or "")[:2000]} for d in context_data[:3]]

                    targeted_data = []
                    for d in context_data[:3]:
                        t = self.extract_sections(d.get("text", "") or "", sections)
                        if t:
                            targeted_data.append({"title": d.get("title",""), "text": t})
                    if targeted_data:
                        return targeted_data

        except Exception as e:
            logger.error(f"Section identification failed: {e}")

        # Fallback to original data
        return context_data[:3]

    def extract_sections(self, text: str, section_names: str) -> str:
        """
        Extract specific sections from wiki text based on section names
        """
        if not section_names or "PREVIEW" in section_names.upper():
            return text[:3000]  # Return first part if no specific sections

        # Common section patterns in OSRS wiki
        section_keywords = section_names.lower().split()
        extracted_parts = []

        lines = text.split('\n')
        current_section = ""
        capture = False

        for line in lines:
            line_lower = line.lower()

            # Check if this line starts a relevant section
            if any(keyword in line_lower for keyword in section_keywords):
                capture = True
                current_section = line
                extracted_parts.append(f"\n=== {line.strip()} ===")
                continue

            # Check if we hit a new section (stop capturing previous)
            if line.startswith('=') and capture:
                if not any(keyword in line_lower for keyword in section_keywords):
                    capture = False
                    continue

            # Capture content if we're in a relevant section
            if capture:
                extracted_parts.append(line)

        result = '\n'.join(extracted_parts)
        return result if result.strip() else text[:3000]  # Fallback to beginning

    def semantic_tool_search(self, query: str, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Tool-based semantic search for when direct information isn't found

        Args:
            query: Original user query
            search_terms: Relevant terms to search for

        Returns:
            List of relevant content from semantic searches
        """
        all_results = []
        used_titles = set()

        # Check if user is asking about temporary content
        include_temporary = self.query_mentions_temporary_content(query)

        for term in search_terms[:5]:  # Limit to 5 searches for performance
            try:
                # Search for this specific term
                results = self.find_similar_content(term, top_k=6)  # Get more to allow filtering

                for content, score in results:
                    title = content['title']
                    text = content.get('text', '')

                    # Skip if already used
                    if title in used_titles:
                        continue

                    # Filter out temporary content unless specifically requested
                    if not include_temporary and self.is_temporary_content(title, text):
                        logger.info(f"Filtered out temporary content: '{title}'")
                        continue

                    # Only include high-quality matches
                    if score > 0.7:
                        all_results.append(content)
                        used_titles.add(title)
                        logger.info(f"Added semantic result: '{title}' (score: {score:.3f})")

                        if len(all_results) >= 8:  # Limit total results
                            break

                if len(all_results) >= 8:
                    break

            except Exception as e:
                logger.error(f"Semantic tool search failed for '{term}': {e}")
                continue

        return all_results[:8]  # Return top 8 results

    def is_temporary_content(self, title: str, text: str = "") -> bool:
        """
        Check if content is from temporary game modes (Leagues, Deadman, etc.)

        Args:
            title: Page title
            text: Page content (optional)

        Returns:
            True if content is from temporary game modes
        """
        # Keywords that indicate temporary/special game modes
        temporary_keywords = [
            # Leagues
            'shattered relics', 'trailblazer', 'twisted league', 'league', 'leagues',
            # Deadman tournaments
            'deadman', 'tournament', 'dmm', 'deadman mode',
            # Temporary events
            'beta', 'testing', 'seasonal', 'limited time',
            # Specific temporary content patterns
            'raging echoes league', 'combat achievements/league'
        ]

        title_lower = title.lower()
        text_lower = text.lower() if text else ""

        # Check title for temporary keywords
        for keyword in temporary_keywords:
            if keyword in title_lower:
                return True

        # Check content for strong indicators (only first 500 chars for performance)
        content_sample = text_lower[:500] if text_lower else ""
        league_indicators = ['league', 'deadman', 'tournament', 'seasonal event']

        for indicator in league_indicators:
            if indicator in content_sample:
                return True

        return False

    def query_mentions_temporary_content(self, query: str) -> bool:
        """
        Check if user query specifically mentions temporary game modes

        Args:
            query: User's question

        Returns:
            True if query mentions temporary content
        """
        temporary_query_keywords = [
            'league', 'leagues', 'shattered relics', 'trailblazer', 'twisted league',
            'deadman', 'dmm', 'tournament', 'seasonal', 'beta'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in temporary_query_keywords)

    def extract_search_terms(self, query: str) -> List[str]:
        """
        Extract relevant OSRS terms from query for semantic search
        """
        # Common OSRS terms and mechanics
        osrs_terms = [
            # Combat styles
            'melee', 'range', 'ranged', 'magic', 'mage', 'prayer',
            # Item types
            'weapon', 'armor', 'armour', 'shield', 'helmet', 'boots', 'gloves',
            'rune', 'runes', 'scroll', 'prayer scroll', 'augury', 'rigour',
            # Combat mechanics
            'boss', 'monster', 'combat', 'fight', 'phase', 'special attack',
            'defence', 'defense', 'accuracy', 'damage', 'dps',
            # Economic terms
            'price', 'gold', 'gp', 'profit', 'loss', 'market', 'trade',
            'expensive', 'cheap', 'valuable', 'worthless'
        ]

        query_lower = query.lower()
        found_terms = []

        # Extract explicit terms mentioned in query
        for term in osrs_terms:
            if term in query_lower:
                found_terms.append(term)

        # Add contextual terms based on query content
        if any(word in query_lower for word in ['boss', 'fight', 'kill']):
            found_terms.extend(['boss mechanics', 'combat strategy', 'boss drops'])

        if any(word in query_lower for word in ['prayer', 'scroll']):
            found_terms.extend(['prayer scroll', 'augury', 'rigour', 'preserve'])

        if any(word in query_lower for word in ['range', 'ranged', 'mage', 'magic']):
            found_terms.extend(['ranged combat', 'magic combat', 'combat triangle'])

        if any(word in query_lower for word in ['price', 'gold', 'gp', 'profit', 'loss']):
            found_terms.extend(['market analysis', 'item prices', 'profit margins'])

        # Remove duplicates and return
        return list(set(found_terms))

    def detect_query_intent(self, query: str) -> Set[str]:
        """Lightweight, domain-agnostic intent tags derived from the user's wording.
        Used only to gate heuristics; does not inject knowledge.
        """
        q = (query or "").lower()
        intents: Set[str] = set()
        # Damage modality / allowed damage types
        if any(tok in q for tok in ["immune", "immunity", "cannot be damaged", "only damaged by", "weak to", "vulnerable to", "melee", "magic", "ranged"]):
            intents.add("damage_modality")
        # Drops
        if any(tok in q for tok in ["drop", "drops", "unique drop", "drop table"]):
            intents.add("drops")
        # Requirements
        if any(tok in q for tok in ["requirement", "requirements", "level req", "level requirement", "equip requirement"]):
            intents.add("requirements")
        # Location
        if any(tok in q for tok in ["where", "located", "location", "found in", "found at", "north", "south", "east", "west", "near", "map"]):
            intents.add("location")
        # Stats/Bestiary
        if any(tok in q for tok in ["stats", "stat", "hp", "hitpoints", "max hit", "attack speed", "combat level", "defence", "defense", "accuracy", "bestiary"]):
            intents.add("stats")
        # Strategy/Mechanics
        if any(tok in q for tok in ["how do", "how to", "avoid", "mechanics", "phase", "phases", "strategy", "safe spot", "safespot", "attack cycle"]):
            intents.add("strategy")
        # Economy/Value
        if any(tok in q for tok in ["price", "value", "market", "grand exchange", "ge", "profit", "profitable", "worth"]):
            intents.add("economy")
        # Lore/Background
        if any(tok in q for tok in ["lore", "background", "story", "history"]):
            intents.add("lore")
        # Quests
        if any(tok in q for tok in ["quest", "quest requirement", "start", "step", "subquest"]):
            intents.add("quests")
        # Skills/Training
        if any(tok in q for tok in ["skill", "training", "xp", "experience", "method"]):
            intents.add("skills")
        # Items/Equipment
        if any(tok in q for tok in ["item", "equipment", "weapon", "armour", "armor", "bonuses", "stats", "weight"]):
            intents.add("items")
        # NPCs/Monsters
        if any(tok in q for tok in ["npc", "monster", "boss", "bestiary"]):
            intents.add("npcs")
        # Minigames
        if any(tok in q for tok in ["minigame", "waves", "points", "reward"]):
            intents.add("minigames")
        # Combat mechanics
        if any(tok in q for tok in ["tick", "prayer", "overhead", "cycle", "special attack", "spec", "enrage"]):
            intents.add("combat_mechanics")
        # Equipment stats
        if any(tok in q for tok in ["stab", "slash", "crush", "magic bonus", "ranged bonus", "strength bonus", "prayer bonus"]):
            intents.add("equipment_stats")
        # Comparison
        if any(tok in q for tok in ["compare", "vs", "versus", "difference between"]):
            intents.add("comparison")
        # Timeline/updates (recency/new content)
        if any(tok in q for tok in ["update", "patch", "changelog", "release", "date", "latest", "newest", "what's new", "most recent", "recent", "new content", "recent update"]):
            intents.add("timeline")
        # Slayer
        if any(tok in q for tok in ["slayer", "task", "assigned", "master", "block", "extend"]):
            intents.add("slayer")
        # Clues
        if any(tok in q for tok in ["clue", "emote", "coordinate", "cryptic"]):
            intents.add("clue")
        # Spells/Prayers
        if any(tok in q for tok in ["spell", "prayer", "protect from", "cast", "book"]):
            intents.add("spell_prayer")
        return intents

    def _intent_patterns(self, intents: set):
        """Return regex patterns (domain-agnostic phrases) to prioritize snippets per intent."""
        pats = []
        if 'damage_modality' in intents:
            pats += [r"\bimmune to\b", r"\bcannot be damaged\b", r"\bonly (?:damaged|affected) by\b", r"\bweak to\b", r"\bvulnerable to\b",
                     r"\bmelee\b", r"\bmagic\b", r"\branged\b"]
        if 'drops' in intents:
            pats += [r"\bdrop table\b", r"\bdrops?\b", r"\bunique\b", r"\brare drop table\b", r"\bloot\b"]
        if 'requirements' in intents:
            pats += [r"\brequirements?\b", r"\blevel\b", r"\bquest requirement\b", r"\bskill\b"]
        if 'location' in intents:
            pats += [r"\blocated (?:in|at|on)\b", r"\bfound (?:in|at)\b", r"\b(?:north|south|east|west) of\b", r"\bnear\b", r"\bisland\b", r"\bdungeon\b", r"\bcave\b", r"\bcoordinates?\b"]
        if 'stats' in intents:
            pats += [r"\bhitpoints\b", r"\bhp\b", r"\bmax hit\b", r"\battack speed\b", r"\bcombat level\b", r"\bdefen[cs]e level\b", r"\baccuracy\b"]
        if 'strategy' in intents:
            pats += [r"\bstrategy\b", r"\bmechanics\b", r"\bphase\b", r"\battack cycle\b", r"\bavoid\b", r"\bdodge\b", r"\bsafe spot\b"]
        if 'economy' in intents:
            pats += [r"\bprice\b", r"\bgrand exchange\b", r"\bge\b", r"\bmarket\b", r"\bvalue\b"]
        if 'lore' in intents:
            pats += [r"\blore\b", r"\bhistory\b", r"\bbackground\b", r"\bstory\b"]
        if 'quests' in intents:
            pats += [r"\bquest\b", r"\brequirements?\b", r"\bstart\b", r"\bcompletion\b", r"\bsteps?\b"]
        if 'skills' in intents:
            pats += [r"\btraining\b", r"\bxp\b", r"\bexperience\b", r"\bmethod\b"]
        if 'items' in intents:
            pats += [r"\bitem\b", r"\bequipment\b", r"\bbonuses\b", r"\bweight\b"]
        if 'npcs' in intents:
            pats += [r"\bnpc\b", r"\bmonster\b", r"\bboss\b", r"\bbestiary\b"]
        if 'minigames' in intents:
            pats += [r"\bminigame\b", r"\bwaves?\b", r"\bpoints?\b", r"\brewards?\b"]
        if 'combat_mechanics' in intents:
            pats += [r"\btick\b", r"\boverhead\b", r"\bprayer\b", r"\battack cycle\b", r"\bspecial attack\b"]
        if 'equipment_stats' in intents:
            pats += [r"\bstab\b", r"\bslash\b", r"\bcrush\b", r"\bmagic bonus\b", r"\branged bonus\b", r"\bstrength bonus\b", r"\bprayer bonus\b"]
        if 'comparison' in intents:
            pats += [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b"]
        if 'timeline' in intents:
            pats += [r"\bupdate\b", r"\bpatch\b", r"\bchangelog\b", r"\brelease\b", r"\bdate\b"]
        if 'slayer' in intents:
            pats += [r"\bslayer\b", r"\btask\b", r"\bassigned\b", r"\bmaster\b", r"\bextend\b", r"\bblock\b"]
        if 'clue' in intents:
            pats += [r"\bclue\b", r"\bemote\b", r"\bcoordinate\b", r"\bcryptic\b"]
        if 'spell_prayer' in intents:
            pats += [r"\bspell\b", r"\bprayer\b", r"\bprotect from\b", r"\bcast\b"]
        return pats



    def generate_response(self, query: str, context_data: List[Dict[str, Any]]) -> str:
        """
        Generate response using two-phase approach with semantic tool fallback

        Args:
            query: User's question
            context_data: Retrieved similar content

        Returns:
            Generated response
        """
        # Phase 1: Identify relevant sections (fast)
        targeted_data = self.identify_relevant_sections(query, context_data)

        # Check if we have good direct information
        has_direct_info = any(score > 0.8 for _, score in
                             [(d, 0.9) for d in targeted_data])  # Placeholder check

        # Phase 2A: If no direct info, try semantic tool search
        if not has_direct_info:
            logger.info("No direct information found, attempting semantic tool search...")
            search_terms = self.extract_search_terms(query)
            if search_terms:
                logger.info(f"Extracted search terms: {search_terms}")
                semantic_results = self.semantic_tool_search(query, search_terms)
        # Guard against hallucination when evidence is absent for certain intents
        intents = self.detect_query_intent(query)
        try:
            import re as _re
            ctx_l = " ".join([(d.get('text') or '') for d in (context_data or [])]).lower()
        except Exception:
            ctx_l = ""
        if 'requirements' in intents:
            try:
                pats = self._intent_patterns({'requirements'})
                if not any(_re.search(p, ctx_l) for p in pats):
                    return ("The provided sources do not state explicit requirements. "
                            "If equip requirements are not stated in these sources, treat them as not specified here. "
                            "Acquisition/unlock requirements are also not listed in the provided context.")
            except Exception:
                pass
        if 'location' in intents:
            try:
                pats = self._intent_patterns({'location'})
                if not any(_re.search(p, ctx_l) for p in pats):
                    return ("Unknown from provided sources.")
            except Exception:
                pass
        if 'stats' in intents:
            try:
                pats = self._intent_patterns({'stats'})
                if not any(_re.search(p, ctx_l) for p in pats):
                    return ("Unknown from provided sources.")
            except Exception:
                pass
        if 'drops' in intents:
            try:
                pats = self._intent_patterns({'drops'})
                if not any(_re.search(p, ctx_l) for p in pats):
                    return ("Unknown from provided sources.")
            except Exception:
                pass

        if 'semantic_results' in locals() and semantic_results:
            # Combine original results with semantic results
            targeted_data.extend(semantic_results)
            logger.info(f"Added {len(semantic_results)} semantic search results")

        # Phase 2B: Build focused context from targeted sections
        context_parts = []
        total_chars = 0

        for i, data in enumerate(targeted_data[:8], 1):  # Limit to 8 sources
            title = data['title']
            text = data['text']

            context_parts.append(f"[{i}] {title}:\n{text}")
            total_chars += len(text) + len(title) + 20

        context = "\n\n".join(context_parts)


        # Fast grounded verdict for damage modality (organic text scan; no hardcoded entities)
        intents = self.detect_query_intent(query)
        if 'damage_modality' in intents:
            try:
                import re as _re
                neg_patterns = [
                    r"immune to melee",
                    r"cannot be damaged by melee",
                    r"melee (?:cannot|can't|won't) (?:hit|damage|affect)",
                    r"only (?:damaged|affected|harmed|hit|attacked) by (?:magic|ranged)",
                    r"can only be (?:damaged|affected|harmed|hit|attacked) by (?:magic|ranged)",
                    r"must use (?:magic|ranged)"
                ]
                pos_patterns = [
                    r"can be damaged by melee",
                    r"melee (?:can|does) (?:hit|damage|affect)",
                    r"vulnerable to melee",
                    r"weak to melee"
                ]
                for d in targeted_data:
                    t = (d.get('text') or '')
                    tl = t.lower()
                    for pat in neg_patterns:
                        m = _re.search(pat, tl)
                        if m:
                            quote_line = next((ln.strip() for ln in t.split('\n') if m.group(0) in ln.lower()), m.group(0))
                            src = f"{d.get('title') or ''}"
                            return f"No.\nQuote: \"{quote_line[:240]}\"\nSource: {src}"
                    for pat in pos_patterns:
                        m = _re.search(pat, tl)
                        if m:
                            quote_line = next((ln.strip() for ln in t.split('\n') if m.group(0) in ln.lower()), m.group(0))
                            src = f"{d.get('title') or ''}"
                            return f"Yes.\nQuote: \"{quote_line[:240]}\"\nSource: {src}"
            except Exception:
                pass

        # Strict extraction for requirements: only report explicit lines if present
        if 'requirements' in intents:
            try:
                import re as _re2
                req_lines: List[Tuple[str, str]] = []
                line_pats = [
                    r"\brequires?\b.{0,100}\b(level|lvl|quest|skill|completion)\b",
                    r"\bto (?:wear|equip)\b.{0,80}\b(level|lvl)\b",
                    r"\brequirements?\b\s*:\s*",
                ]
                for d in targeted_data:
                    t = (d.get('text') or '')
                    title = d.get('title') or ''
                    for ln in t.split('\n'):
                        lnl = ln.lower()
                        if any(_re2.search(p, lnl) for p in line_pats):
                            req_lines.append((ln.strip(), title))
                if req_lines:
                    parts = ["Extracted from wiki (explicit lines):"]
                    for ln, src in req_lines[:8]:
                        parts.append(f"- \"{ln[:220]}\" — {src}")
                    return "\n".join(parts)
                else:
                    src_titles = ", ".join({(d.get('title') or '') for d in targeted_data[:5] if d.get('title')})
                    return f"Not explicitly stated in the provided sources. Sources: {src_titles}"
            except Exception:
                pass

        # Calculate dynamic context window based on actual content
        context_tokens = int(len(context) // 3.5)  # More accurate token estimation
        prompt_overhead = 500  # Estimated tokens for prompt structure
        response_tokens = 1500  # Space for response generation

        total_tokens_needed = context_tokens + prompt_overhead + response_tokens

        # Set dynamic context window (minimum 8k, maximum 128k)
        dynamic_ctx = max(8192, min(131072, total_tokens_needed))

        logger.info(f"Dynamic context: {total_chars} chars → {context_tokens} tokens → {dynamic_ctx} ctx window")

        # Create OSRS-specific prompt with hypothetical scenario handling (intent-aware)
        intents = self.detect_query_intent(query)
        base_instructions = [
            "- Answer based ONLY on the provided OSRS wiki context",
            "- Be accurate and provide mechanical explanations when possible",
            "- If the user's question contains misconceptions, correct them clearly",
            "- Look for WHY things work the way they do (game mechanics, requirements, limitations)",
            "- If information isn't in the context, say so clearly and don't speculate",
            "- When discussing combat, always consider weapon types, ranges, and requirements",
        ]
        if 'damage_modality' in intents:
            base_instructions.extend([
                "- Do NOT confuse an enemy's \"attack style(s)\" field (describes how the enemy attacks) with what the player can use.",
                "  Determine viable player damage types using explicit phrases in the context like:",
                "  \"immune to\", \"cannot be damaged by\", \"only damaged by\", \"weak to\", \"vulnerable to\", and Strategy/Mechanics text.",
                "- If the context states immunity (e.g., immune to melee), answer that clearly and avoid implying the opposite.",
                "- Format: First word must be a strict Yes or No; then one short sentence.",
                "- Include one short direct quote from the provided context that states immunity/allowed damage (surrounded by quotes).",
                "- If no explicit statement exists in the context, say 'Unknown from provided sources.' and stop.",
            ])
        if 'requirements' in intents:
            base_instructions.extend([
                "- Extract explicit requirements mentioned in the context (skills, quests, items).",
                "- Distinguish equip requirements vs. acquisition/unlock requirements. If no equip level is stated, say so, and provide the unlock requirements instead.",
                "- Present requirements clearly (bulleted or comma-separated) and keep it concise.",
            ])
        if 'drops' in intents:
            base_instructions.extend([
                "- List named drops shown in the context; note if the page explicitly marks them as \"unique\" or part of a drop table.",
                "- Keep the answer short and attach the sources.",
            ])
        if 'location' in intents:
            base_instructions.extend([
                "- Provide a short location description using the context (e.g., area names, landmarks, directional phrases).",
            ])
        if 'stats' in intents:
            base_instructions.extend([
                "- Provide explicit stats found in the context (e.g., hitpoints, max hit, attack speed) and note any variation if indicated.",
            ])
        if 'strategy' in intents:
            base_instructions.extend([
                "- Summarize mechanics/strategy succinctly; focus on the key steps from the context.",
            ])
        if 'economy' in intents:
            base_instructions.extend([
                "- Use only the values/prices explicitly present in the provided context; do not assume live prices.",
                "- Keep the answer short and attach the sources.",
            ])
        if 'lore' in intents:
            base_instructions.extend([
                "- Provide a brief lore/background summary strictly from the provided context.",
            ])
        if 'quests' in intents:
            base_instructions.extend([
                "- List explicit quest requirements/steps only if they appear in the context; otherwise say it's not stated.",
            ])
        if 'skills' in intents:
            base_instructions.extend([
                "- Summarize any training methods or XP details only if present; avoid inventing rates.",
            ])
        if 'items' in intents or 'equipment_stats' in intents:
            base_instructions.extend([
                "- Report explicit equipment stats/bonuses and effects only if present in the context.",
            ])
        if 'npcs' in intents:
            base_instructions.extend([
                "- If combat info is requested, cite bestiary/stat lines from the context; otherwise keep NPC descriptions concise.",
            ])
        if 'minigames' in intents:
            base_instructions.extend([
                "- Summarize objectives/rewards based on the context; keep it short.",
            ])
        if 'combat_mechanics' in intents:
            base_instructions.extend([
                "- Explain mechanics succinctly using explicit lines from the context (ticks, prayer interactions, phases).",
            ])
        if 'comparison' in intents:
            base_instructions.extend([
                "- Compare only facts present in the context; if one side is missing, state there isn't enough evidence.",
            ])
        if 'timeline' in intents:
            base_instructions.extend([
                "- Cite explicit update/patch notes text and dates from the context if present.",
            ])
        if 'slayer' in intents:
            base_instructions.extend([
                "- Mention assignment/master/task details only if explicitly present; otherwise say it's not stated.",
            ])
        if 'clue' in intents:
            base_instructions.extend([
                "- Quote the exact clue line(s) from the context when providing the solution.",
            ])
        if 'spell_prayer' in intents:
            base_instructions.extend([
                "- Quote exact spell/prayer effects or levels only if present in the context; otherwise state it's not included.",
            ])

        base_instructions.extend([
            "- For hypothetical scenarios or speculation about new content:",
            "  * Acknowledge the speculative nature",
            "  * Use your OSRS knowledge to provide strategic analysis based on similar existing mechanics",
            "  * Focus on practical advice using established game patterns",
            "  * Consider economic implications based on similar historical updates",
            "- Parse all types of wiki data formats flexibly - stats can appear in many different layouts",
            "- Only include additional context if directly relevant to the specific question asked",
        ])
        instructions = "\n".join(base_instructions)

        prompt = f"""You are an expert Old School RuneScape (OSRS) assistant. Use the following OSRS wiki information to answer the user's question directly and precisely.

OSRS Wiki Context:
{context}

User Question: {query}

Instructions:
{instructions}

Answer:"""

        try:
            response = requests.post(
                f"{self.llama_url}/api/generate",
                json={
                    "model": self.llama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": dynamic_ctx,  # Use calculated dynamic context window
                        "num_predict": response_tokens  # Use calculated response tokens
                    }
                },
                timeout=max(180, dynamic_ctx // 1000)  # Scale timeout with context size
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception:
                    data = None
                if isinstance(data, dict) and isinstance(data.get("response"), str):
                    return data["response"]
                logger.error(f"LLaMA JSON missing 'response' or invalid; data={data}")
                return "Sorry, I couldn't generate a response right now."
            else:
                logger.error(f"LLaMA request failed: {response.text}")
                return "Sorry, I couldn't generate a response right now."

        except Exception as e:

            logger.error(f"Error generating response: {e}")
            return "Sorry, there was an error generating the response."
    # --- Rerank expansion and excerpt selection helpers (functional definitions) ---
    def _expand_candidates_by_neighbors(self, base_list: List[Tuple[Dict[str, Any], float]], expand_top: int = None, neighbors: int = None) -> List[Dict[str, Any]]:
        try:
            if not base_list:
                return []
            expand_top = expand_top or int(os.getenv('OSRS_RERANK_EXPAND_TOP', '5'))
            neighbors = neighbors or int(os.getenv('OSRS_RERANK_EXPAND_NEIGHBORS', '3'))
            expand_top = max(0, expand_top)
            neighbors = max(0, neighbors)
            if expand_top == 0 or neighbors == 0:
                return [cd for cd, _ in base_list]
            norm = getattr(self, 'embeddings_norm', None)
            if norm is None or norm.size == 0:
                return [cd for cd, _ in base_list]
            rows = set()
            for cd, _ in base_list:
                r = cd.get('_row')
                if isinstance(r, int):
                    rows.add(r)
            base_rows = list(rows)

            # Augment with KG neighbors (if available) for top seeds
            try:
                kg_links = getattr(self, 'kg_links', None)
                title_to_row = getattr(self, '_title_to_row', None)
                if kg_links and title_to_row:
                    for cd, _ in base_list[:expand_top]:
                        title = (cd.get('title') or '').strip()
                        nbrs = list(kg_links.get(title, []))[:neighbors]
                        for t in nbrs:
                            rr = title_to_row.get(t)
                            if isinstance(rr, int):
                                rows.add(rr)
                                if len(rows) >= len(base_rows) + expand_top * neighbors + 10:
                                    break
            except Exception:
                pass

            # Augment with KG embedding neighbors (optional)
            try:
                if getattr(self, 'use_kg_embeddings', False):
                    kg_norm = getattr(self, 'kg_entity_norm', None)
                    kg_e2i = getattr(self, 'kg_entity_to_id', None)
                    kg_i2e = getattr(self, 'kg_id_to_entity', None)
                    title_to_row = getattr(self, '_title_to_row', None)
                    if kg_norm is not None and kg_e2i and kg_i2e and title_to_row:
                        for cd, _ in base_list[:expand_top]:
                            title = (cd.get('title') or '').strip()
                            kid = kg_e2i.get(title)
                            if kid is None or not (0 <= kid < kg_norm.shape[0]):
                                continue
                            vec = kg_norm[kid]
                            sims = kg_norm @ vec
                            k = min(neighbors + 1, sims.shape[0])
                            idxs = np.argpartition(-sims, range(k))[:k]
                            idxs = sorted(idxs.tolist(), key=lambda i: float(sims[i]), reverse=True)
                            for i in idxs:
                                if i == kid:
                                    continue
                                ent = kg_i2e[i] if 0 <= i < len(kg_i2e) else None
                                if not ent:
                                    continue
                                rr = title_to_row.get(ent)
                                if isinstance(rr, int):
                                    rows.add(rr)
                                    if len(rows) >= len(base_rows) + expand_top * neighbors + 10:
                                        break
            except Exception:
                pass

            for cd, _ in base_list[:expand_top]:
                r = cd.get('_row')
                if not isinstance(r, int):
                    continue
                vec = norm[r]
                sims = norm @ vec
                k = min(neighbors + 1, sims.shape[0])
                idxs = np.argpartition(-sims, range(k))[:k]
                idxs = sorted(idxs.tolist(), key=lambda i: float(sims[i]), reverse=True)
                for i in idxs:
                    if i == r:
                        continue
                    rows.add(i)
                    if len(rows) >= len(base_rows) + expand_top * neighbors + 10:
                        break
            out = []
            seen_titles = set()
            for i in rows:
                if 0 <= i < len(self.embeddings_data):
                    cd = self.embeddings_data[i]
                    title = cd.get('title')
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    out.append(cd)
            return out
        except Exception as e:
            logger.warning(f"Neighbor expansion failed, using base list only: {e}")
            return [cd for cd, _ in base_list]

    def _select_excerpts(self, query: str, docs: List[Dict[str, Any]], per_doc: int = None) -> List[Dict[str, Any]]:
        per_doc = per_doc or int(os.getenv('OSRS_EXCERPTS_PER_DOC', '2'))
        excerpts = []
        use_reranker = self.reranker and getattr(self.reranker, 'available', False)
        for cd in docs:
            text = (cd.get('text') or '').strip()
            title = cd.get('title', '')
            if not text:
                continue
            chunks = []
            window = 500
            stride = 300
            for start in range(0, len(text), stride):
                end = min(len(text), start + window)
                if end - start < 50:
                    break
                snippet = text[start:end]
                chunks.append((start, end, snippet))
                if end == len(text):
                    break
            if not chunks:
                continue
            # Base ranking: reranker or simple lexical
            if use_reranker:
                docs_str = [sn for _, _, sn in chunks]
                base_scores = list(self.reranker.score(query, docs_str))
                base_rank = list(zip(chunks, base_scores))
            else:
                q_terms = set([t.lower() for t in query.split() if len(t) > 2])
                def score_snip(sn: str):
                    return sum(1 for t in q_terms if t in sn.lower())
                base_rank = [(c, score_snip(c[2])) for c in chunks]

            # Intent-aware phrase bonus (domain-agnostic); inactive if no clear intent
            intents = self.detect_query_intent(query)
            patterns = self._intent_patterns(intents)
            import re
            def phrase_bonus(sn: str) -> float:
                if not patterns:
                    return 0.0
                s = sn.lower()
                bonus = 0.0
                for p in patterns:
                    try:
                        if re.search(p, s):
                            bonus += 0.3
                    except re.error:
                        continue
                return bonus

            ranked2 = sorted(
                [(c, sc + phrase_bonus(c[2])) for (c, sc) in base_rank],
                key=lambda x: x[1], reverse=True
            )

            for (start, end, snip), _s in ranked2[:per_doc]:
                excerpts.append({'title': title, 'snippet': snip, 'start': start, 'end': end})
        return excerpts

    def _build_wiki_url(self, title: str) -> str:
        slug = (title or '').replace(' ', '_')
        return f"https://oldschool.runescape.wiki/w/{slug}"

    def _recent_contributors(self, title: str, limit: int = 5) -> List[Dict[str, Any]]:
        import requests
        try:
            params = {
                'action': 'query', 'prop': 'revisions', 'titles': title,
                'rvprop': 'timestamp|user|comment|ids', 'rvlimit': str(max(1, min(limit, 20))),
                'format': 'json', 'formatversion': '2'
            }
            headers = {'User-Agent': 'OSRS-AI/1.0 (contact: dev@local)'}
            resp = requests.get('https://oldschool.runescape.wiki/api.php', params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get('query', {}).get('pages', [])
            out = []
            for p in pages:
                for rev in p.get('revisions', [])[:limit]:
                    out.append({'user': rev.get('user', 'Unknown'), 'timestamp': rev.get('timestamp'), 'comment': rev.get('comment', ''), 'revid': rev.get('revid')})
            return out
        except Exception as e:
            logger.warning(f"Failed to fetch recent contributors for {title}: {e}")
            return []


    def _compute_recency_score(self, text: str) -> float:
        """Heuristic recency from text content: detects years/months; higher is newer (0..1)."""
        try:
            import re
            t = (text or "")[:8000]
            years = [int(y) for y in re.findall(r"\b(20\d{2})\b", t)]
            if not years:
                return 0.0
            y_max = max(years)
            months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december"
            has_month = bool(re.search(rf"\b({months})\b", t.lower()))
            base = max(0.0, min(1.0, (y_max - 2013) / 17.0))
            return min(1.0, base + (0.05 if has_month else 0.0))
        except Exception:
            return 0.0

    def _extract_expansion_seeds(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Use titles and neutral update phrases as seeds for recursive semantic expansion."""
        import re
        seeds: List[str] = []
        for d in docs[:12]:
            # Title-derived seed (strip parentheticals)
            t = (d.get('title') or '').strip()
            if t:
                s = t.split('(')[0].strip()
                if s and s not in seeds:
                    seeds.append(s)
            # Extract generic update/release phrases (domain-agnostic)
            txt = (d.get('text') or '')[:4000]
            # Match patterns like: "Update: Varlamore: The Final Dawn Out Now" or "Release Date: 23 July 2025"
            for m in re.findall(r"(?i)\b(update|release(?:\s+date)?)\s*[:|-]\s*([^\n\r]{4,120})", txt):
                phrase = m[1].strip()
                # Trim trailing boilerplate
                phrase = re.sub(r"\b(out now|now available)\b.*$", "", phrase, flags=re.I).strip()
                phrase = phrase.strip(' .,:;!')
                if 3 <= len(phrase) <= 120 and phrase not in seeds:
                    seeds.append(phrase)
        return seeds

    def _recursive_embedding_expand(self, base_query: str, seeds: List[str], rounds: int = 2, per_seed: int = 5, cap: int = 40) -> List[Dict[str, Any]]:
        """Recursively query embeddings with neutral seeds to gather related docs."""
        seen: set[str] = set()
        out: List[Dict[str, Any]] = []
        frontier: List[str] = [base_query] + [s for s in seeds if s]
        for _ in range(max(0, rounds)):
            next_frontier: List[str] = []
            for term in frontier[:12]:
                try:
                    results = self.find_similar_content(term, top_k=per_seed)
                    for cd, _score in results:
                        title = cd.get('title')
                        if not title or title in seen:
                            continue
                        seen.add(title)
                        out.append(cd)
                        next_frontier.append(title)
                        if len(out) >= cap:
                            return out
                except Exception:
                    continue
            frontier = next_frontier
            if not frontier:
                break
        return out

    def _filter_candidates_by_intent(self, query: str, intents: set, candidates: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Downselect candidates based on intent, but only filter items if it's clearly a monster/strategy query.
        Never remove item pages for item-acquisition queries (e.g., "bulk rune items").
        """
        if not candidates:
            return candidates
        try:
            qlow = (query or '').lower()
            item_intent = any(w in qlow for w in [
                " item", " items", "rune ", "runes", "buy", "purchase", "obtain", "get",
                "craft", "smith", "make", "alch", "grand exchange", " ge ", "price", "prices"
            ])
            monster_intent = any(w in qlow for w in [
                "kill", "slayer", "task", "monster", "boss", "burst", "barrage", "chinning",
                "catacombs", "dungeon", "cave"
            ]) or ('strategy' in (intents or set())) or ('location' in (intents or set()))
        except Exception:
            item_intent = False
            monster_intent = True  # default toward filtering only when clearly monsterish

        # Only filter item-like pages when focusing on monsters/strategy and NOT an item-acquisition query
        if not (monster_intent and not item_intent):
            return candidates

        def is_itemish(cats: List[Any], title: str) -> bool:
            tl = (title or '').lower()
            cl = [str(c).lower() for c in (cats or [])]
            item_markers = [
                'items', 'item sets', 'tradeable items', 'untradeable items', 'grand exchange',
                'drops', 'materials', 'resources', 'rocks', 'ores', 'enchants', 'crafting', 'consumables',
                'pets', 'cosmetics', 'tokens', 'scrolls', 'dust', 'granite', 'sanguine', 'metamorphic'
            ]
            if any(m in tl for m in ['granite dust', 'sanguine dust', 'metamorphic dust']):
                return True
            if any(any(m in c for m in item_markers) for c in cl):
                return True
            if any(tl.endswith(suf) for suf in [' dust', ' rocks', ' ore', ' potion', ' seeds']):
                return True
            return False

        def is_monsterish(cats: List[Any], title: str) -> bool:
            tl = (title or '').lower()
            cl = [str(c).lower() for c in (cats or [])]
            if tl == 'dust devil' or '(monster' in tl or 'dust devil' in tl:
                return True
            if any('slayer monsters' in c or 'monsters' in c for c in cl):
                return True
            if any(w in tl for w in ['catacombs', 'dungeon', 'cave']) and not is_itemish(cats, title):
                return True
            return False

        kept: List[Tuple[Dict[str, Any], float]] = []
        fallback: List[Tuple[Dict[str, Any], float]] = []
        for cd, sc in candidates:
            title = cd.get('title', '')
            cats = cd.get('categories', []) or []
            if is_itemish(cats, title):
                continue
            if is_monsterish(cats, title):
                kept.append((cd, sc))
            else:
                fallback.append((cd, sc))

        return kept if kept else fallback


    def _self_research_aggregate(self, query: str, initial_candidates: List[Tuple[Dict[str, Any], float]], retrieval_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Iteratively issue multiple embedding searches, aggregate and deduplicate best pages.
        Used when reranker is unavailable to improve evidence quality.
        """
        try:
            intents = self.detect_query_intent(query)
        except Exception:
            intents = set()

        # Build a small set of follow-up queries
        qset: List[str] = []
        qset.append(query)
        try:
            qlow = (query or '').lower()
            bulkish = any(w in qlow for w in ["bulk", "burst", "barrage", "chinning", "efficient", "fast"]) or ('strategy' in intents)
            if 'strategy' in intents or bulkish:
                qset.extend([f"{query} strategy", f"{query} best method", f"{query} efficient setup"])
            if 'location' in intents:
                qset.extend([f"{query} locations", f"{query} where to find", f"{query} catacombs dungeon cave"])
        except Exception:
            pass

        # Seed with top titles from initial candidates
        top_titles: List[str] = []
        for cd, _s in (initial_candidates or [])[:5]:
            try:
                t = (cd.get('title') or '').strip()
                if t and t not in top_titles:
                    top_titles.append(t)
            except Exception:
                continue
        for t in top_titles[:3]:
            qset.extend([f"{t} strategy", f"{t} location", f"{t} drops"])

        # Dedup queries and cap total
        seen_q = set()
        qlist: List[str] = []
        for q in qset:
            if not q or q in seen_q:
                continue
            seen_q.add(q)
            qlist.append(q)
            if len(qlist) >= 8:
                break

        def _base_title(title: str) -> str:
            import re as _re
            t = _re.sub(r"\s*\(.*?\)", "", (title or "")).strip()
            return (t.split('/') [0]).strip()

        # Aggregate pool: base_title -> (best_content, best_score, hits)
        pool: Dict[str, Tuple[Dict[str, Any], float, int]] = {}

        for subq in qlist:
            try:
                for content, score in (self.find_similar_content(subq, top_k=min(12, retrieval_k)) or []):
                    title = content.get('title', '')
                    if not title:
                        continue
                    bt = _base_title(title)
                    cats = content.get('categories', []) or []
                    bonus = 0.0
                    # Light category weighting for strategy/location
                    cl = [str(c).lower() for c in cats]
                    if ('strategy' in intents) and any('slayer' in c or 'monster' in c for c in cl):
                        bonus += 0.04
                    if ('location' in intents) and any('dungeon' in c or 'catacombs' in c or 'cave' in c for c in cl):
                        bonus += 0.03
                    # Slight preference for longer pages
                    try:
                        L = len(content.get('text') or '')
                        if L > 1500:
                            bonus += 0.02
                    except Exception:
                        pass
                    agg_score = float(score) + bonus
                    if bt not in pool:
                        pool[bt] = (content, agg_score, 1)
                    else:
                        prev_c, prev_s, hits = pool[bt]
                        # Keep the better scoring content; increment hit count
                        if agg_score > prev_s:
                            pool[bt] = (content, agg_score, hits + 1)
                        else:
                            pool[bt] = (prev_c, prev_s, hits + 1)
            except Exception:
                continue

        # Turn into list and add small multi-hit bonus
        aggregated: List[Tuple[Dict[str, Any], float]] = []
        for bt, (content, score, hits) in pool.items():
            aggregated.append((content, score + 0.01 * min(5, hits - 1)))

        aggregated.sort(key=lambda x: x[1], reverse=True)
        # Return more than top_k so later stages can pick excerpts
        cap = max(retrieval_k, 2 * max(5, retrieval_k // 2))
        return aggregated[:cap] if aggregated else (initial_candidates or [])


    def query(self, question: str, top_k: int = 5, show_sources: bool = True, chat_id: str = None) -> Dict[str, Any]:
        """
        Main query method - retrieve similar content and generate response (non-streaming)
        """
        logger.info(f"Processing query: {question} (chat_id: {chat_id or 'default'})")

        # Get or create chat session
        session = self.get_or_create_chat_session(chat_id)

        # Resolve contextual references using conversation history
        enhanced_question = self.resolve_contextual_references(question, chat_id)
        if enhanced_question != question:
            logger.info(f"Enhanced query: {enhanced_question}")

        # Retrieve broader candidate set, then optionally rerank with cross-encoder
        retrieval_k = max(top_k, int(os.getenv('OSRS_RERANK_CANDIDATES', '30')))
        with self._reload_lock:
            similar_candidates = self.find_similar_content(enhanced_question, retrieval_k)
        try:
            logger.info(f"similar_candidates: type={type(similar_candidates)} len={len(similar_candidates) if isinstance(similar_candidates, list) else 'N/A'}")
        except Exception:
            logger.info("similar_candidates: could not introspect")
        if not isinstance(similar_candidates, list):
            logger.warning("find_similar_content returned non-list; defaulting to empty list")
            similar_candidates = []
        if not similar_candidates:
            # Title-based fallback to avoid empty retrieval
            qlow = (enhanced_question or "").lower()
            toks = [t for t in qlow.split() if t]
            fallback = []
            for d in self.embeddings_data:
                if not isinstance(d, dict):
                    continue
                t = (d.get('title') or '').lower()
                if t and any(tok in t for tok in toks):
                    fallback.append((d, 0.5))
                    if len(fallback) >= top_k:
                        break
            if fallback:
                logger.info(f"Fallback retrieval used; {len(fallback)} candidates from title match")
                similar_candidates = fallback

        # Self-research aggregation if reranker is unavailable
        if not (self.reranker and getattr(self.reranker, 'available', False)):
            try:
                similar_candidates = self._self_research_aggregate(enhanced_question, similar_candidates, retrieval_k)
                logger.info(f"Self-research aggregated candidates: {len(similar_candidates)}")
            except Exception as e:
                logger.warning(f"Self-research aggregation failed: {e}")


        # Recursive semantic expansion for timeline/newest content queries
        intents = set()
        try:
            intents = self.detect_query_intent(enhanced_question)
            if 'timeline' in intents:
                seeds = self._extract_expansion_seeds([cd for cd, _ in (similar_candidates or [])])
                expanded_docs = self._recursive_embedding_expand(
                    enhanced_question,
                    seeds,
                    rounds=int(os.getenv('OSRS_RECURSE_ROUNDS', '2')),
                    per_seed=int(os.getenv('OSRS_RECURSE_PER_SEED', '5')),
                    cap=int(os.getenv('OSRS_RECURSE_CAP', '40')),
                )
                # Add generic recency seeds (domain-agnostic) to improve recall
                seeds.extend([s for s in ["out now","release date","update","patch notes","recent update"] if s not in seeds])
                try:
                    y = datetime.utcnow().year
                    for yr in [str(y), str(y-1)]:
                        if yr not in seeds:
                            seeds.append(yr)
                except Exception:
                    pass

                seen_titles = set([cd.get('title') for cd, _ in similar_candidates])
                for cd in expanded_docs:
                    t = cd.get('title')
                    if t and t not in seen_titles:
                        similar_candidates.append((cd, 0.0))
                        seen_titles.add(t)
                # If reranker disabled, apply recency heuristic to prioritize newer content
                if not (self.reranker and getattr(self.reranker, 'available', False)):
                    def _recency_key(item):
                        cdi, _ = item
                        return self._compute_recency_score(cdi.get('text', '') or '')
                    similar_candidates = sorted(similar_candidates, key=_recency_key, reverse=True)
        except Exception as e:
            logger.warning(f"Recursive expansion failed: {e}")

        # Intent-aware filtering to avoid item pages when asking for strategy/bulk
        try:
            similar_candidates = self._filter_candidates_by_intent(enhanced_question, intents, similar_candidates)
        except Exception as e:
            logger.warning(f"Intent-aware filter failed: {e}")

        # Optional reranking (BGE cross-encoder) over candidates + neighbor expansion
        if self.reranker and getattr(self.reranker, 'available', False) and similar_candidates:
            try:
                # First rerank over retrieval_k candidates
                docs = []
                for content_data, _score in similar_candidates:
                    title = content_data.get('title', '')
                    text = content_data.get('text', '') or ''
                    docs.append(f"{title}\n\n{text[:4000]}")
                scores = self.reranker.score(enhanced_question, docs)
                ranked = sorted(zip(similar_candidates, scores), key=lambda x: x[1], reverse=True)
                first_pass = [pair[0] for pair in ranked]

                # Expand by neighbors of top-N (graph-style expansion)
                expanded_docs = self._expand_candidates_by_neighbors(first_pass,
                    expand_top=int(os.getenv('OSRS_RERANK_EXPAND_TOP', '8')),
                    neighbors=int(os.getenv('OSRS_RERANK_EXPAND_NEIGHBORS', '4')))

                # Second rerank over expanded set
                docs2 = [f"{cd.get('title','')}\n\n{(cd.get('text','') or '')[:4000]}" for cd in expanded_docs]
                scores2 = self.reranker.score(enhanced_question, docs2)
                ranked2 = sorted(zip(expanded_docs, scores2), key=lambda x: x[1], reverse=True)
                final_list = [(cd, float(s)) for cd, s in ranked2]
                similar_content = final_list[:top_k]
                logger.info(f"Reranked {len(similar_candidates)} → expanded {len(expanded_docs)} → using top {len(similar_content)}")
            except Exception as e:
                logger.warning(f"Reranking/expansion failed, falling back to similarity order: {e}")
                similar_content = similar_candidates[:top_k]
        else:
            # Final slice after filter
            similar_content = similar_candidates[:top_k]

        if not similar_content:
            return {
                "response": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "similarity_scores": []
            }


        # Final intent-aware cleaning of chosen content (belt-and-suspenders)
        try:
            similar_content = self._filter_candidates_by_intent(enhanced_question, intents if 'intents' in locals() else set(), similar_content) or similar_content
        except Exception:
            pass

        # Extract just the content data for response generation
        context_data = [item[0] for item in similar_content]

        # Select short, highly-relevant excerpts for attribution
        excerpts = self._select_excerpts(enhanced_question, context_data, per_doc=int(os.getenv('OSRS_EXCERPTS_PER_DOC', '2')))
        # Attach attribution metadata (URL + per-revision attestation for snippet)
        excerpts_with_attr = []
        max_attr = int(os.getenv('OSRS_ATTR_MAX_EXCERPTS', '5'))
        for idx, ex in enumerate(excerpts):
            title = ex.get('title', '')
            ex['url'] = self._build_wiki_url(title)
            if idx < max_attr:
                try:
                    snippet = (ex.get('snippet') or '')[:400]
                    att = self.attribution.find_text_contributor(title, snippet, max_checks=12)
                    ex['attestation'] = att
                except Exception as e:
                    ex['attestation'] = {'found': False, 'message': str(e)}
            excerpts_with_attr.append(ex)

        # Generate response
        response = self.generate_response(question, context_data)

        # Extract entities from the response for context tracking
        entities = self.extract_entities_from_response(question, response)
        logger.info(f"Extracted entities: {entities}")

        # Update entity context for this chat session
        session['entity_context'].update(entities)
        logger.info(f"Updated entity context for chat {chat_id or 'default'}: {session['entity_context']}")

        # Add to conversation history for this chat session
        conversation_entry = {
            "query": question,
            "enhanced_query": enhanced_question if enhanced_question != question else None,
            "response": response,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        }
        session['conversation_history'].append(conversation_entry)

        # Keep only last 10 conversations to prevent memory bloat
        if len(session['conversation_history']) > 10:
            session['conversation_history'] = session['conversation_history'][-10:]

        # Prepare result
        result = {
            "response": response,
            "query": question,
            "timestamp": datetime.now().isoformat()
        }
        variants, suggestions = self._augment_query_variants(enhanced_question)

        # Include spelling suggestions/variants metadata to enable UI prompts
        try:
            result["spell"] = {
                "enabled": bool(self._spellcorr_enabled),
                "variants": [v for v in variants if v != enhanced_question],
                "suggestions": suggestions
            }
        except Exception:
            pass

        if show_sources:
            sources = []
            similarity_scores = []

            for content_data, similarity_score in similar_content:
                if not isinstance(content_data, dict):
                    continue
                t = content_data.get('title')
                if not t:
                    continue
                sim = max(0.0, min(0.999, float(similarity_score)))
                # Skip item-like pages only when the query is clearly monster/strategy focused
                try:
                    qlow = (enhanced_question or '').lower()
                    item_intent = any(w in qlow for w in [
                        " item", " items", "rune ", "runes", "buy", "purchase", "obtain", "get",
                        "craft", "smith", "make", "alch", "grand exchange", " ge ", "price", "prices"
                    ])
                    monster_intent = any(w in qlow for w in [
                        "kill", "slayer", "task", "monster", "boss", "burst", "barrage", "chinning",
                        "catacombs", "dungeon", "cave"
                    ])
                    cl = [str(c).lower() for c in (content_data.get('categories', []) or [])]
                    looks_item = (any('item' in c for c in cl) or any('grand exchange' in c for c in cl) or t.lower().endswith(' dust')) and ('monster' not in ' '.join(cl))
                    is_itemish = (monster_intent and not item_intent and looks_item)
                except Exception:
                    is_itemish = False
                if is_itemish:
                    continue
                sources.append({
                    "title": t,
                    "categories": content_data.get('categories', []),
                    "similarity": sim,
                    "url": self._build_wiki_url(t)
                })
                similarity_scores.append(sim)

            result["sources"] = sources
            result["similarity_scores"] = similarity_scores
            result["excerpts"] = excerpts_with_attr

        return result

    def query_stream(self, question: str, top_k: int = 5, show_sources: bool = True, chat_id: str = None):
        """Generator that yields progress dictionaries suitable for SSE from API.
        It mirrors retrieval from `query` but streams LLM generation using Ollama.
        """
        logger.info(f"[stream] Processing query: {question} (chat_id: {chat_id or 'default'})")

        # Resolve and retrieval stages
        yield {"stage": "context_resolution", "progress": 5, "message": "Resolving contextual references..."}
        enhanced_question = self.resolve_contextual_references(question, chat_id)
        if enhanced_question != question:
            logger.info(f"[stream] Enhanced query: {enhanced_question}")

        yield {"stage": "embedding_search", "progress": 15, "message": "Searching embeddings..."}
        retrieval_k = max(top_k, int(os.getenv('OSRS_RERANK_CANDIDATES', '30')))
        with self._reload_lock:
            similar_candidates = self.find_similar_content(enhanced_question, retrieval_k)
        if not isinstance(similar_candidates, list):
            similar_candidates = []
        if not similar_candidates:
            qlow = (enhanced_question or "").lower()
            toks = [t for t in qlow.split() if t]
            fallback = []
            for d in self.embeddings_data:
                if not isinstance(d, dict):
                    continue
                t = (d.get('title') or '').lower()
                if t and any(tok in t for tok in toks):
                    fallback.append((d, 0.5))
                    if len(fallback) >= top_k:
                        break
            if fallback:
                similar_candidates = fallback

        # Self-research aggregation if reranker is unavailable
        if not (self.reranker and getattr(self.reranker, 'available', False)):
            yield {"stage": "self_research", "progress": 24, "message": "Running multi-hop self-research..."}
            try:
                similar_candidates = self._self_research_aggregate(enhanced_question, similar_candidates, retrieval_k)
            except Exception as e:
                logger.warning(f"[stream] Self-research aggregation failed: {e}")
            # Apply intent-aware filtering in stream path as well
            try:
                similar_candidates = self._filter_candidates_by_intent(enhanced_question, set(), similar_candidates)
            except Exception as e:
                logger.warning(f"[stream] Intent-aware filter failed: {e}")

        # Recursive semantic expansion for timeline/newest content queries
        intents = set()
        try:
            intents = self.detect_query_intent(enhanced_question)
            if 'timeline' in intents:
                yield {"stage": "research_expansion", "progress": 26, "message": "Expanding search recursively for newest content..."}
                seeds = self._extract_expansion_seeds([cd for cd, _ in (similar_candidates or [])])
                # Add generic recency seeds (domain-agnostic) to improve recall
                seeds.extend([s for s in ["out now","release date","update","patch notes","recent update"] if s not in seeds])
                try:
                    y = datetime.now(timezone.utc).year
                    for yr in [str(y), str(y-1)]:
                        if yr not in seeds:
                            seeds.append(yr)
                except Exception:
                    pass
                expanded_docs = self._recursive_embedding_expand(
                    enhanced_question,
                    seeds,
                    rounds=int(os.getenv('OSRS_RECURSE_ROUNDS', '2')),
                    per_seed=int(os.getenv('OSRS_RECURSE_PER_SEED', '5')),
                    cap=int(os.getenv('OSRS_RECURSE_CAP', '40')),
                )
                seen_titles = set([cd.get('title') for cd, _ in similar_candidates])
                for cd in expanded_docs:
                    t = cd.get('title')
                    if t and t not in seen_titles:
                        similar_candidates.append((cd, 0.0))
                        seen_titles.add(t)
                if not (self.reranker and getattr(self.reranker, 'available', False)):
                    def _recency_key(item):
                        cdi, _ = item
                        return self._compute_recency_score(cdi.get('text', '') or '')
                    similar_candidates = sorted(similar_candidates, key=_recency_key, reverse=True)
        except Exception as e:
            logger.warning(f"Recursive expansion (stream) failed: {e}")

        # Rerank/expand
        if self.reranker and getattr(self.reranker, 'available', False) and similar_candidates:
            yield {"stage": "rerank", "progress": 28, "message": "Reranking candidates..."}
            try:
                docs = [f"{cd.get('title','')}\n\n{(cd.get('text','') or '')[:4000]}" for cd, _ in similar_candidates]
                scores = self.reranker.score(enhanced_question, docs)
                ranked = sorted(zip(similar_candidates, scores), key=lambda x: x[1], reverse=True)
                first_pass = [pair[0] for pair in ranked]
                expanded_docs = self._expand_candidates_by_neighbors(first_pass,
                    expand_top=int(os.getenv('OSRS_RERANK_EXPAND_TOP', '8')),
                    neighbors=int(os.getenv('OSRS_RERANK_EXPAND_NEIGHBORS', '4')))
                docs2 = [f"{cd.get('title','')}\n\n{(cd.get('text','') or '')[:4000]}" for cd in expanded_docs]
                scores2 = self.reranker.score(enhanced_question, docs2)
                ranked2 = sorted(zip(expanded_docs, scores2), key=lambda x: x[1], reverse=True)
                final_list = [(cd, float(s)) for cd, s in ranked2]
                similar_content = final_list[:top_k]
            except Exception as e:
                logger.warning(f"[stream] Reranking failed: {e}")
                similar_content = similar_candidates[:top_k]
        else:
            # Final slice after filter
            similar_content = similar_candidates[:top_k]

        if not similar_content:
            yield {"stage": "complete", "progress": 100, "result": {
                "response": "I couldn't find relevant information to answer your question.",
                "sources": [], "similarity_scores": []
            }}
            return


        # Final intent-aware cleaning of chosen content (belt-and-suspenders)
        try:
            similar_content = self._filter_candidates_by_intent(enhanced_question, intents if 'intents' in locals() else set(), similar_content) or similar_content
        except Exception:
            pass

        # Build excerpts and context
        yield {"stage": "content_filtering", "progress": 35, "message": "Selecting excerpts..."}
        context_data = [item[0] for item in similar_content]
        excerpts = self._select_excerpts(enhanced_question, context_data, per_doc=int(os.getenv('OSRS_EXCERPTS_PER_DOC', '2')))
        excerpts_with_attr = []
        max_attr = int(os.getenv('OSRS_ATTR_MAX_EXCERPTS', '5'))
        for idx, ex in enumerate(excerpts):
            title = ex.get('title', '')
            ex['url'] = self._build_wiki_url(title)
            if idx < max_attr:
                try:
                    snippet = (ex.get('snippet') or '')[:400]
                    att = self.attribution.find_text_contributor(title, snippet, max_checks=12)
                    ex['attestation'] = att
                except Exception as e:
                    ex['attestation'] = {'found': False, 'message': str(e)}
            excerpts_with_attr.append(ex)

        # Identify relevant sections and build context similar to generate_response
        yield {"stage": "section_identification", "progress": 45, "message": "Identifying relevant sections..."}
        targeted_data = self.identify_relevant_sections(enhanced_question, context_data)
        context_parts = []
        total_chars = 0
        for i, data in enumerate(targeted_data[:8], 1):
            title = data['title']; text = data['text']
            context_parts.append(f"[{i}] {title}:\n{text}")
            total_chars += len(text) + len(title) + 20
        context = "\n\n".join(context_parts)

        # Token planning
        context_tokens = int(len(context) // 3.5)
        prompt_overhead = 500
        response_tokens = 1500
        total_tokens = context_tokens + prompt_overhead + response_tokens
        dynamic_ctx = max(8192, min(131072, total_tokens))
        yield {"stage": "metrics", "progress": 50, "message": "Planned token budget.",
               "metrics": {"context_tokens": context_tokens, "response_tokens": response_tokens, "ctx_window": dynamic_ctx}}

        # Build prompt (reuse same instruction logic)
        intents = self.detect_query_intent(enhanced_question)
        base_instructions = [
            "- Answer based ONLY on the provided OSRS wiki context",
            "- Be accurate and provide mechanical explanations when possible",
            "- If the user's question contains misconceptions, correct them clearly",
            "- Look for WHY things work the way they do (game mechanics, requirements, limitations)",
            "- If information isn't in the context, say so clearly and don't speculate",
            "- When discussing combat, always consider weapon types, ranges, and requirements",
        ]
        if 'damage_modality' in intents:
            base_instructions.extend([
                "- Do NOT confuse an enemy's \"attack style(s)\" field (describes how the enemy attacks) with what the player can use.",
                "  Determine viable player damage types using explicit phrases in the context like:",
                "  \"immune to\", \"cannot be damaged by\", \"only damaged by\", \"weak to\", \"vulnerable to\", and Strategy/Mechanics text.",
                "- If the context states immunity (e.g., immune to melee), answer that clearly and avoid implying the opposite.",
                "- Format: First word must be a strict Yes or No; then one short sentence.",
                "- Include one short direct quote from the provided context that states immunity/allowed damage (surrounded by quotes).",
                "- If no explicit statement exists in the context, say 'Unknown from provided sources.' and stop.",
            ])
        # (trim: other intents handled similarly as in generate_response)
        instructions = "\n".join(base_instructions)
        prompt = f"""You are an expert Old School RuneScape (OSRS) assistant. Use the following OSRS wiki information to answer the user's question directly and precisely.

OSRS Wiki Context:
{context}

User Question: {enhanced_question}

Instructions:
{instructions}

Answer:"""

        # Stream generation from Ollama
        import requests as _req
        try:
            yield {"stage": "llama_generation", "progress": 55, "message": "Generating response..."}
            r = _req.post(f"{self.llama_url}/api/generate", json={
                "model": self.llama_model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": dynamic_ctx, "num_predict": response_tokens}
            }, stream=True, timeout=max(180, dynamic_ctx // 1000))
            r.raise_for_status()

            acc = []
            produced_chars = 0
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                # per-chunk text
                chunk = data.get("response")
                if isinstance(chunk, str) and chunk:
                    acc.append(chunk)
                    produced_chars += len(chunk)
                    est_tokens = int(produced_chars // 3.5) or 1
                    # Map generation to 55..99%
                    gen_pct = min(99, 55 + int((est_tokens / max(1, response_tokens)) * 44))
                    yield {"stage": "generation_progress", "progress": gen_pct, "generated_tokens": est_tokens, "target_tokens": response_tokens}
                if data.get("done"):
                    final_text = "".join(acc)
                    break
            else:
                final_text = "".join(acc)
        except Exception as e:
            yield {"stage": "error", "progress": 0, "message": f"Error during generation: {e}"}
            return

        # Entity extraction and session update
        entities = self.extract_entities_from_response(question, final_text)
        session = self.get_or_create_chat_session(chat_id)
        session['entity_context'].update(entities)
        session['conversation_history'].append({
            "query": question,
            "enhanced_query": enhanced_question if enhanced_question != question else None,
            "response": final_text,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        })
        if len(session['conversation_history']) > 10:
            session['conversation_history'] = session['conversation_history'][-10:]

        # Prepare final result
        sources = []
        similarity_scores = []
        for content_data, similarity_score in similar_content:
            t = content_data.get('title')
            if not t:
                continue
            sim = max(0.0, min(0.999, float(similarity_score)))
            # Skip item-like pages only when the query is clearly monster/strategy focused
            try:
                qlow = (enhanced_question or '').lower()
                item_intent = any(w in qlow for w in [
                    " item", " items", "rune ", "runes", "buy", "purchase", "obtain", "get",
                    "craft", "smith", "make", "alch", "grand exchange", " ge ", "price", "prices"
                ])
                monster_intent = any(w in qlow for w in [
                    "kill", "slayer", "task", "monster", "boss", "burst", "barrage", "chinning",
                    "catacombs", "dungeon", "cave"
                ])
                cl = [str(c).lower() for c in (content_data.get('categories', []) or [])]
                looks_item = (any('item' in c for c in cl) or any('grand exchange' in c for c in cl) or t.lower().endswith(' dust')) and ('monster' not in ' '.join(cl))
                is_itemish = (monster_intent and not item_intent and looks_item)
            except Exception:
                is_itemish = False
            if is_itemish:
                continue
            sources.append({"title": t, "categories": content_data.get('categories', []),
                            "similarity": sim, "url": self._build_wiki_url(t)})
            similarity_scores.append(sim)

        result = {
            "response": final_text,
            "query": question,
            "timestamp": datetime.now().isoformat(),
            "sources": sources,
            "similarity_scores": similarity_scores,
            "excerpts": excerpts_with_attr,
        }
        yield {"stage": "complete", "progress": 100, "result": result}

    def test_queries(self):
        """Test the RAG service with sample OSRS questions"""
        test_questions = [
            "What are the combat stats of a dragon scimitar?",
            "How do I get to the God Wars Dungeon?",
            "What drops bandos armor?",
            "What are the requirements for barrows gloves?",
            "How much does a whip cost?"
        ]

        logger.info("Testing OSRS RAG service...")

        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Q: {question}")
            print('='*60)

            result = self.query(question, top_k=3)

            print(f"A: {result['response']}")

            if result.get('sources'):
                print(f"\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['title']} (similarity: {source['similarity']:.3f})")

def main():
    try:
        rag_service = OSRSRAGService()

        if len(sys.argv) > 1 and sys.argv[1] == '--test':
            rag_service.test_queries()
        else:
            # Interactive mode
            print("🎮 OSRS AI Assistant Ready!")
            print("Ask me anything about Old School RuneScape!")
            print("Type 'quit' to exit.\n")

            while True:
                question = input("❓ Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                print("🤔 Thinking...")
                result = rag_service.query(question)

                print(f"\n🎯 Answer: {result['response']}")

                if result.get('sources'):
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['title']} (similarity: {source['similarity']:.3f})")

                print()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
