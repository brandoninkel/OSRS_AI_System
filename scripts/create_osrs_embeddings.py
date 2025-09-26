#!/usr/bin/env python3
"""
Create OSRS Embeddings from Wiki Content
Processes the properly parsed OSRS wiki content JSONL and creates embeddings
"""

import json
import os
import sys
import argparse
import asyncio
import hashlib
import time
from typing import List, Dict, Any
import logging
from datetime import datetime
from pathlib import Path
import math

import fcntl
import signal

# Add the api embeddings to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api', 'embeddings'))
from embedding_service import EmbeddingService, EmbeddingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OSRSEmbeddingCreator:
    def __init__(self):
        # Paths
        self.wiki_content_path = "/Users/brandon/Documents/projects/GE/data/osrs_wiki_content.jsonl"
        self.embeddings_output_path = "/Users/brandon/Documents/projects/GE/data/osrs_embeddings.jsonl"

        # Initialize embedding service
        config = EmbeddingConfig(
            model_name="mxbai-embed-large:latest",
            batch_size=32,  # Smaller batches for stability
            max_retries=3,
            timeout=60
        )
        self.embedding_service = EmbeddingService(config)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.embeddings_output_path), exist_ok=True)

    def load_wiki_content(self) -> List[Dict[str, Any]]:
        """Load OSRS wiki content from JSONL file"""
        logger.info(f"Loading wiki content from: {self.wiki_content_path}")

        if not os.path.exists(self.wiki_content_path):
            raise FileNotFoundError(f"Wiki content file not found: {self.wiki_content_path}")

        content = []
        with open(self.wiki_content_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        data = json.loads(line)
                        content.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(content)} wiki pages")
        return content

    def prepare_text_for_embedding(self, page: Dict[str, Any]) -> str:
        """
        Prepare page content for embedding with enhanced context
        Includes title, categories, and structured content
        """
        title = page.get('title', 'Unknown')
        categories = page.get('categories', [])
        text = page.get('text', '')

        # Create enhanced text with metadata for better embeddings
        enhanced_text = f"Title: {title}\n"

        if categories:
            # Clean up categories (remove object references)
            clean_categories = []
            for cat in categories:
                if isinstance(cat, str):
                    clean_categories.append(cat)
                elif isinstance(cat, dict) and 'category' in cat:
                    clean_categories.append(cat['category'])

            if clean_categories:
                enhanced_text += f"Categories: {', '.join(clean_categories)}\n"
        enhanced_text += f"Content: {text}"

        return enhanced_text


    def get_page_key(self, page: Dict[str, Any]) -> str:
        """Stable key per page revision: title|revid or title|hash(text)."""
        title = page.get('title', 'Unknown')
        revid = page.get('revid')
        if revid is None:
            text = page.get('text', '') or ''
            h = hashlib.md5(text.encode('utf-8')).hexdigest()
            return f"{title}|h:{h}"
        return f"{title}|r:{revid}"

    def build_existing_index(self) -> set:
        """Scan existing embeddings file and build a set of page keys already embedded."""
        idx = set()
        if not os.path.exists(self.embeddings_output_path):
            return idx
        with open(self.embeddings_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    title = rec.get('title')
                    meta = rec.get('metadata', {})
                    revid = meta.get('revid')
                    text = rec.get('text', '')
                    if title is None:
                        continue
                    if revid is None:
                        h = hashlib.md5((text or '').encode('utf-8')).hexdigest()
                        key = f"{title}|h:{h}"
                    else:
                        key = f"{title}|r:{revid}"
                    idx.add(key)
                except Exception:
                    continue
        return idx

    def select_pages_to_embed(self, wiki_pages: List[Dict[str, Any]], existing_idx: set) -> List[Dict[str, Any]]:
        """Return only pages whose key is not present in the existing index."""
        selected = []
        for p in wiki_pages:
            key = self.get_page_key(p)
            if key not in existing_idx:
                selected.append(p)
        return selected

    def notify_rag_update(self):
        """Notify the RAG service (if running) to reload embeddings via SIGUSR1."""
        try:
            pid_file = os.getenv('OSRS_RAG_PID_FILE', '/Users/brandon/Documents/projects/GE/data/rag_service.pid')
            if os.path.exists(pid_file):
                with open(pid_file, 'r') as pf:
                    pid_str = pf.read().strip()
                if pid_str.isdigit():
                    os.kill(int(pid_str), signal.SIGUSR1)
                    logger.info("🔔 Notified RAG to reload (SIGUSR1)")
        except Exception as e:
            logger.debug(f"RAG notify skipped: {e}")

    def embed_and_append(self, pages: List[Dict[str, Any]], use_async: bool = False, chunk_size: int = 200) -> int:
        """Embed provided pages in chunks and append to embeddings file. Returns count written."""
        if not pages:
            return 0
        total = len(pages)
        total_chunks = (total + chunk_size - 1) // chunk_size
        written = 0
        failed = 0
        start_time = time.time()
        with open(self.embeddings_output_path, 'a', encoding='utf-8') as f:
            for cidx in range(total_chunks):
                start = cidx * chunk_size
                end = min(start + chunk_size, total)
                chunk = pages[start:end]
                texts_to_embed = []
                page_metadata = []
                for page in chunk:
                    texts_to_embed.append(self.prepare_text_for_embedding(page))
                    page_metadata.append({
                        'title': page.get('title', 'Unknown'),
                        'categories': page.get('categories', []),
                        'revid': page.get('revid'),
                        'timestamp': page.get('timestamp'),
                        'text_length': len(page.get('text', ''))
                    })
                # Embed this chunk
                if use_async:
                    embeddings = asyncio.run(self.embedding_service.embed_texts_async(texts_to_embed))
                else:
                    embeddings = self.embedding_service.embed_texts(texts_to_embed)
                # Append results immediately with exclusive file lock, then notify RAG
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                except Exception:
                    pass
                for i, (embedding, metadata, original_text) in enumerate(zip(embeddings, page_metadata, texts_to_embed)):
                    if embedding:
                        embedding_data = {
                            'id': int(datetime.now().timestamp()*1000) + i,
                            'title': metadata['title'],
                            'categories': metadata['categories'],
                            'text': original_text,
                            'embedding': embedding,
                            'metadata': {
                                'revid': metadata['revid'],
                                'timestamp': metadata['timestamp'],
                                'text_length': metadata['text_length'],
                                'embedding_model': self.embedding_service.config.model_name,
                                'created_at': datetime.now().isoformat()
                            }
                        }
                        f.write(json.dumps(embedding_data) + '\n')
                        written += 1
                    else:
                        failed += 1
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass

                # Notify RAG service to reload (event-driven, no polling)
                self.notify_rag_update()

                elapsed = max(0.001, time.time() - start_time)
                done = end
                rate = done / elapsed
                logger.info(f"🔄 Chunk {cidx+1}/{total_chunks} | {done}/{total} done | wrote {written} | failed {failed} | {rate:.1f}/s")
        return written

    def run_incremental_once(self, limit: int = None, use_async: bool = False, chunk_size: int = 200) -> Dict[str, int]:
        """Run a single incremental pass: detect new/changed pages and append embeddings."""
        wiki_pages = self.load_wiki_content()
        if limit:
            wiki_pages = wiki_pages[:limit]
        existing_idx = self.build_existing_index()
        to_process = self.select_pages_to_embed(wiki_pages, existing_idx)
        logger.info(f"📊 Incremental: {len(to_process)} pages to embed (existing: {len(existing_idx)}) | chunk_size={chunk_size}")
        written = self.embed_and_append(to_process, use_async=use_async, chunk_size=chunk_size)
        return {"to_process": len(to_process), "written": written}

    def follow(self, interval_sec: int = 60, use_async: bool = False, chunk_size: int = 200):
        """Watch the wiki content file and incrementally append embeddings forever."""
        logger.info(f"👀 Follow mode: scanning every {interval_sec}s for new/updated pages | chunk_size={chunk_size}")
        try:
            while True:
                stats = self.run_incremental_once(use_async=use_async, chunk_size=chunk_size)
                logger.info(f"✅ Incremental pass complete: wrote {stats['written']} of {stats['to_process']} candidates")
                time_to_sleep = max(1, interval_sec)
                for _ in range(time_to_sleep):
                    print(" .", end='', flush=True)
                    time.sleep(1)
                print()
        except KeyboardInterrupt:
            logger.info("🛑 Follow mode stopped by user")


    def create_embeddings(self, limit: int = None) -> None:
        """Create embeddings for OSRS wiki content"""
        logger.info("Starting OSRS embedding creation...")

        # Load wiki content
        wiki_pages = self.load_wiki_content()

        if limit:
            wiki_pages = wiki_pages[:limit]
            logger.info(f"Processing limited set of {limit} pages")

        # Prepare texts for embedding
        logger.info("Preparing texts for embedding...")
        texts_to_embed = []
        page_metadata = []

        for page in wiki_pages:
            enhanced_text = self.prepare_text_for_embedding(page)
            texts_to_embed.append(enhanced_text)

            # Store metadata for later
            page_metadata.append({
                'title': page.get('title', 'Unknown'),
                'categories': page.get('categories', []),
                'revid': page.get('revid'),
                'timestamp': page.get('timestamp'),
                'text_length': len(page.get('text', ''))
            })

        logger.info(f"Prepared {len(texts_to_embed)} texts for embedding")

        # Create embeddings
        logger.info("Creating embeddings... This may take a while...")
        embeddings = self.embedding_service.embed_texts(texts_to_embed)

        # Save embeddings with metadata
        logger.info(f"Saving embeddings to: {self.embeddings_output_path}")

        with open(self.embeddings_output_path, 'w', encoding='utf-8') as f:
            for i, (embedding, metadata, original_text) in enumerate(zip(embeddings, page_metadata, texts_to_embed)):
                if embedding:  # Only save if embedding was successful
                    embedding_data = {
                        'id': i,
                        'title': metadata['title'],
                        'categories': metadata['categories'],
                        'text': original_text,
                        'embedding': embedding,
                        'metadata': {
                            'revid': metadata['revid'],
                            'timestamp': metadata['timestamp'],
                            'text_length': metadata['text_length'],
                            'embedding_model': self.embedding_service.config.model_name,
                            'created_at': datetime.now().isoformat()
                        }
                    }

                    f.write(json.dumps(embedding_data) + '\n')

        # Print statistics
        successful_embeddings = len([e for e in embeddings if e])
        logger.info(f"✅ Successfully created {successful_embeddings}/{len(texts_to_embed)} embeddings")
        logger.info(f"📁 Embeddings saved to: {self.embeddings_output_path}")

        # Print cache stats
        cache_stats = self.embedding_service.get_cache_stats()
        logger.info(f"📊 Cache stats: {cache_stats}")

    def test_embeddings(self) -> None:
        """Test the created embeddings"""
        logger.info("Testing embeddings...")

        if not os.path.exists(self.embeddings_output_path):
            logger.error("No embeddings file found. Run create_embeddings first.")
            return

        # Load a few embeddings
        with open(self.embeddings_output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Test first 3
                    break

                data = json.loads(line)
                title = data['title']
                embedding_dim = len(data['embedding'])
                text_preview = data['text'][:100] + "..." if len(data['text']) > 100 else data['text']

                logger.info(f"✅ {title}: {embedding_dim}D embedding")
                logger.info(f"   Text preview: {text_preview}")

def main():
    parser = argparse.ArgumentParser(description="OSRS Embeddings: full, incremental, or follow mode")
    parser.add_argument('--full', action='store_true', help='Full rebuild of embeddings (overwrites output)')
    parser.add_argument('--incremental', action='store_true', help='Run a single incremental pass (default if no mode given)')
    parser.add_argument('--follow', action='store_true', help='Watch wiki content and append embeddings continuously')
    parser.add_argument('--interval', type=int, default=60, help='Follow-mode scan interval in seconds (default 60)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of pages for quick runs')
    parser.add_argument('--async', dest='async_mode', action='store_true', help='Use async embedding with concurrent requests')
    parser.add_argument('--max-concurrency', type=int, default=None, help='Max concurrent requests for async embedding (default config)')
    parser.add_argument('--chunk-size', type=int, default=200, help='Chunk size for streaming embedding/appends (default 200)')
    parser.add_argument('--test-embeddings', action='store_true', help='Load a few embeddings and print stats')
    args = parser.parse_args()

    creator = OSRSEmbeddingCreator()

    # Optional override for max concurrency
    if args.max_concurrency is not None:
        try:
            creator.embedding_service.config.max_concurrent_requests = max(1, int(args.max_concurrency))
            logger.info(f"Set max_concurrent_requests = {creator.embedding_service.config.max_concurrent_requests}")
        except Exception:
            pass

    # MODE SELECTION
    if args.full:
        if args.limit:
            logger.info(f"Full rebuild with limit={args.limit}")
        creator.create_embeddings(limit=args.limit)
    elif args.follow:
        creator.follow(interval_sec=args.interval, use_async=args.async_mode, chunk_size=args.chunk_size)
    else:
        # Default to incremental if no explicit mode
        stats = creator.run_incremental_once(limit=args.limit, use_async=args.async_mode, chunk_size=args.chunk_size)
        logger.info(f"Done: wrote {stats['written']} of {stats['to_process']} candidates")

    if args.test_embeddings:
        creator.test_embeddings()

if __name__ == "__main__":
    main()
