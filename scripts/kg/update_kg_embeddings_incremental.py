#!/usr/bin/env python3
"""
Incrementally update KG entity embeddings based on changed wiki pages.

Instead of regenerating all 149k entity embeddings (18-21 minutes), this script:
1. Identifies which entities are affected by changed pages
2. Only re-embeds those entities (typically 10-100 entities, 1-10 seconds)
3. Updates the existing embeddings file with new embeddings and metadata

Usage:
  # Update entities from specific changed pages
  python3 scripts/kg/update_kg_embeddings_incremental.py --changed-pages "Abyssal whip,Dragon scimitar"
  
  # Full rebuild (regenerate all embeddings)
  python3 scripts/kg/update_kg_embeddings_incremental.py --full-rebuild
  
  # Auto-detect changes from watchdog metadata
  python3 scripts/kg/update_kg_embeddings_incremental.py --auto-detect
"""

import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.embeddings.embedding_service import EmbeddingService, EmbeddingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
KG_MODEL_DIR = DATA_DIR / "kg_model"
EMBEDDINGS_FILE = DATA_DIR / "kg_entity_embeddings_mxbai.jsonl"
ENTITY_TO_PAGES_FILE = DATA_DIR / "kg_entity_to_pages.json"
PAGE_TO_ENTITIES_FILE = DATA_DIR / "kg_page_to_entities.json"
ENTITY_TO_ID_FILE = KG_MODEL_DIR / "entity_to_id.json"
TRIPLES_CSV = DATA_DIR / "osrs_kg_triples.csv"


class IncrementalKGEmbeddingUpdater:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.embedding_service = None
        self.entity_to_pages = {}
        self.page_to_entities = {}
        self.entity_to_id = {}
        self.existing_embeddings = {}
        
    def initialize(self):
        """Initialize embedding service and load mappings"""
        logger.info("🚀 Initializing incremental KG embedding updater...")
        
        # Initialize embedding service
        config = EmbeddingConfig(
            model_name="mxbai-embed-large:latest",
            max_concurrent_requests=64
        )
        self.embedding_service = EmbeddingService(config)
        logger.info("✅ Embedding service initialized")
        
        # Load entity mappings
        if not ENTITY_TO_PAGES_FILE.exists():
            logger.error(f"❌ Entity mapping not found: {ENTITY_TO_PAGES_FILE}")
            logger.info("Run: python3 scripts/kg/build_entity_mapping.py")
            return False
        
        with open(ENTITY_TO_PAGES_FILE, 'r') as f:
            self.entity_to_pages = json.load(f)
        logger.info(f"✅ Loaded {len(self.entity_to_pages):,} entity → pages mappings")
        
        with open(PAGE_TO_ENTITIES_FILE, 'r') as f:
            self.page_to_entities = json.load(f)
        logger.info(f"✅ Loaded {len(self.page_to_entities):,} page → entities mappings")
        
        # Load entity IDs
        if not ENTITY_TO_ID_FILE.exists():
            logger.error(f"❌ Entity IDs not found: {ENTITY_TO_ID_FILE}")
            return False
        
        with open(ENTITY_TO_ID_FILE, 'r') as f:
            self.entity_to_id = json.load(f)
        logger.info(f"✅ Loaded {len(self.entity_to_id):,} entity IDs")
        
        # Load existing embeddings
        if EMBEDDINGS_FILE.exists():
            logger.info(f"📂 Loading existing embeddings from {EMBEDDINGS_FILE}")
            with open(EMBEDDINGS_FILE, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    title = data.get('title')
                    if title:
                        self.existing_embeddings[title] = data
            logger.info(f"✅ Loaded {len(self.existing_embeddings):,} existing embeddings")
        else:
            logger.info("ℹ️  No existing embeddings found - will create from scratch")
        
        return True
    
    def find_affected_entities(self, changed_pages: List[str]) -> Set[str]:
        """Find all entities affected by changed pages"""
        affected = set()
        
        for page in changed_pages:
            if page in self.page_to_entities:
                entities = self.page_to_entities[page]
                affected.update(entities)
                logger.info(f"  {page}: {len(entities)} entities")
        
        return affected
    
    def get_source_revids_for_entity(self, entity: str) -> List[int]:
        """Get all source page revids for an entity"""
        # This would require loading the triples and finding all revids
        # For now, return empty list - we'll add this in a follow-up
        return []
    
    async def embed_entities(self, entities: List[str], progress_callback=None) -> Dict[str, List[float]]:
        """Embed a list of entities using batch API with progress reporting"""
        logger.info(f"🔥 Embedding {len(entities)} entities using batch API...")

        result = {}
        total = len(entities)

        # Process in smaller chunks to avoid Ollama timeouts
        # Ollama's batch API can handle ~100-200 texts efficiently
        chunk_size = 100  # Smaller chunks for reliability

        for i in range(0, total, chunk_size):
            chunk = entities[i:i + chunk_size]
            chunk_end = min(i + chunk_size, total)

            try:
                # Embed this chunk with timeout protection
                embeddings = await asyncio.wait_for(
                    self.embedding_service.embed_texts_async(chunk, use_batch_api=True),
                    timeout=120.0  # 2 minute timeout per chunk
                )

                # Store results
                for entity, embedding in zip(chunk, embeddings):
                    if embedding:
                        result[entity] = embedding

            except asyncio.TimeoutError:
                logger.error(f"⏱️  Timeout embedding chunk {i}-{chunk_end}, skipping...")
                continue
            except Exception as e:
                logger.error(f"❌ Error embedding chunk {i}-{chunk_end}: {e}")
                continue

            # Report progress
            progress_pct = (chunk_end / total) * 100
            if progress_callback:
                progress_callback(chunk_end, total, progress_pct)

            # Force flush to ensure progress is visible
            print(f"Progress: {progress_pct:.1f}%", flush=True)
            print(f"Status: embedding KG entities ({chunk_end}/{total})", flush=True)
            sys.stdout.flush()

        return result
    
    def update_embeddings(self, changed_pages: List[str] = None, deleted_pages: List[str] = None, full_rebuild: bool = False):
        """Update embeddings incrementally or do full rebuild"""

        if full_rebuild:
            logger.info("🔄 Full rebuild requested - checking for missing embeddings")
            all_entities = set(self.entity_to_id.keys())
            already_embedded = set(self.existing_embeddings.keys())
            entities_to_update = list(all_entities - already_embedded)

            if entities_to_update:
                logger.info(f"📊 Found {len(already_embedded):,} existing embeddings")
                logger.info(f"🚀 Need to embed {len(entities_to_update):,} remaining entities")
            else:
                logger.info(f"✅ All {len(all_entities):,} entities already embedded!")
                return
        elif changed_pages or deleted_pages:
            affected_entities = set()

            if changed_pages:
                logger.info(f"🔍 Finding entities affected by {len(changed_pages)} changed pages...")
                affected_entities.update(self.find_affected_entities(changed_pages))

            if deleted_pages:
                logger.info(f"🗑️  Finding entities affected by {len(deleted_pages)} deleted pages...")
                deleted_entities = self.find_affected_entities(deleted_pages)
                affected_entities.update(deleted_entities)
                logger.info(f"   Found {len(deleted_entities)} entities from deleted pages")

            entities_to_update = list(affected_entities)
            logger.info(f"✅ Found {len(entities_to_update)} total affected entities")
        else:
            logger.error("❌ Must specify either --changed-pages, --deleted-pages, or --full-rebuild")
            return

        if not entities_to_update:
            logger.info("✅ No entities to update")
            return
        
        # Embed entities
        logger.info(f"🚀 Embedding {len(entities_to_update)} entities...")
        start_time = datetime.now()

        # Progress callback
        def report_progress(current, total, pct):
            print(f"Progress: {pct:.1f}%", flush=True)

        new_embeddings = asyncio.run(self.embed_entities(entities_to_update, progress_callback=report_progress))

        elapsed = (datetime.now() - start_time).total_seconds()
        rate = len(new_embeddings) / elapsed if elapsed > 0 else 0
        logger.info(f"✅ Embedded {len(new_embeddings)} entities in {elapsed:.1f}s ({rate:.1f} entities/sec)")
        
        # Update existing embeddings
        updated_count = 0
        new_count = 0
        
        for entity, embedding in new_embeddings.items():
            entity_data = {
                'title': entity,
                'text': entity,
                'source': 'knowledge_graph',
                'kg_entity': True,
                'entity_id': self.entity_to_id.get(entity, -1),
                'url': f"https://oldschool.runescape.wiki/w/{entity.replace(' ', '_')}",
                'embedding': embedding,
                'metadata': {
                    'source_pages': self.entity_to_pages.get(entity, []),
                    'updated_at': datetime.now().isoformat(),
                    'embedding_model': 'mxbai-embed-large:latest'
                }
            }
            
            if entity in self.existing_embeddings:
                updated_count += 1
            else:
                new_count += 1
            
            self.existing_embeddings[entity] = entity_data
        
        logger.info(f"📊 Updated {updated_count} existing embeddings, added {new_count} new embeddings")
        
        # Save updated embeddings
        logger.info(f"💾 Saving embeddings to {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            for entity_data in self.existing_embeddings.values():
                f.write(json.dumps(entity_data) + '\n')
        
        logger.info(f"✅ Incremental update complete!")
        logger.info(f"📈 Total embeddings: {len(self.existing_embeddings):,}")


def main():
    parser = argparse.ArgumentParser(description="Incrementally update KG entity embeddings")
    parser.add_argument('--changed-pages', type=str, help='Comma-separated list of changed page titles')
    parser.add_argument('--deleted-pages', type=str, help='Comma-separated list of deleted page titles')
    parser.add_argument('--full-rebuild', action='store_true', help='Rebuild all embeddings from scratch')
    parser.add_argument('--auto-detect', action='store_true', help='Auto-detect changes from watchdog metadata')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for embedding')

    args = parser.parse_args()

    updater = IncrementalKGEmbeddingUpdater(batch_size=args.batch_size)

    if not updater.initialize():
        logger.error("❌ Initialization failed")
        return 1

    # Parse changed and deleted pages
    changed_pages = None
    deleted_pages = None

    if args.changed_pages:
        changed_pages = [p.strip() for p in args.changed_pages.split(',') if p.strip()]

    if args.deleted_pages:
        deleted_pages = [p.strip() for p in args.deleted_pages.split(',') if p.strip()]

    if args.auto_detect:
        # TODO: Implement auto-detection from watchdog metadata
        logger.warning("⚠️  Auto-detect not yet implemented, use --changed-pages for now")
        return 1

    # Update embeddings
    updater.update_embeddings(changed_pages=changed_pages, deleted_pages=deleted_pages, full_rebuild=args.full_rebuild)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

