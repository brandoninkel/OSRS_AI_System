#!/usr/bin/env python3
"""
Create unified mxbai-embed-large embeddings for KG entities

This script takes existing KG entities and creates embeddings using the same
mxbai-embed-large model used for wiki embeddings, creating a unified embedding space.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import optimized embedding service
sys.path.append(str(Path(__file__).resolve().parents[2] / "api" / "embeddings"))
from embedding_service import EmbeddingService, EmbeddingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MXBAIKGEmbeddingCreator:
    def __init__(self, max_workers: int = 128):
        """Initialize the ULTRA HIGH-PERFORMANCE mxbai KG embedding creator"""
        self.repo_root = Path(__file__).resolve().parents[2]
        self.kg_model_dir = self.repo_root / "data" / "kg_model"
        self.output_file = self.repo_root / "data" / "kg_entity_embeddings_mxbai.jsonl"
        self.max_workers = max_workers

        # Use direct Ollama API calls for maximum speed (no embedding service overhead)
        self.model_name = "mxbai-embed-large:latest"
        self.ollama_url = "http://localhost:11434/api/embeddings"

        # Ultra-high performance settings
        self.batch_size = 200  # Larger batches for better throughput
        self.timeout = 15  # Shorter timeout for faster failure recovery
        self.max_retries = 2  # Fewer retries for speed

        # Thread-safe counters
        self.processed_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()

        # Load existing KG entity mappings
        self.entity_to_id = None
        self.load_kg_entities()

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using direct Ollama API for maximum speed"""
        for attempt in range(self.max_retries):
            try:
                # Direct API call to Ollama for maximum performance
                response = requests.post(
                    self.ollama_url,
                    json={
                        'model': self.model_name,
                        'prompt': text
                    },
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result['embedding']
                else:
                    if attempt == self.max_retries - 1:
                        return []

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return []
                time.sleep(0.05)  # Brief pause before retry
        return []
        
    def load_kg_entities(self):
        """Load existing KG entity mappings"""
        entity_map_path = self.kg_model_dir / "entity_to_id.json"
        if not entity_map_path.exists():
            raise FileNotFoundError(f"KG entity mapping not found: {entity_map_path}")
            
        with open(entity_map_path, 'r', encoding='utf-8') as f:
            self.entity_to_id = json.load(f)
            
        logger.info(f"Loaded {len(self.entity_to_id)} entities from KG")
        
    def _process_entity(self, entity_name: str) -> Dict[str, Any]:
        """Process a single entity (thread-safe)"""
        try:
            # Create embedding for entity name
            embedding = self.create_embedding(entity_name)

            if embedding:
                # Create entry in same format as wiki embeddings
                entry = {
                    'title': entity_name,
                    'text': f"OSRS entity: {entity_name}",
                    'embedding': embedding,
                    'source': 'knowledge_graph',
                    'kg_entity': True,
                    'entity_id': self.entity_to_id[entity_name],
                    'url': f"https://oldschool.runescape.wiki/w/{entity_name.replace(' ', '_')}"
                }

                # Thread-safe counter update
                with self.lock:
                    self.processed_count += 1
                    if self.processed_count % 50 == 0:
                        progress_msg = f"Processed {self.processed_count}/{len(self.entity_to_id)} entities ({self.processed_count/len(self.entity_to_id)*100:.1f}%)"
                        logger.info(progress_msg)
                        print(progress_msg, flush=True)

                return entry
            else:
                with self.lock:
                    self.failed_count += 1
                logger.warning(f"Failed to embed entity: {entity_name}")
                return None

        except Exception as e:
            with self.lock:
                self.failed_count += 1
            logger.error(f"Error processing entity '{entity_name}': {e}")
            return None

    def create_entity_embeddings(self):
        """Create mxbai embeddings for all KG entities using parallel processing"""
        if not self.entity_to_id:
            raise ValueError("No entities loaded")

        entities = list(self.entity_to_id.keys())
        total_entities = len(entities)

        logger.info(f"🚀 ULTRA HIGH-PERFORMANCE: Creating mxbai embeddings for {total_entities} entities using {self.max_workers} parallel workers...")
        print(f"⚡ MAXIMUM SPEED MODE: {total_entities:,} entities with {self.max_workers} concurrent API calls", flush=True)

        # Prepare output file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Use direct parallel processing for MAXIMUM SPEED (like the original 200/sec approach)
        start_time = time.time()

        with open(self.output_file, 'w', encoding='utf-8') as f:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks for parallel processing
                future_to_entity = {executor.submit(self._process_entity, entity): entity
                                  for entity in entities}

                # Process completed tasks as they finish
                for future in as_completed(future_to_entity):
                    entity_name = future_to_entity[future]
                    try:
                        entry = future.result()
                        if entry:
                            # Write to JSONL file (thread-safe since we're in main thread)
                            f.write(json.dumps(entry) + '\n')
                            f.flush()  # Ensure data is written immediately

                            with self.lock:
                                self.processed_count += 1

                                # Progress reporting every 100 entities
                                if self.processed_count % 100 == 0:
                                    elapsed = time.time() - start_time
                                    entities_per_sec = self.processed_count / elapsed
                                    progress_pct = self.processed_count / total_entities * 100

                                    progress_msg = f"🔥 SPEED: {entities_per_sec:.1f} entities/sec | Processed {self.processed_count}/{total_entities} ({progress_pct:.1f}%)"
                                    logger.info(progress_msg)
                                    print(progress_msg, flush=True)

                    except Exception as e:
                        logger.error(f"Future failed for entity '{entity_name}': {e}")
                        with self.lock:
                            self.failed_count += 1

        final_msg = f"✅ Created mxbai embeddings for {self.processed_count}/{total_entities} entities ({self.failed_count} failed)"
        logger.info(final_msg)
        print(final_msg, flush=True)

        file_msg = f"📁 Saved to: {self.output_file}"
        logger.info(file_msg)
        print(file_msg, flush=True)

        return self.processed_count
        
    def verify_embeddings(self):
        """Verify the created embeddings"""
        if not self.output_file.exists():
            logger.error("No embeddings file found")
            return False
            
        count = 0
        sample_entries = []
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        count += 1
                        
                        if len(sample_entries) < 3:
                            sample_entries.append(entry)
                            
            logger.info(f"✅ Verification: {count} embeddings found")
            
            for i, entry in enumerate(sample_entries, 1):
                embedding_dim = len(entry.get('embedding', []))
                logger.info(f"   Sample {i}: '{entry['title']}' - {embedding_dim} dimensions")
                
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
            
    def get_statistics(self):
        """Get statistics about the created embeddings"""
        if not self.output_file.exists():
            return {}
            
        stats = {
            'total_entities': 0,
            'embedding_dimension': 0,
            'file_size_mb': 0,
            'sample_entities': []
        }
        
        try:
            # File size
            stats['file_size_mb'] = round(self.output_file.stat().st_size / (1024 * 1024), 2)
            
            # Count and sample
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        stats['total_entities'] += 1
                        
                        if not stats['embedding_dimension']:
                            stats['embedding_dimension'] = len(entry.get('embedding', []))
                            
                        if len(stats['sample_entities']) < 5:
                            stats['sample_entities'].append(entry['title'])
                            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            
        return stats

    def create_embeddings_incremental(self) -> int:
        """Create embeddings only for new/changed entities"""
        logger.info("🔄 Creating embeddings incrementally...")

        # Load existing embeddings to see what we already have
        existing_entities = set()
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        existing_entities.add(data['entity'])
            except Exception as e:
                logger.warning(f"Could not load existing embeddings: {e}")

        # Get all entities from KG model (already loaded in __init__)
        all_entities = list(self.entity_to_id.keys())

        # Find new entities that need embeddings
        new_entities = []
        for entity in all_entities:
            if entity not in existing_entities:
                new_entities.append(entity)

        if not new_entities:
            logger.info("✅ No new entities found, embeddings are up to date")
            return 0

        logger.info(f"📊 Found {len(new_entities)} new entities to process")

        # Process only new entities and append to existing file
        return self._process_entities_incremental(new_entities)

    def create_embeddings_for_entities_file(self, entities_file: str) -> int:
        """Create embeddings for entities listed in a file"""
        logger.info(f"🎯 Creating embeddings for entities from {entities_file}")

        target_entities = []
        try:
            with open(entities_file, 'r') as f:
                for line in f:
                    entity = line.strip()
                    if entity and not entity.startswith('#'):
                        target_entities.append(entity)
        except Exception as e:
            logger.error(f"Could not read entities file: {e}")
            return 0

        if not target_entities:
            logger.warning("No entities found in file")
            return 0

        logger.info(f"📊 Processing {len(target_entities)} specific entities")
        return self._process_entities_incremental(target_entities)

    def _process_entities_incremental(self, entities: List[str]) -> int:
        """Process entities and append to existing embeddings file"""
        if not entities:
            return 0

        processed_count = 0

        # Open file in append mode to add new embeddings
        with open(self.output_file, 'a') as f:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all embedding tasks
                future_to_entity = {
                    executor.submit(self.create_embedding, entity): entity
                    for entity in entities
                }

                # Process results as they complete
                for future in as_completed(future_to_entity):
                    entity = future_to_entity[future]
                    try:
                        embedding = future.result()
                        if embedding:
                            # Write embedding to file
                            entry = {
                                'entity': entity,
                                'embedding': embedding,
                                'timestamp': time.time()
                            }
                            f.write(json.dumps(entry) + '\n')
                            f.flush()  # Ensure data is written
                            processed_count += 1

                            # Progress update
                            if processed_count % 100 == 0:
                                logger.info(f"📊 Processed {processed_count}/{len(entities)} entities")

                    except Exception as e:
                        logger.warning(f"Failed to process entity '{entity}': {e}")

        logger.info(f"✅ Incremental processing complete: {processed_count} entities added")
        return processed_count

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Create unified mxbai KG embeddings")
    parser.add_argument("--workers", type=int, default=128, help="Number of parallel workers (default: 128 for maximum speed)")
    parser.add_argument("--incremental", action="store_true", help="Only process new/changed entities")
    parser.add_argument("--entities-file", type=str, help="File containing specific entities to process (one per line)")
    args = parser.parse_args()

    if args.incremental:
        print("🔄 INCREMENTAL mxbai-embed-large KG Embeddings")
        print("=" * 60)
        print(f"⚡ INCREMENTAL MODE: {args.workers} concurrent workers")
        print("🎯 Processing only new/changed entities for maximum efficiency")
    elif args.entities_file:
        print("🎯 TARGETED mxbai-embed-large KG Embeddings")
        print("=" * 60)
        print(f"⚡ TARGETED MODE: {args.workers} concurrent workers")
        print(f"📋 Processing entities from: {args.entities_file}")
    else:
        print("🚀 ULTRA HIGH-PERFORMANCE mxbai-embed-large KG Embeddings")
        print("=" * 70)
        print(f"⚡ MAXIMUM SPEED MODE: {args.workers} concurrent workers")
        print("🔥 Direct Ollama API calls targeting 200+ entities/sec performance")
        print("🎯 Optimized with shorter timeouts and larger batches")

    try:
        creator = MXBAIKGEmbeddingCreator(max_workers=args.workers)

        # Create embeddings based on mode
        if args.incremental:
            processed = creator.create_embeddings_incremental()
        elif args.entities_file:
            processed = creator.create_embeddings_for_entities_file(args.entities_file)
        else:
            processed = creator.create_entity_embeddings()

        # Verify results
        if creator.verify_embeddings():
            stats = creator.get_statistics()

            print("\n✅ SUCCESS! Unified KG embeddings created")
            print(f"📊 Statistics:")
            print(f"   Entities: {stats['total_entities']:,}")
            print(f"   Dimensions: {stats['embedding_dimension']}")
            print(f"   File size: {stats['file_size_mb']} MB")
            print(f"   Sample entities: {', '.join(stats['sample_entities'][:3])}")
            print(f"\n📁 File: {creator.output_file}")
            print("\n🎯 Next step: RAG service will automatically use these unified embeddings!")

        else:
            print("❌ Verification failed")
            return 1

    except Exception as e:
        logger.error(f"Failed to create KG embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
