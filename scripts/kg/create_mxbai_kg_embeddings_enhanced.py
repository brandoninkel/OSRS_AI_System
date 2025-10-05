#!/usr/bin/env python3
"""
Enhanced mxbai-embed-large KG embeddings creator with resume capability and change tracking

Features:
- Resume capability: Skip already processed entities
- Change tracking: Detect when source KG files change
- Apple Metal GPU acceleration (automatic with Ollama)
- Incremental writing: Save progress as it goes
- Metadata tracking: Track processing stats and source file hashes
"""

import json
import logging
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Direct Ollama integration (no dependency on embedding service)
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMXBAIKGEmbeddingCreator:
    def __init__(self, max_workers: int = 8, resume: bool = True, force_rebuild: bool = False):
        """Initialize the enhanced mxbai KG embedding creator"""
        self.repo_root = Path(__file__).resolve().parents[2]
        self.kg_model_dir = self.repo_root / "data" / "kg_model"
        self.output_file = self.repo_root / "data" / "kg_entity_embeddings_mxbai.jsonl"
        self.metadata_file = self.repo_root / "data" / "kg_entity_embeddings_mxbai_metadata.json"
        self.max_workers = max_workers
        self.resume = resume and not force_rebuild
        self.force_rebuild = force_rebuild
        
        # Thread-safe counters
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.lock = threading.Lock()
        
        # Source file tracking
        self.source_files = {
            'entities': self.kg_model_dir / "entity_to_id.json",
            'kg_nodes': self.repo_root / "data" / "osrs_kg_nodes.jsonl",
            'kg_triples': self.repo_root / "data" / "osrs_kg_triples.csv"
        }
        
        # Load existing data
        self.entity_to_id = {}
        self.existing_entities: Set[str] = set()
        self.metadata = {}
        
        self._load_entity_mappings()
        if self.resume:
            self._load_existing_embeddings()
            self._load_metadata()
            
        # Check if rebuild is needed
        if not force_rebuild and self._needs_rebuild():
            logger.info("🔄 Source files changed - full rebuild required")
            self.resume = False
            self.existing_entities.clear()
        
        # Initialize direct Ollama integration (Apple Metal GPU auto-enabled)
        self.model_name = "mxbai-embed-large:latest"
        logger.info("✅ Direct Ollama integration initialized with mxbai-embed-large:latest")
        logger.info("🚀 Apple Metal GPU acceleration enabled automatically")

    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding using direct Ollama API call"""
        try:
            # Use Ollama API to create embedding
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': self.model_name,
                    'prompt': text
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result['embedding']
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Embedding creation failed for '{text}': {e}")
            return None

    def _load_entity_mappings(self):
        """Load entity to ID mappings from KG model"""
        entity_file = self.source_files['entities']
        if not entity_file.exists():
            raise FileNotFoundError(f"Entity mapping file not found: {entity_file}")
        
        with open(entity_file, 'r', encoding='utf-8') as f:
            self.entity_to_id = json.load(f)
        
        logger.info(f"Loaded {len(self.entity_to_id)} entities from KG")

    def _load_existing_embeddings(self):
        """Load existing embeddings to support resume"""
        if not self.output_file.exists():
            return
            
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.existing_entities.add(entry['title'])
            
            logger.info(f"Found {len(self.existing_entities)} existing embeddings")
        except Exception as e:
            logger.warning(f"Could not load existing embeddings: {e}")
            self.existing_entities.clear()

    def _load_metadata(self):
        """Load processing metadata"""
        if not self.metadata_file.exists():
            return
            
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            self.metadata = {}

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        if not file_path.exists():
            return ""
            
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _needs_rebuild(self) -> bool:
        """Check if source files have changed since last run"""
        if not self.metadata.get('source_file_hashes'):
            return True
            
        current_hashes = {}
        for name, path in self.source_files.items():
            current_hashes[name] = self._calculate_file_hash(path)
            
        stored_hashes = self.metadata.get('source_file_hashes', {})
        
        for name, current_hash in current_hashes.items():
            if stored_hashes.get(name) != current_hash:
                logger.info(f"📝 Source file changed: {name}")
                return True
                
        return False

    def _save_metadata(self):
        """Save processing metadata with source file hashes"""
        # Calculate current source file hashes
        source_hashes = {}
        for name, path in self.source_files.items():
            source_hashes[name] = self._calculate_file_hash(path)
        
        metadata = {
            'last_update': time.time(),
            'last_update_iso': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'total_entities': len(self.entity_to_id),
            'processed_entities': self.processed_count,
            'failed_entities': self.failed_count,
            'skipped_entities': self.skipped_count,
            'source_file_hashes': source_hashes,
            'max_workers': self.max_workers,
            'version': '2.0-enhanced'
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _process_entity(self, entity_name: str) -> Dict[str, Any]:
        """Process a single entity and create its embedding"""
        try:
            # Create embedding using mxbai-embed-large
            embedding = self._create_embedding(entity_name)
            if embedding is None:
                with self.lock:
                    self.failed_count += 1
                return None
            
            # Create entry in same format as wiki embeddings
            entry = {
                'title': entity_name,
                'source': 'kg_entity',
                'embedding': embedding,
                'entity_id': self.entity_to_id.get(entity_name),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            }
            
            # Update counters thread-safely
            with self.lock:
                self.processed_count += 1
                
            # Log progress every 50 entities
            if self.processed_count % 50 == 0:
                progress_pct = (self.processed_count / len(self.entity_to_id)) * 100
                logger.info(f"Processed {self.processed_count}/{len(self.entity_to_id)} entities ({progress_pct:.1f}%)")
                print(f"Processed {self.processed_count}/{len(self.entity_to_id)} entities ({progress_pct:.1f}%)", flush=True)
            
            return entry
            
        except Exception as e:
            with self.lock:
                self.failed_count += 1
            logger.error(f"Error processing entity '{entity_name}': {e}")
            return None

    def create_entity_embeddings(self):
        """Create mxbai embeddings for all KG entities with resume support"""
        if not self.entity_to_id:
            raise ValueError("No entities loaded")

        entities = list(self.entity_to_id.keys())
        total_entities = len(entities)
        
        # Filter out existing entities if resuming
        if self.resume and self.existing_entities:
            entities_to_process = [e for e in entities if e not in self.existing_entities]
            self.skipped_count = len(self.existing_entities)
            logger.info(f"📋 Resume mode: Processing {len(entities_to_process)}/{total_entities} entities (skipping {self.skipped_count} existing)")
        else:
            entities_to_process = entities
            logger.info(f"📋 Full rebuild: Processing all {len(entities_to_process)} entities")

        if not entities_to_process:
            logger.info("✅ All entities already processed!")
            return self.processed_count

        print(f"🚀 Creating embeddings for {len(entities_to_process):,} entities using {self.max_workers} parallel workers")
        
        # Open file in append mode if resuming, write mode if rebuilding
        file_mode = 'a' if (self.resume and self.existing_entities) else 'w'
        
        # Process entities in parallel with incremental writing
        with open(self.output_file, file_mode, encoding='utf-8') as f:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_entity = {executor.submit(self._process_entity, entity): entity
                                  for entity in entities_to_process}

                # Process completed tasks and write immediately
                for future in as_completed(future_to_entity):
                    entity_name = future_to_entity[future]
                    try:
                        entry = future.result()
                        if entry:
                            # Write to JSONL file immediately (thread-safe since we're in main thread)
                            f.write(json.dumps(entry) + '\n')
                            f.flush()  # Ensure data is written to disk
                    except Exception as e:
                        logger.error(f"Future failed for entity '{entity_name}': {e}")
                        with self.lock:
                            self.failed_count += 1

        # Save metadata
        self._save_metadata()

        final_msg = f"✅ Created mxbai embeddings for {self.processed_count}/{total_entities} entities"
        if self.skipped_count > 0:
            final_msg += f" (skipped {self.skipped_count} existing)"
        if self.failed_count > 0:
            final_msg += f" ({self.failed_count} failed)"
            
        logger.info(final_msg)
        print(final_msg, flush=True)

        file_msg = f"📁 Saved to: {self.output_file}"
        logger.info(file_msg)
        print(file_msg, flush=True)

        return self.processed_count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create enhanced mxbai KG embeddings')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--no-resume', action='store_true', help='Disable resume capability')
    parser.add_argument('--force-rebuild', action='store_true', help='Force full rebuild even if no changes detected')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced mxbai-embed-large KG Embeddings Creator")
    print("=" * 60)
    print(f"Workers: {args.workers}")
    print(f"Resume: {not args.no_resume}")
    print(f"Force rebuild: {args.force_rebuild}")
    print()
    
    try:
        creator = EnhancedMXBAIKGEmbeddingCreator(
            max_workers=args.workers,
            resume=not args.no_resume,
            force_rebuild=args.force_rebuild
        )
        
        processed = creator.create_entity_embeddings()
        
        print()
        print(f"🎉 Successfully processed {processed} entities!")
        print("🔗 Unified mxbai embedding space ready for enhanced RAG!")
        
    except Exception as e:
        logger.error(f"❌ Failed to create embeddings: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🧪 SCRIPT STARTING - Enhanced mxbai KG embeddings")
    main()
