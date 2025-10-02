#!/usr/bin/env python3
"""
Build entity → pages mapping from KG triples for incremental embedding updates.

This script reads osrs_kg_triples.csv and creates a mapping of which wiki pages
contribute to each entity. This allows incremental updates - when a page changes,
we only need to re-embed the entities that appear in that page.

Outputs:
- data/kg_entity_to_pages.json: {"entity_name": ["page1", "page2", ...]}
- data/kg_page_to_entities.json: {"page_name": ["entity1", "entity2", ...]}

Usage:
  python3 scripts/kg/build_entity_mapping.py
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRIPLES_CSV = DATA_DIR / "osrs_kg_triples.csv"
ENTITY_TO_PAGES_JSON = DATA_DIR / "kg_entity_to_pages.json"
PAGE_TO_ENTITIES_JSON = DATA_DIR / "kg_page_to_entities.json"


def build_entity_mappings():
    """Build bidirectional mapping between entities and pages"""
    print("🔍 Building entity ↔ pages mapping from KG triples...")
    
    if not TRIPLES_CSV.exists():
        print(f"❌ Triples file not found: {TRIPLES_CSV}")
        return
    
    # Use sets to avoid duplicates
    entity_to_pages: Dict[str, Set[str]] = defaultdict(set)
    page_to_entities: Dict[str, Set[str]] = defaultdict(set)
    
    total_triples = 0
    
    with open(TRIPLES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            head = row.get('head', '').strip()
            tail = row.get('tail', '').strip()
            source_title = row.get('source_title', '').strip()
            
            if not head or not source_title:
                continue
            
            # Head entity appears in this page
            entity_to_pages[head].add(source_title)
            page_to_entities[source_title].add(head)
            
            # Tail entity also appears in this page (if it's not a category/relation)
            if tail and not tail.startswith('Category:'):
                entity_to_pages[tail].add(source_title)
                page_to_entities[source_title].add(tail)
            
            total_triples += 1
            
            if total_triples % 50000 == 0:
                print(f"  Processed {total_triples:,} triples...")
    
    print(f"✅ Processed {total_triples:,} triples")
    print(f"📊 Found {len(entity_to_pages):,} unique entities")
    print(f"📊 Found {len(page_to_entities):,} unique pages")
    
    # Convert sets to sorted lists for JSON serialization
    entity_to_pages_list = {
        entity: sorted(list(pages))
        for entity, pages in entity_to_pages.items()
    }
    
    page_to_entities_list = {
        page: sorted(list(entities))
        for page, entities in page_to_entities.items()
    }
    
    # Save mappings
    print(f"💾 Saving entity → pages mapping to {ENTITY_TO_PAGES_JSON}")
    with open(ENTITY_TO_PAGES_JSON, 'w', encoding='utf-8') as f:
        json.dump(entity_to_pages_list, f, indent=2)
    
    print(f"💾 Saving page → entities mapping to {PAGE_TO_ENTITIES_JSON}")
    with open(PAGE_TO_ENTITIES_JSON, 'w', encoding='utf-8') as f:
        json.dump(page_to_entities_list, f, indent=2)
    
    # Print some statistics
    avg_pages_per_entity = sum(len(pages) for pages in entity_to_pages.values()) / len(entity_to_pages)
    avg_entities_per_page = sum(len(entities) for entities in page_to_entities.values()) / len(page_to_entities)
    
    print(f"\n📈 Statistics:")
    print(f"  Average pages per entity: {avg_pages_per_entity:.1f}")
    print(f"  Average entities per page: {avg_entities_per_page:.1f}")
    
    # Find entities mentioned in most pages
    top_entities = sorted(entity_to_pages.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print(f"\n🔝 Top 10 most-referenced entities:")
    for entity, pages in top_entities:
        print(f"  {entity}: {len(pages)} pages")
    
    print("\n✅ Entity mapping complete!")


if __name__ == "__main__":
    build_entity_mappings()

