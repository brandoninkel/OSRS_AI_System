# KG Incremental Updates Analysis

## Current State

### What Has Revision Tracking:

1. ✅ **Wiki Content** (`osrs_wiki_content.jsonl`)
   - Has `revid` and `timestamp` for each page
   - Can detect which pages changed

2. ✅ **Regular Embeddings** (`osrs_embeddings.jsonl`)
   - Has `metadata.revid` and `metadata.timestamp`
   - Can do incremental updates (only re-embed changed pages)

3. ✅ **KG Triples** (`osrs_kg_triples.csv`)
   - Has `revid` column tracking source page revision
   - Can detect which triples came from changed pages

### What DOESN'T Have Revision Tracking:

4. ❌ **KG Entity Embeddings** (`kg_entity_embeddings_mxbai.jsonl`)
   - Only has: `title`, `text`, `source`, `kg_entity`, `entity_id`, `url`, `embedding`
   - NO `revid` or tracking of which wiki pages contributed to this entity
   - **Currently regenerates ALL 149k entities every time**

5. ❌ **PyKEEN Model** (`data/kg_model/`)
   - Trained on the full set of triples
   - **Needs full retrain when triples change** (this is correct - can't incrementally train)

## The Problem

### Current Behavior:
When a single wiki page changes:
1. ✅ Watchdog detects the change via `revid`
2. ✅ KG triples are regenerated (only for changed pages)
3. ❌ **ALL 149k KG entity embeddings are regenerated** (~18-21 minutes)
4. ✅ PyKEEN model is retrained on full triple set (correct behavior)

### What SHOULD Happen:
When a single wiki page changes:
1. ✅ Watchdog detects the change via `revid`
2. ✅ KG triples are regenerated (only for changed pages)
3. ✅ **Only affected entity embeddings are regenerated** (seconds, not minutes)
4. ✅ PyKEEN model is retrained on full triple set (correct - can't be incremental)

## Why PyKEEN Can't Be Incremental

**PyKEEN training MUST be full retrain** because:
- Knowledge graph embeddings are learned in relation to ALL entities
- Changing one triple affects the embedding space for all entities
- Can't just "update" a few entity embeddings - the whole space shifts
- This is correct behavior and can't be optimized

**However**, PyKEEN training is fast (~2-3 minutes), so this is acceptable.

## Why KG Entity Embeddings CAN Be Incremental

**KG entity embeddings CAN be incremental** because:
- Each entity embedding is independent (just text → vector)
- Entity text is deterministic (same text = same embedding)
- Only entities from changed pages need re-embedding
- Most updates affect <1% of entities

### Example:
- Wiki page "Abyssal whip" changes (revid 14769106 → 14769200)
- This affects ~10 entities: "Abyssal whip", "Abyssal demon", "Slayer", etc.
- Currently: Re-embed ALL 149k entities (~18 min)
- Should: Re-embed only 10 entities (~1 second)

## Solution Design

### Step 1: Add Revision Tracking to KG Entity Embeddings

Modify `kg_entity_embeddings_mxbai.jsonl` format:
```json
{
  "title": "Abyssal whip",
  "text": "Abyssal whip",
  "source": "knowledge_graph",
  "kg_entity": true,
  "entity_id": 42,
  "url": "https://oldschool.runescape.wiki/w/Abyssal_whip",
  "embedding": [...],
  "metadata": {
    "source_revids": [14769106, 14769200],  // NEW: Which wiki pages contributed
    "created_at": "2025-10-02T11:00:00Z",
    "embedding_model": "mxbai-embed-large:latest"
  }
}
```

### Step 2: Track Entity → Page Mapping

Create `data/kg_entity_to_pages.json`:
```json
{
  "Abyssal whip": ["Abyssal whip", "Slayer", "Abyssal demon"],
  "Slayer": ["Slayer", "Combat", "Skills"],
  ...
}
```

This maps each entity to the wiki pages that mention it.

### Step 3: Incremental Update Logic

```python
def update_kg_embeddings_incremental(changed_pages_with_revids):
    """
    Only re-embed entities affected by changed pages
    
    Args:
        changed_pages_with_revids: {"Abyssal whip": 14769200, ...}
    """
    # 1. Load entity → pages mapping
    entity_to_pages = load_entity_to_pages_mapping()
    
    # 2. Find affected entities
    affected_entities = set()
    for page in changed_pages_with_revids:
        for entity in entity_to_pages:
            if page in entity_to_pages[entity]:
                affected_entities.add(entity)
    
    # 3. Load existing embeddings
    existing_embeddings = load_kg_embeddings()
    
    # 4. Re-embed only affected entities
    for entity in affected_entities:
        new_embedding = embed_text(entity)
        existing_embeddings[entity] = {
            "embedding": new_embedding,
            "metadata": {
                "source_revids": get_revids_for_entity(entity),
                "updated_at": now()
            }
        }
    
    # 5. Save updated embeddings
    save_kg_embeddings(existing_embeddings)
```

### Step 4: Full Rebuild Trigger

Still need full rebuild when:
- ❌ Embedding model changes (mxbai-embed-large → different model)
- ❌ KG structure changes significantly (new entity extraction logic)
- ❌ Manual request (--force-rebuild flag)
- ✅ Normal wiki updates → incremental only

## Implementation Plan

### Phase 1: Add Tracking (No Behavior Change)
1. Modify `create_osrs_embeddings.py` to add `metadata.source_revids`
2. Build `kg_entity_to_pages.json` during triple generation
3. Test that embeddings still work correctly

### Phase 2: Implement Incremental Logic
1. Add `--incremental` flag to `create_osrs_embeddings.py`
2. Implement affected entity detection
3. Load existing embeddings and update only changed ones
4. Test with small changes

### Phase 3: Integrate with Watchdog
1. Modify `kg_auto_updater.py` to use incremental mode
2. Pass changed page list from watchdog
3. Fall back to full rebuild if needed
4. Monitor and verify correctness

## Expected Performance Improvement

### Current (Full Rebuild):
- **Time**: 18-21 minutes for 149k entities
- **Frequency**: Every wiki update
- **Waste**: 99%+ of entities unchanged

### With Incremental Updates:
- **Time**: 1-10 seconds for typical update (10-100 entities)
- **Frequency**: Every wiki update
- **Waste**: 0% - only changed entities

### Example Scenarios:
| Change | Entities Affected | Current Time | Incremental Time | Speedup |
|--------|-------------------|--------------|------------------|---------|
| 1 page | ~10 entities | 18 min | 1 sec | 1080x |
| 10 pages | ~100 entities | 18 min | 10 sec | 108x |
| 100 pages | ~1000 entities | 18 min | 100 sec | 10.8x |
| Full rebuild | 149k entities | 18 min | 18 min | 1x |

## Summary

**Your intuition is 100% correct!**

1. ✅ **KG entity embeddings SHOULD be incremental** - currently they're not
2. ✅ **PyKEEN training MUST be full retrain** - this is correct and necessary
3. ✅ **We have the data to do incremental** - `revid` is already tracked in triples
4. ❌ **We're not using it** - currently regenerating everything

**Next Steps:**
1. Implement revision tracking in KG entity embeddings
2. Build entity → pages mapping
3. Implement incremental update logic
4. Integrate with watchdog system

This will reduce typical update time from **18 minutes → 1-10 seconds** (100-1000x speedup)!

