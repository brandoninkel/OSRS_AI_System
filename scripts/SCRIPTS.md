# OSRS AI System - Scripts Documentation

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Core Scripts](#core-scripts)
4. [Knowledge Graph Scripts](#knowledge-graph-scripts)
5. [Command Files](#command-files)
6. [Data Files Interaction](#data-files-interaction)
7. [Usage Guide](#usage-guide)
8. [Maintenance](#maintenance)

---

## Overview

The `scripts/` directory contains data processing and maintenance scripts for the OSRS AI system. These scripts handle:

- **Wiki Content Processing**: Fetching and parsing OSRS wiki pages
- **Embedding Generation**: Creating semantic embeddings for wiki content
- **Knowledge Graph**: Building and maintaining the OSRS knowledge graph
- **Data Maintenance**: Updating, validating, and optimizing data files

**Key Principle**: Scripts are run manually or via cron jobs. The API server does NOT depend on these scripts at runtime.

---

## Directory Structure

```
scripts/
├── create_osrs_embeddings.py          # Generate wiki embeddings
├── streamlined-watchdog.js            # Monitor wiki for changes
├── populate_price_history.py          # Populate price history database
├── knowledge-graph.command            # Build knowledge graph
├── train-kg-embeddings.command        # Train KG embeddings
├── eval-kg-embeddings.command         # Evaluate KG embeddings
├── kg-status.command                  # Check KG status
├── kg/                                # Knowledge graph scripts
│   ├── build_kg.py                    # Build KG from wiki data
│   ├── build_entity_mapping.py        # Create entity mappings
│   ├── create_mxbai_kg_embeddings.py  # Generate KG embeddings
│   ├── create_mxbai_kg_embeddings_enhanced.py  # Enhanced KG embeddings
│   ├── train_kg_embeddings.py         # Train KG with PyKEEN
│   ├── eval_kg_embeddings.py          # Evaluate KG quality
│   └── old/                           # Archived KG scripts
└── old/                               # Archived scripts
    ├── orchestrator.py                # Old orchestration system
    ├── test_*.py                      # Test scripts
    └── ...                            # Other archived files
```

---

## Core Scripts

### 1. streamlined-watchdog.js
**Purpose**: Monitor OSRS wiki for changes and maintain up-to-date content

**What It Does**:
- Fetches pages from OSRS wiki (Main and Guide namespaces)
- Detects new pages and updates to existing pages
- Parses wikitext and extracts structured content
- Maintains revision tracking to avoid duplicate fetches
- Alphabetizes output files for consistency

**Data Files**:
- **Reads**: None (fetches from wiki API)
- **Writes**:
  - `data/osrs_wiki_content.jsonl` (170MB) - Parsed wiki content
  - `data/osrs_wikitext_content.jsonl` (101MB) - Raw wikitext
  - `data/osrs_page_titles.txt` (644KB) - List of all page titles
  - `data/osrs_filtered_pages.txt` (5KB) - Filtered out pages
  - `data/osrs_null_pages.txt` (11KB) - Pages with no content
  - `data/osrs_template_issues.jsonl` (78MB) - Template parsing issues
  - `data/osrs_watchdog_tracking.json` - Tracking metadata

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
node streamlined-watchdog.js
```

**Options**:
- `--full-refetch` - Refetch all pages (ignores existing data)
- `--batch-fetch` - Use batch fetching (faster)
- `--batch-size N` - Set batch size (default: 50)
- `--test-batch N` - Only process N batches for testing
- `--skip-reprocess` - Skip reprocessing existing pages
- `--skip-checker` - Skip template checker

**When to Run**:
- Weekly: Check for wiki updates
- After major game updates: Capture new content
- When data seems stale: Full refetch

**Performance**:
- Full run: ~2-4 hours (35,000+ pages)
- Incremental: ~10-30 minutes (only changed pages)

---

### 2. create_osrs_embeddings.py
**Purpose**: Generate semantic embeddings for wiki content using mxbai-embed-large

**What It Does**:
- Loads parsed wiki content from JSONL
- Prepares text with title, categories, and content
- Creates 1024-dimensional embeddings using Ollama
- Supports incremental updates (only embeds new/changed pages)
- Writes embeddings in streaming fashion

**Data Files**:
- **Reads**:
  - `data/osrs_wiki_content.jsonl` (170MB) - Source content
- **Writes**:
  - `data/osrs_embeddings.jsonl` (844MB) - Wiki embeddings

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
python3 create_osrs_embeddings.py
```

**Options**:
- `--full` - Regenerate all embeddings (ignores existing)
- `--progress-mode` - Show progress for orchestration
- `--async` - Use async embedding (faster)
- `--chunk-size N` - Process N pages at a time (default: 200)

**When to Run**:
- After running watchdog with changes
- When wiki content is updated
- When switching embedding models

**Performance**:
- Full run: ~2-3 hours (35,000+ pages)
- Incremental: ~5-20 minutes (only new pages)
- Memory: ~2GB RAM
- Requires: Ollama running with mxbai-embed-large

**Incremental Updates**:
The script automatically detects which pages need embedding by:
1. Building index of existing embeddings (title + revid)
2. Comparing with wiki content
3. Only embedding pages not in index

---

### 3. populate_price_history.py
**Purpose**: Populate price history database with historical data

**What It Does**:
- Fetches current prices for popular OSRS items
- Populates the price_history.db SQLite database
- Useful for initializing the Economic Dashboard with data
- Can be run periodically to build historical data

**Data Files**:
- **Reads**: None (fetches from OSRS Wiki API)
- **Writes**:
  - `data/price_history.db` (SQLite) - Price history database

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
python3 populate_price_history.py
```

**Options**:
- `--items` - Comma-separated list of items to fetch (default: popular items)
- `--count N` - Number of items to fetch (default: 50)

**When to Run**:
- After setting up the system for the first time
- To build historical data for the Economic Dashboard
- Periodically (e.g., daily) to track price trends

**Performance**:
- Full run: ~5-10 minutes (50 items)
- Memory: ~100MB RAM
- Requires: API server running

**Example**:
```bash
# Populate with default popular items
python3 populate_price_history.py

# Populate specific items
python3 populate_price_history.py --items "Abyssal whip,Dragon scimitar,Bandos chestplate"

# Populate top 100 items
python3 populate_price_history.py --count 100
```

---

## Knowledge Graph Scripts

### 1. kg/build_kg.py
**Purpose**: Build knowledge graph from wiki content

**What It Does**:
- Extracts entities from wiki pages
- Identifies relationships between entities
- Parses internal links and infobox data
- Creates triples (head, relation, tail)
- Generates node and edge files

**Data Files**:
- **Reads**:
  - `data/osrs_wiki_content.jsonl` (170MB) - Parsed content
  - `data/osrs_wikitext_content.jsonl` (101MB) - Raw wikitext
- **Writes**:
  - `data/osrs_kg_triples.csv` (93MB) - KG triples
  - `data/osrs_kg_nodes.jsonl` (6.1MB) - Entity nodes
  - `data/osrs_kg_edges.jsonl` (179MB) - Relationships
  - `data/osrs_kg.meta.json` - Build metadata

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts/kg
python3 build_kg.py --snapshot
```

**Options**:
- `--snapshot` - Create snapshot of input files (safer)
- `--max-pages N` - Limit to N pages (for testing)
- `--workers N` - Use N parallel workers

**When to Run**:
- After major wiki updates
- When adding new entity types
- Monthly: Refresh KG structure

**Performance**:
- Full run: ~30-60 minutes
- Memory: ~4GB RAM
- CPU: Uses multiple cores

---

### 2. kg/build_entity_mapping.py
**Purpose**: Create entity-to-ID mappings for KG

**What It Does**:
- Scans KG nodes
- Creates bidirectional mappings
- Generates entity_to_id.json and id_to_entity.json

**Data Files**:
- **Reads**:
  - `data/osrs_kg_nodes.jsonl` (6.1MB)
- **Writes**:
  - `data/kg_model/entity_to_id.json`
  - `data/kg_model/id_to_entity.json`

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts/kg
python3 build_entity_mapping.py
```

**When to Run**:
- After building KG
- Before training KG embeddings

---

### 3. kg/create_mxbai_kg_embeddings.py
**Purpose**: Generate embeddings for KG entities using mxbai-embed-large

**What It Does**:
- Loads entity mappings
- Creates embeddings for each entity
- Uses same model as wiki embeddings (unified space)
- Supports parallel processing

**Data Files**:
- **Reads**:
  - `data/kg_model/entity_to_id.json`
- **Writes**:
  - `data/kg_entity_embeddings_mxbai.jsonl` (2.0GB)

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts/kg
python3 create_mxbai_kg_embeddings.py
```

**Options**:
- `--max-workers N` - Use N parallel workers (default: 128)
- `--batch-size N` - Batch size (default: 200)

**When to Run**:
- After building entity mappings
- When updating KG structure

**Performance**:
- Full run: ~1-2 hours (149,000+ entities)
- Memory: ~3GB RAM
- Requires: Ollama running

---

### 4. kg/create_mxbai_kg_embeddings_enhanced.py
**Purpose**: Enhanced KG embeddings with context

**What It Does**:
- Similar to basic version
- Includes entity context from wiki pages
- Better quality embeddings

**Usage**: Same as basic version

---

### 5. kg/train_kg_embeddings.py
**Purpose**: Train KG embeddings using PyKEEN (TransE model)

**What It Does**:
- Loads KG triples
- Trains TransE model
- Generates entity and relation embeddings
- Evaluates model quality

**Data Files**:
- **Reads**:
  - `data/osrs_kg_triples.csv` (93MB)
- **Writes**:
  - `data/kg_model/entity_embeddings.npy`
  - `data/kg_model/relation_embeddings.npy`
  - `data/kg_model/model_checkpoint.pt`

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts/kg
python3 train_kg_embeddings.py --resume
```

**Options**:
- `--resume` - Resume from checkpoint
- `--epochs N` - Train for N epochs (default: 100)
- `--backend pykeen` - Use PyKEEN backend
- `--backend auto` - Auto-select backend

**When to Run**:
- After building KG
- When improving KG quality

**Performance**:
- Full run: ~2-4 hours
- Memory: ~8GB RAM
- Requires: PyKEEN installed in .kg-venv

---

### 6. kg/eval_kg_embeddings.py
**Purpose**: Evaluate KG embedding quality

**What It Does**:
- Tests link prediction accuracy
- Measures embedding quality
- Generates evaluation metrics

**Data Files**:
- **Reads**:
  - `data/kg_entity_embeddings_mxbai.jsonl` (2.0GB)
  - `data/osrs_kg_triples.csv` (93MB)

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts/kg
python3 eval_kg_embeddings.py
```

**When to Run**:
- After generating embeddings
- To compare different embedding methods

---

## Command Files

### 1. knowledge-graph.command
**Purpose**: Convenient wrapper to build knowledge graph

**What It Does**:
- Sets up Python environment
- Creates log directory
- Runs build_kg.py with logging
- Captures output to timestamped log file

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
./knowledge-graph.command
```

**Log Location**: `logs/kg/build_YYYYMMDD_HHMMSS.log`

---

### 2. train-kg-embeddings.command
**Purpose**: Convenient wrapper to train KG embeddings

**What It Does**:
- Selects appropriate Python environment (.kg-venv for PyKEEN)
- Runs train_kg_embeddings.py with logging
- Supports all training options

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
./train-kg-embeddings.command --backend pykeen
```

**Log Location**: `logs/kg/train_YYYYMMDD_HHMMSS.log`

---

### 3. eval-kg-embeddings.command
**Purpose**: Convenient wrapper to evaluate KG embeddings

**What It Does**:
- Runs eval_kg_embeddings.py with logging
- Captures evaluation metrics

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
./eval-kg-embeddings.command
```

**Log Location**: `logs/kg/eval_YYYYMMDD_HHMMSS.log`

---

### 4. kg-status.command
**Purpose**: Check status of KG files and processes

**What It Does**:
- Lists KG data files with sizes
- Shows last modification times
- Checks if processes are running

**Usage**:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
./kg-status.command
```

---

## Data Files Interaction

### Input Files (Read-Only)
Scripts read from these files but never modify them:

| File | Size | Used By | Purpose |
|------|------|---------|---------|
| `osrs_wiki_content.jsonl` | 170MB | create_osrs_embeddings.py, build_kg.py | Parsed wiki content |
| `osrs_wikitext_content.jsonl` | 101MB | build_kg.py | Raw wikitext for parsing |
| `osrs_kg_nodes.jsonl` | 6.1MB | build_entity_mapping.py | KG entity nodes |
| `osrs_kg_triples.csv` | 93MB | train_kg_embeddings.py, eval_kg_embeddings.py | KG relationships |

### Output Files (Written)
Scripts generate or update these files:

| File | Size | Generated By | Used By API |
|------|------|--------------|-------------|
| `osrs_embeddings.jsonl` | 844MB | create_osrs_embeddings.py | ✅ Yes (osrs_agentic_rag.py) |
| `kg_entity_embeddings_mxbai.jsonl` | 2.0GB | create_mxbai_kg_embeddings.py | ✅ Yes (osrs_agentic_rag.py) |
| `osrs_kg_edges.jsonl` | 179MB | build_kg.py | ❌ No (future use) |
| `osrs_kg_nodes.jsonl` | 6.1MB | build_kg.py | ❌ No (future use) |
| `osrs_kg_triples.csv` | 93MB | build_kg.py | ❌ No (training only) |

### Tracking Files
Scripts use these for state management:

| File | Purpose |
|------|---------|
| `osrs_page_titles.txt` | List of all wiki page titles |
| `osrs_filtered_pages.txt` | Pages filtered out (redirects, etc.) |
| `osrs_null_pages.txt` | Pages with no content |
| `osrs_watchdog_tracking.json` | Watchdog state and metadata |
| `osrs_template_issues.jsonl` | Template parsing issues log |

---

## Usage Guide

### Complete Data Pipeline

**Step 1: Fetch Wiki Content**
```bash
cd /Users/brandon/Documents/projects/GE/scripts
node streamlined-watchdog.js
```
**Output**: `osrs_wiki_content.jsonl`, `osrs_wikitext_content.jsonl`

---

**Step 2: Generate Wiki Embeddings**
```bash
cd /Users/brandon/Documents/projects/GE/scripts
python3 create_osrs_embeddings.py
```
**Output**: `osrs_embeddings.jsonl`

---

**Step 3: Build Knowledge Graph**
```bash
cd /Users/brandon/Documents/projects/GE/scripts
./knowledge-graph.command
```
**Output**: `osrs_kg_triples.csv`, `osrs_kg_nodes.jsonl`, `osrs_kg_edges.jsonl`

---

**Step 4: Create Entity Mappings**
```bash
cd /Users/brandon/Documents/projects/GE/scripts/kg
python3 build_entity_mapping.py
```
**Output**: `kg_model/entity_to_id.json`, `kg_model/id_to_entity.json`

---

**Step 5: Generate KG Embeddings**
```bash
cd /Users/brandon/Documents/projects/GE/scripts/kg
python3 create_mxbai_kg_embeddings.py
```
**Output**: `kg_entity_embeddings_mxbai.jsonl`

---

**Step 6: (Optional) Train PyKEEN Embeddings**
```bash
cd /Users/brandon/Documents/projects/GE/scripts
./train-kg-embeddings.command --backend pykeen
```
**Output**: `kg_model/entity_embeddings.npy`, `kg_model/relation_embeddings.npy`

---

**Step 7: (Optional) Evaluate KG Quality**
```bash
cd /Users/brandon/Documents/projects/GE/scripts
./eval-kg-embeddings.command
```

---

### Incremental Updates

**Weekly Update Workflow**:
```bash
# 1. Check for wiki changes
cd /Users/brandon/Documents/projects/GE/scripts
node streamlined-watchdog.js

# 2. Update embeddings (only new pages)
python3 create_osrs_embeddings.py

# 3. Restart API server to reload embeddings
cd /Users/brandon/Documents/projects/GE/api
# Kill existing server (Ctrl+C)
python3 osrs_api_server.py --host 0.0.0.0
```

**Note**: KG updates are optional for weekly updates. Only rebuild KG when:
- Major wiki structure changes
- New entity types added
- Monthly maintenance

---

### Quick Commands

**Check Data Status**:
```bash
ls -lh /Users/brandon/Documents/projects/GE/data/*.jsonl
```

**Check Last Update Times**:
```bash
ls -lt /Users/brandon/Documents/projects/GE/data/*.jsonl | head -5
```

**Count Wiki Pages**:
```bash
wc -l /Users/brandon/Documents/projects/GE/data/osrs_wiki_content.jsonl
```

**Count Embeddings**:
```bash
wc -l /Users/brandon/Documents/projects/GE/data/osrs_embeddings.jsonl
```

**Check Ollama Status**:
```bash
ollama list
```

---

## Maintenance

### Regular Maintenance Schedule

**Weekly**:
- Run watchdog to check for wiki updates
- Update embeddings if changes detected
- Restart API server

**Monthly**:
- Full wiki refetch (`--full-refetch`)
- Rebuild knowledge graph
- Regenerate all embeddings
- Evaluate KG quality

**Quarterly**:
- Review and clean up old backups
- Optimize data files
- Update documentation

---

### Backup Strategy

**Before Major Updates**:
```bash
cd /Users/brandon/Documents/projects/GE/data

# Backup embeddings
cp osrs_embeddings.jsonl osrs_embeddings_backup_$(date +%Y%m%d).jsonl

# Backup KG embeddings
cp kg_entity_embeddings_mxbai.jsonl kg_entity_embeddings_mxbai_backup_$(date +%Y%m%d).jsonl

# Backup wiki content
cp osrs_wiki_content.jsonl osrs_wiki_content_backup_$(date +%Y%m%d).jsonl
```

---

### Troubleshooting

**Issue: Watchdog fails to fetch pages**
```bash
# Check internet connection
ping oldschool.runescape.wiki

# Check User-Agent is set correctly
# Edit streamlined-watchdog.js line 24
```

**Issue: Embedding generation fails**
```bash
# Check Ollama is running
ollama list

# Check model is available
ollama pull mxbai-embed-large:latest

# Check memory usage
free -h  # Linux
vm_stat  # macOS
```

**Issue: KG build fails**
```bash
# Check input files exist
ls -lh data/osrs_wiki_content.jsonl data/osrs_wikitext_content.jsonl

# Check disk space
df -h

# Run with limited pages for testing
python3 scripts/kg/build_kg.py --max-pages 100
```

**Issue: Out of memory**
```bash
# Reduce batch size
python3 create_osrs_embeddings.py --chunk-size 50

# Or reduce workers
python3 kg/create_mxbai_kg_embeddings.py --max-workers 32
```

---

### Performance Optimization

**Faster Watchdog**:
```bash
# Use batch fetching
node streamlined-watchdog.js --batch-fetch --batch-size 100
```

**Faster Embeddings**:
```bash
# Use async mode
python3 create_osrs_embeddings.py --async --chunk-size 500
```

**Faster KG Embeddings**:
```bash
# Increase workers (if you have RAM)
python3 kg/create_mxbai_kg_embeddings.py --max-workers 256
```

---

## Archived Scripts (scripts/old/)

These scripts are no longer actively used but kept for reference:

- **orchestrator.py** - Old orchestration system (replaced by manual workflow)
- **memory_manager.py** - Memory management utilities (no longer needed)
- **incremental_kg_updater.py** - Incremental KG updates (replaced by full rebuild)
- **kg_auto_updater.py** - Automatic KG updates (replaced by manual workflow)
- **deduplicate_data.py** - Data deduplication (no longer needed)
- **repair_jsonl_with_parser.py** - JSONL repair utility (no longer needed)
- **test_*.py** - Test scripts for development

---

## Dependencies

### Python Dependencies
```bash
# Core dependencies (in main venv)
pip install requests aiohttp numpy

# KG dependencies (in .kg-venv)
pip install pykeen torch
```

### Node.js Dependencies
```bash
cd /Users/brandon/Documents/projects/GE/scripts
npm install fs-extra axios chalk ora
```

### System Dependencies
- **Ollama**: Required for embedding generation
- **Node.js 18+**: Required for watchdog
- **Python 3.10+**: Required for all Python scripts

---

## Credits

**Developed by**: Brandon Inkel
**Data Source**: Old School RuneScape Wiki
**Embedding Model**: mxbai-embed-large by MixedBread
**KG Framework**: PyKEEN

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Verify data files exist and are not corrupted
3. Check Ollama is running
4. Review this documentation
5. Check disk space and memory

---

**Last Updated**: October 5, 2025
**Version**: 1.1.0

