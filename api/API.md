# OSRS Agentic RAG API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [System Requirements](#system-requirements)
4. [Installation & Setup](#installation--setup)
5. [Running the System](#running-the-system)
6. [API Endpoints](#api-endpoints)
7. [Core Components](#core-components)
8. [Data Files](#data-files)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The OSRS Agentic RAG (Retrieval-Augmented Generation) system is an AI-powered assistant for Old School RuneScape that uses:
- **LangGraph** for agentic workflow orchestration
- **gpt-oss:20b** (OpenAI's open-source model) for natural language understanding and generation
- **mxbai-embed-large** for semantic search embeddings
- **MediaWiki API** for contributor attribution with revision caching
- **Flask** for HTTP API server
- **React** for web-based GUI

The system provides intelligent, context-aware answers about OSRS by:
1. Using an AI agent to decide what information to search for
2. Searching a comprehensive wiki knowledge base (35,884 pages)
3. Leveraging a knowledge graph (149,047 entities)
4. Generating answers with proper citations and attributions
5. Tracking wiki contributors for transparency

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (React)                     │
│                    http://localhost:3005                     │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────┐
│                   Flask API Server                           │
│                  (osrs_api_server.py)                        │
│                  http://localhost:5001                       │
└─────┬──────────────────────┬──────────────────────┬─────────┘
      │                      │                      │
      ▼                      ▼                      ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Agentic    │    │   Attribution    │    │  Embedding   │
│  RAG V3     │    │    Service       │    │   Service    │
│ (LangGraph) │    │  (MediaWiki API) │    │   (Ollama)   │
└──────┬──────┘    └──────────────────┘    └──────┬───────┘
       │                                           │
       ▼                                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Ollama Server                           │
│                  http://localhost:11434                      │
│  Models: gpt-oss:20b, mxbai-embed-large:latest              │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Files                              │
│  - osrs_embeddings.jsonl (844MB)                            │
│  - osrs_wiki_content.jsonl (170MB)                          │
│  - kg_entity_embeddings_mxbai.jsonl (2.0GB)                 │
│  - osrs_kg_nodes.jsonl (6.1MB)                              │
│  - osrs_kg_edges.jsonl (179MB)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## System Requirements

### Hardware
- **Minimum**: 16GB RAM, Apple Silicon M1 or equivalent
- **Recommended**: 24GB+ RAM, Apple Silicon M4 Pro or better
- **Storage**: 5GB+ free space for data files and models

### Software
- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher (for frontend)
- **Ollama**: Latest version
- **Operating System**: macOS (tested), Linux (should work), Windows (untested)

### Required Models (Ollama)
```bash
ollama pull gpt-oss:20b
ollama pull mxbai-embed-large:latest
```

---

## Directory Structure

```
api/
├── API.md                              # This documentation (1100+ lines)
├── QUICKSTART.md                       # Quick start guide (200 lines)
├── osrs_api_server.py                  # ✅ Flask API server (414 lines)
├── osrs_agentic_rag.py                 # ✅ V3 LangGraph RAG (1074 lines)
├── attribution_service.py              # ✅ Contributor lookup (301 lines)
├── attribution_service_cached.py       # ✅ Cached attribution (500+ lines)
├── revision_cache.py                   # ✅ Revision caching (300+ lines)
├── citation_tool.py                    # ✅ Citation generation (300 lines)
├── citation_injector.py                # ✅ Auto citation injection (300 lines)
├── price_history.py                    # ✅ Price tracking (212 lines)
├── api_queue_manager.py                # ✅ API rate limiting (200+ lines)
├── start-gui.command                   # ✅ Start all services + GUI
├── start-services-only.command         # ✅ Start core services only
├── start-data.command                  # ✅ Start data pipeline
├── stop-all.command                    # ✅ Stop all services
├── stop-data.command                   # ✅ Stop data pipeline
├── embeddings/
│   └── embedding_service.py            # ✅ Ollama embeddings (373 lines)
└── old/                                # Archived files
    ├── osrs_rag_service.py             # ❌ Legacy V2 RAG
    ├── osrs_rag_service_v1.py          # ❌ Legacy V1 RAG
    ├── kg_query_service.py             # ❌ Unused KG queries
    ├── query_segmentation.py           # ❌ Legacy query analysis
    ├── wiki_template_parser.py         # ❌ Preprocessing only
    ├── wiki_template_checker.py        # ❌ Preprocessing only
    ├── reranker_service.py             # ❌ Legacy reranker
    └── tests/                          # ❌ Old test files
```

**Active Files**: 15 Python files + 5 command files = 20 files
**Archived Files**: 7+ files in `api/old/`

---

## Installation & Setup

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

### 2. Pull Required Models
```bash
ollama pull gpt-oss:20b
ollama pull mxbai-embed-large:latest
```

### 3. Install Python Dependencies
```bash
cd /Users/brandon/Documents/projects/GE/api
pip3 install -r requirements.txt
```

Required packages:
- `flask` - Web server
- `flask-cors` - CORS support
- `langchain-ollama` - Ollama integration
- `langchain-core` - LangChain core
- `langgraph` - Agentic workflow framework
- `numpy` - Numerical operations
- `requests` - HTTP client

### 4. Install Frontend Dependencies
```bash
cd /Users/brandon/Documents/projects/GE/frontend
npm install
```

### 5. Verify Data Files
Ensure these files exist in `/Users/brandon/Documents/projects/GE/data/`:
- `osrs_embeddings.jsonl` (844MB) - Wiki page embeddings
- `osrs_wiki_content.jsonl` (170MB) - Parsed wiki content
- `kg_entity_embeddings_mxbai.jsonl` (2.0GB) - Knowledge graph embeddings
- `osrs_kg_nodes.jsonl` (6.1MB) - KG entity nodes
- `osrs_kg_edges.jsonl` (179MB) - KG relationships

---

## Running the System

### Quick Start (All Services)

**Terminal 1: Start Ollama**
```bash
ollama serve
```

**Terminal 2: Start API Server**
```bash
cd /Users/brandon/Documents/projects/GE/api
python3 osrs_api_server.py --host 0.0.0.0
```

**Terminal 3: Start Frontend**
```bash
cd /Users/brandon/Documents/projects/GE/frontend
npm run dev
```

**Access the GUI**: http://localhost:3005

### Individual Components

#### API Server Only
```bash
cd /Users/brandon/Documents/projects/GE/api
python3 osrs_api_server.py --host localhost --port 5001
```

Options:
- `--host`: Host to bind to (default: localhost)
- `--port`: Port to bind to (default: 5001)
- `--debug`: Enable debug mode

#### Frontend Only
```bash
cd /Users/brandon/Documents/projects/GE/frontend
npm run dev
```

The frontend will be available at http://localhost:3005

---

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "OSRS RAG API",
  "version": "3.0",
  "timestamp": "2025-10-03T05:00:00.000000"
}
```

### Chat (Main Query Endpoint)
```http
POST /chat
Content-Type: application/json

{
  "query": "What is Zulrah's combat level?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Zulrah has a combat level of 725.",
  "sources": [
    {
      "title": "Zulrah",
      "url": "https://oldschool.runescape.wiki/w/Zulrah",
      "relevance": "0.95",
      "excerpt": "Zulrah is a level 725 solo-only snake boss..."
    }
  ],
  "reasoning": [
    "🔍 Called get_full_wiki_page with: {'title': 'Zulrah'}"
  ],
  "tool_calls": [
    {
      "tool": "get_full_wiki_page",
      "args": {"title": "Zulrah"}
    }
  ],
  "citations": [
    {
      "text": "Zulrah has a combat level of 725",
      "start": 0,
      "end": 33,
      "source_title": "Zulrah",
      "source_text": "Combat Level: 725"
    }
  ],
  "timestamp": "2025-10-03T05:00:00.000000"
}
```

### Attributions (Get Contributors)
```http
POST /attributions
Content-Type: application/json

{
  "citations": [
    {
      "text": "Zulrah has a combat level of 725",
      "start": 0,
      "end": 33,
      "source_title": "Zulrah",
      "source_text": "Combat Level: 725"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "attributions": [
    {
      "text": "Zulrah has a combat level of 725",
      "start": 0,
      "end": 33,
      "source_title": "Zulrah",
      "source_url": "https://oldschool.runescape.wiki/w/Zulrah",
      "excerpt": "Combat Level: 725",
      "author": "Microbrews",
      "timestamp": "2024-01-15T10:30:00Z",
      "revision_url": "https://oldschool.runescape.wiki/w/index.php?title=Zulrah&oldid=14997277",
      "is_original_author": true,
      "comment": "Updated combat stats",
      "revision_id": 14997277
    }
  ],
  "timestamp": "2025-10-03T05:00:00.000000"
}
```

**Attribution Fields:**
- `text`: The paraphrased text in the AI response
- `start`/`end`: Character positions in the response
- `source_title`: Wiki page title
- `source_url`: Link to the wiki page
- `excerpt`: Exact text from the wiki that was cited
- `author`: Username of the contributor who added/edited this text
- `timestamp`: ISO 8601 timestamp of the revision
- `revision_url`: Direct link to the specific revision
- `is_original_author`: Boolean indicating if this user originally added the text (always true with new system)
- `comment`: Edit summary provided by the contributor
- `revision_id`: Numeric revision ID
- `section`: Page section where the text was added (e.g., "Infobox", "Drops", "Strategy")
- `line_number`: Line number in the page where the text appears
- `context`: Array of surrounding lines for context (up to 5 lines)

### Search (Legacy Endpoint)
```http
POST /search
Content-Type: application/json

{
  "query": "dragon slayer quest"
}
```

**Response:**
```json
{
  "success": true,
  "results": [...],
  "query": "dragon slayer quest",
  "total_results": 10,
  "timestamp": "2025-10-03T05:00:00.000000"
}
```

---

## Core Components

### Active Components

These are the core files actively used by the V3 Agentic RAG system:

---

### 1. osrs_api_server.py
**Purpose**: Flask HTTP API server that exposes the RAG system to the frontend

**Status**: ✅ **ACTIVE** - Main API server

**Key Features**:
- CORS-enabled for cross-origin requests
- Health check endpoint
- Chat endpoint for queries
- Attribution endpoint for contributor lookup
- Error handling and logging

**Dependencies**:
- `osrs_agentic_rag.py` - Main RAG service
- `attribution_service.py` - Wiki contributor lookup

**Initialization**:
```python
server = OSRSAPIServer(host='0.0.0.0', port=5001)
server.run()
```

---

### 2. osrs_agentic_rag.py
**Purpose**: V3 Agentic RAG implementation using LangGraph

**Status**: ✅ **ACTIVE** - Core RAG engine

**Key Features**:
- **LangGraph State Machine**: Orchestrates agent workflow
- **Tool-Based Architecture**: Agent decides which tools to call
- **Multi-Hop Reasoning**: Can make multiple searches if needed
- **Citation Generation**: AI includes citation markers in responses
- **Organic Behavior**: No hardcoded patterns or fallbacks

**Tools Available**:
1. `search_osrs_wiki(query)` - Semantic search across wiki pages
2. `get_full_wiki_page(title)` - Retrieve complete page content

**Agent Workflow**:
```
START → Agent (decides action) → Tools (if needed) → Agent (generates answer) → END
```

**System Prompt Highlights**:
- Forces tool calling before answering
- Assumes first version for numbered content (e.g., "Dragon Slayer" = "Dragon Slayer I")
- Requires citation markers: `[CITE:source="Page"|text="exact text"]paraphrased[/CITE]`
- Explicit citation placement rules with WRONG/RIGHT examples
- Understands game mechanics (e.g., Zulrah combat styles)

**Configuration**:
- Model: `gpt-oss:20b` (OpenAI's open-source model for agentic tasks)
- Temperature: 0.1 (deterministic)
- Context Window: 32,768 tokens
- Embedding Model: `mxbai-embed-large:latest`
- Recursion Limit: 50 (increased from 25 for complex queries)

---

### 3. attribution_service.py
**Purpose**: Finds wiki contributors for specific text snippets with location tracking

**Status**: ✅ **ACTIVE** - Enhanced attribution system with complete page change tracking

**Key Features**:
- **Complete Page Tracking**: Tracks changes to ENTIRE page content (not just infobox fields)
  - Works for: infobox fields, drop tables, strategy sections, trivia, article text, links, images, **anywhere on the page**
- **Diff-Based Change Tracking**: Computes line-by-line diffs between consecutive revisions
- **Location Information**: Provides section, line number, and context (up to 5 surrounding lines)
- **Incremental Updates**:
  - Checks for NEW revisions when attribution is requested
  - Continues where it left off (doesn't re-fetch old revisions)
  - Builds cache over time automatically
- **Persistent JSONL Cache**: Stores revision data in `data/cache/wiki_revisions.jsonl`
- **Concurrent Revision Fetching**: Fetches up to 10 revisions concurrently (10x faster than serial)
- **Optimized API Usage**:
  - Concurrent requests with semaphore rate limiting (10 concurrent max)
  - GZip compression
  - maxlag parameter for non-interactive tasks
  - Respects MediaWiki API limits (200 requests/second allowed)
- **Improved Pattern Matching**:
  - Maps common terms to wiki field names (e.g., "Attack Speed" → "speed")
  - Tries multiple patterns before falling back to bare value
  - Only uses bare value if >3 characters to avoid false matches

**Main Method**:
```python
find_attribution(page_title: str, snippet: str) -> Dict[str, Any]
```

**Returns**:
- `found`: bool - Whether attribution was found
- `author`: str - Contributor name
- `timestamp`: str - ISO format date/time
- `revision_id`: int - Revision ID
- `comment`: str - Edit comment
- `snippet`: str - Exact text that was added
- `section`: str - Page section where it was added (e.g., "Infobox", "Drops", "Strategy")
- `line_number`: int - Line number in the page
- `context`: list - Surrounding lines for context
- `wiki_url`: str - Direct link to the revision

**API Endpoints Used**:
- `/rest.php/v1/page/{title}/history` - Get revision list (50 per request)
- `/rest.php/v1/revision/{id}` - Get revision content with source

**Cache Location**: `data/cache/wiki_revisions.jsonl`

**How It Works**:
1. **First Query** (slow - fetches from API):
   - Fetches complete revision history for the page
   - For each revision: fetch full wikitext, compute diff from previous revision
   - Stores: what changed, where, section, line number, context
   - Saves to JSONL cache
2. **Subsequent Queries** (fast - uses cache):
   - Checks for NEW revisions since last cache
   - Only fetches new revisions (old revisions never change)
   - Searches cache for text
   - Returns attribution instantly
3. **Incremental Updates**:
   - Every time "Show Attributions" is pressed, checks for updates
   - Cache stays up to date automatically

**Cache Format** (JSONL):
```json
{
  "page_title": "Zulrah",
  "revision_id": 14997277,
  "timestamp": "2025-10-01T00:35:11Z",
  "author": "Microbrews",
  "comment": "/* Shark drop table */",
  "changes": {
    "added_changes": [
      {
        "line": "''There is a 12/249 chance of rolling the [[shark drop table]].''",
        "line_number": 217,
        "section": "Shark drop table",
        "context": ["===Shark drop table===", "...", "..."]
      }
    ],
    "added_count": 1,
    "removed_count": 1
  }
}
```

**Performance**:
- First query: Slow (fetches complete history)
- Subsequent queries: Fast (uses cache + incremental updates)
- Updates: Only fetches NEW revisions (checks for updates automatically)
- Storage: ~0.5 KB per revision
- Example: Zulrah (1056 revisions) = ~528 KB

**Cache Management**:
- File is sorted by: page_title (alphabetically), then timestamp (newest first)
- Automatic deduplication on write
- Manual sort/dedupe: `RevisionCache().deduplicate_and_sort()`

---

### 4. citation_tool.py
**Purpose**: Generates properly formatted citations from structured wiki data

**Status**: ✅ **ACTIVE** - Citation generation system

**Key Features**:
- **Loads Parsed Wiki Content**: Reads `osrs_wiki_content.jsonl` into memory (32,677 pages)
- **Field Value Lookup**: Finds exact values from infobox fields (combat, hitpoints, etc.)
- **Text Snippet Search**: Searches for text snippets in page content
- **Automatic Formatting**: Generates proper `[CITE:source="..."|text="..."]...[/CITE]` tags
- **Attribution Integration**: Works with attribution service for contributor tracking

**Main Method**:
```python
create_citation(page_title: str, field_or_text: str, paraphrased_text: str) -> Dict[str, Any]
```

**Example**:
```python
citation_tool.create_citation(
    page_title="Zulrah",
    field_or_text="combat",
    paraphrased_text="Zulrah has a combat level of 725"
)
# Returns: {
#   'formatted': '[CITE:source="Zulrah"|text="725"]Zulrah has a combat level of 725[/CITE]',
#   'source_title': 'Zulrah',
#   'source_text': '725',
#   'wiki_url': 'https://oldschool.runescape.wiki/w/Zulrah'
# }
```

**Why This Exists**:
Previously, the AI was asked to manually format citations, which led to:
- ❌ Inconsistent formats (sometimes `[Item Name]`, sometimes `[CITE:...]`)
- ❌ Missing citations
- ❌ Wrong citation placement
- ❌ Hallucinated citation details

Now, the citation tool uses structured wiki data directly to generate citations correctly every time.

---

### 5. citation_injector.py
**Purpose**: Automatically injects citations into AI responses post-processing

**Status**: ✅ **ACTIVE** - Post-processing citation injection

**Key Features**:
- **Fact Extraction**: Parses tool call results to identify facts
  - `get_full_wiki_page` → Extract combat level, hitpoints, stats
  - `get_item_price` → Extract Grand Exchange prices
- **Sentence Parsing**: Splits AI answer into individual sentences
- **Fact Matching**: Matches facts to sentences using flexible patterns
  - Standard matching: page title + value, field + value
  - Price matching: page title + price keywords (flexible for "1.5M GP" vs "1498814")
- **Citation Generation**: Uses citation tool to create proper citations
- **Automatic Injection**: Replaces sentences with cited versions

**How It Works**:
```
User Query → AI calls tools → AI generates answer
                                      ↓
                          No citations found?
                                      ↓
                          Citation Injector
                                      ↓
                    Extract facts from tool calls
                                      ↓
                    Match facts to sentences
                                      ↓
                    Generate citations with tool
                                      ↓
                    Return cited answer ✅
```

**Example**:
```
Input: "Zulrah has a combat level of 725."
Tool calls: [get_full_wiki_page("Zulrah") → {combat: 725}]

Output: "[CITE:source=\"Zulrah\"|text=\"725\"]Zulrah has a combat level of 725.[/CITE]"
```

**Performance**:
- Citation injection: < 100ms per response
- No additional API calls
- All lookups from in-memory data

**Benefits**:
- ✅ **100% Consistent Format** - All citations use proper `[CITE:...]` tags
- ✅ **Automatic** - No reliance on AI following instructions
- ✅ **Exact Wiki Content** - Citations trace back to structured data
- ✅ **Attribution Ready** - Can look up contributors for each fact
- ✅ **Fast** - < 100ms per response, all in-memory lookups

---

### 6. price_history.py
**Purpose**: Tracks Grand Exchange price history over time

**Status**: ✅ **ACTIVE** - Price tracking system

**Key Features**:
- **SQLite Database**: Stores price history in `data/price_history.db`
- **Automatic Recording**: Records prices when `get_item_price` tool is called
- **Time-Range Queries**: Get price history for specific time periods
- **Trend Analysis**: Calculate price changes, volatility, trends
- **Multi-Item Comparison**: Compare prices across multiple items

**Database Schema**:
```sql
CREATE TABLE price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    high_price INTEGER,
    low_price INTEGER,
    high_time INTEGER,
    low_time INTEGER,
    timestamp INTEGER NOT NULL,
    UNIQUE(item_name, timestamp)
);
```

**API Endpoints**:
- `GET /economic/price-history?item=<name>&hours=<hours>` - Get price history
- `POST /economic/compare` - Compare multiple items
- `GET /economic/trends?item=<name>` - Get price trends

---

### 7. api_queue_manager.py
**Purpose**: Centralized API rate limiting and request coordination

**Status**: ✅ **ACTIVE** - API coordination system

**Key Features**:
- **Priority-Based Queueing**: Watchdog gets highest priority
- **Rate Limiting**: Prevents API abuse
- **Request Throttling**: Slows other requests when watchdog is running
- **Statistics Tracking**: Monitor queue performance

**Priority Levels**:
1. **CRITICAL** (watchdog) - Highest priority, runs immediately
2. **HIGH** (user queries) - Normal priority
3. **LOW** (background tasks) - Lowest priority

**API Endpoint**:
- `GET /queue/stats` - Get queue statistics

---

### 8. embeddings/embedding_service.py
**Purpose**: Generates semantic embeddings using Ollama

**Status**: ✅ **ACTIVE** - Embedding generation

**Key Features**:
- **Batch Processing**: Processes up to 64 texts at once
- **Caching**: Avoids re-embedding identical texts
- **Async Support**: Concurrent requests for performance
- **Model Verification**: Auto-pulls models if missing

**Configuration**:
- Model: `mxbai-embed-large:latest`
- Batch Size: 64
- Max Concurrent Requests: 8
- Timeout: 45 seconds

**Usage**:
```python
from embeddings.embedding_service import EmbeddingService

service = EmbeddingService()
embeddings = service.embed_texts(["text1", "text2"])
```

---

### Command Files

These shell scripts are used by the admin GUIs to manage services:

#### 1. start-gui.command
**Purpose**: Start Ollama, RAG API, and Admin GUI

**Used By**: `admin/modern_admin_gui.py`, `admin/admin_gui.py`

**Options**:
- `--with-embedder` - Also start embedding watcher
- `--with-watchdog` - Also start wiki watchdog
- `--with-kg` - Also start KG updater
- `--basic-only` - Start only core services

---

#### 2. start-services-only.command
**Purpose**: Start only core services (no GUI)

**Used By**: `admin/pyqt6_admin_gui.py`

**Features**:
- Starts Ollama if not running
- Starts RAG API server
- Writes PID files for process management

---

#### 3. stop-all.command
**Purpose**: Stop all OSRS AI services

**Used By**: All admin GUIs

**Features**:
- Stops GUI, API, embedder, watchdog
- Uses PID files for clean shutdown
- Optional: `--and-ollama` to stop Ollama too

---

#### 4. start-data.command
**Purpose**: Start data pipeline (watchdog + embedder)

**Options**:
- `--no-api` - Don't start API server

---

#### 5. stop-data.command
**Purpose**: Stop data pipeline

**Options**:
- `--and-api` - Also stop API server

---

### Archived Components (api/old/)

These files have been moved to `api/old/` as they are no longer actively used in V3:

#### 1. osrs_rag_service.py
**Status**: ❌ **ARCHIVED** - Legacy V2 RAG service

**Reason**: Replaced by `osrs_agentic_rag.py` (V3 with LangGraph)

---

#### 2. kg_query_service.py
**Status**: ❌ **ARCHIVED** - Not actively used in V3

**Reason**: V3 agent uses wiki search and full page retrieval instead of explicit KG queries

---

#### 3. query_segmentation.py
**Status**: ❌ **ARCHIVED** - Legacy query analysis

**Reason**: V3 agent handles query understanding internally through LLM reasoning

---

#### 4. wiki_template_parser.py & wiki_template_checker.py
**Status**: ❌ **ARCHIVED** - Preprocessing only

**Reason**: Used during data preprocessing by scripts, not during runtime queries

---

#### 5. reranker_service.py
**Status**: ❌ **ARCHIVED** - Legacy reranker

**Reason**: V3 uses LangGraph agent workflow instead of explicit reranking

---

## Data Files

### Required Files (Must Exist)

#### 1. osrs_embeddings.jsonl (844MB)
**Purpose**: Semantic embeddings for all wiki pages

**Format**:
```json
{
  "title": "Zulrah",
  "embedding": [0.123, -0.456, ...],
  "content_preview": "Zulrah is a level 725 solo-only snake boss..."
}
```

**Generation**: Created by embedding service from `osrs_wiki_content.jsonl`

**Usage**: Loaded at startup for semantic search

---

#### 2. osrs_wiki_content.jsonl (170MB)
**Purpose**: Parsed and cleaned wiki page content

**Format**:
```json
{
  "title": "Zulrah",
  "categories": ["Bosses", "Slayer monsters"],
  "content": "Zulrah is a level 725 solo-only snake boss...",
  "infobox": {...},
  "sections": {...},
  "revid": 15234567,
  "timestamp": "2024-10-01T12:00:00Z"
}
```

**Generation**: Created by wiki watchdog from MediaWiki API

**Usage**: Source of truth for page content

---

#### 3. kg_entity_embeddings_mxbai.jsonl (2.0GB)
**Purpose**: Embeddings for knowledge graph entities

**Format**:
```json
{
  "entity": "Zulrah",
  "entity_type": "Monster",
  "embedding": [0.123, -0.456, ...],
  "context": "Level 725 snake boss located in Zul-Andra"
}
```

**Generation**: Created from knowledge graph nodes

**Usage**: Loaded at startup for KG-enhanced search

---

#### 4. osrs_kg_nodes.jsonl (6.1MB)
**Purpose**: Knowledge graph entity nodes

**Format**:
```json
{
  "id": "Q12345",
  "label": "Zulrah",
  "type": "Monster",
  "properties": {
    "combat_level": 725,
    "hitpoints": 500,
    "location": "Zul-Andra"
  }
}
```

---

#### 5. osrs_kg_edges.jsonl (179MB)
**Purpose**: Knowledge graph relationships

**Format**:
```json
{
  "source": "Q12345",
  "target": "Q67890",
  "relation": "drops",
  "properties": {
    "rarity": "1/512",
    "item": "Tanzanite fang"
  }
}
```

---

### Optional Files

#### osrs_wikitext_content.jsonl (101MB)
**Purpose**: Raw wikitext for advanced parsing

**Status**: Used by attribution service for contributor lookup

---

#### osrs_template_issues.jsonl (78MB)
**Purpose**: Template parsing issues log

**Status**: Diagnostic file, not used at runtime

---

## Configuration

### Environment Variables
None required. All configuration is in code.

### Model Configuration
Edit `osrs_agentic_rag.py`:
```python
llm = ChatOllama(
    model="llama3.1:8b",  # Change model here
    temperature=0.1,       # Adjust creativity
    num_ctx=32768,         # Context window size
)
```

### Embedding Configuration
Edit `embeddings/embedding_service.py`:
```python
@dataclass
class EmbeddingConfig:
    model_name: str = "mxbai-embed-large:latest"
    batch_size: int = 64
    max_concurrent_requests: int = 8
```

### API Server Configuration
Edit `osrs_api_server.py` or use command-line args:
```bash
python3 osrs_api_server.py --host 0.0.0.0 --port 5001 --debug
```

---

## Troubleshooting

### Issue: "Model not found"
**Solution**: Pull the model
```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large:latest
```

### Issue: "Cannot connect to Ollama"
**Solution**: Start Ollama server
```bash
ollama serve
```

### Issue: "File not found: osrs_embeddings.jsonl"
**Solution**: Ensure data files are in correct location
```bash
ls -lh /Users/brandon/Documents/projects/GE/data/osrs_embeddings.jsonl
```

### Issue: "Out of memory"
**Solution**: Reduce batch size in `embedding_service.py`
```python
batch_size: int = 32  # Reduce from 64
```

### Issue: "API server not responding"
**Solution**: Check if port 5001 is in use
```bash
lsof -i :5001
kill -9 <PID>
```

### Issue: "Frontend can't connect to API"
**Solution**: Check CORS and API URL
- API should be running on `http://localhost:5001`
- Frontend should point to correct API URL in `App.jsx`

### Issue: "No citations in response"
**Solution**: Check system prompt and model
- Ensure using `llama3.1:8b` (instruction-tuned)
- Check logs for "X citations" in output
- System prompt must include citation requirements

### Issue: "Contributor not found"
**Solution**: Attribution service limitations
- Only checks recent revisions (default: 8)
- Increase `max_checks` in API call
- Some text may not match due to template formatting

---

## Performance Optimization

### Startup Time
- **First Load**: ~30 seconds (loading embeddings)
- **Subsequent Loads**: ~15 seconds (cached)

### Query Response Time
- **Simple Query**: 5-10 seconds
- **Complex Query**: 15-30 seconds
- **Multi-Hop Query**: 30-60 seconds

### Memory Usage
- **API Server**: ~2GB RAM
- **Ollama (LLaMA 3.1:8b)**: ~5GB RAM
- **Ollama (mxbai-embed-large)**: ~1GB RAM
- **Total**: ~8GB RAM minimum

### Optimization Tips
1. **Reduce Context Window**: Lower `num_ctx` to 16384
2. **Reduce Batch Size**: Lower embedding batch size
3. **Use Smaller Model**: Try `llama3.2:3b` instead
4. **Limit Tool Calls**: Reduce max iterations in agent

---

## Development

### Adding New Tools
Edit `osrs_agentic_rag.py`:
```python
@tool
def my_new_tool(param: str) -> str:
    """Tool description for the AI"""
    # Implementation
    return result

# Add to tools list
osrs_tools = [search_osrs_wiki, get_full_wiki_page, my_new_tool]
```

### Modifying System Prompt
Edit `AGENT_SYSTEM_PROMPT` in `osrs_agentic_rag.py`

### Adding New API Endpoints
Edit `osrs_api_server.py`:
```python
@self.app.route('/my-endpoint', methods=['POST'])
def my_endpoint():
    # Implementation
    return jsonify(result)
```

---

## Testing

### Manual Testing
```bash
# Test health endpoint
curl http://localhost:5001/health

# Test chat endpoint
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Zulrah?"}'
```

### GUI Testing
1. Open http://localhost:5173
2. Ask: "What is Zulrah's combat level?"
3. Click "Show Attributions"
4. Verify tooltip appears with contributor info

---

## Maintenance

### Updating Wiki Data
Run wiki watchdog to refresh data:
```bash
cd /Users/brandon/Documents/projects/GE/scripts
python3 wiki_watchdog.py
```

### Regenerating Embeddings
```bash
cd /Users/brandon/Documents/projects/GE/scripts
python3 generate_embeddings.py
```

### Clearing Attribution Cache
```bash
rm /Users/brandon/Documents/projects/GE/data/attribution_cache.json
```

---

## Architecture Decisions

### Why LangGraph?
- Industry-standard agentic framework
- Built-in tool calling support
- State management for multi-step reasoning
- Easy to debug and extend

### Why gpt-oss:20b?
- Specifically designed for agentic tasks and structured outputs
- Excellent instruction following for citation format compliance
- Native function calling support
- Better optimization than general-purpose models
- Runs efficiently on Mac Mini M4 Pro
- Supports large context windows (32K tokens)
- Faster inference than similarly-sized models

### Why mxbai-embed-large?
- High-quality embeddings for semantic search
- Optimized for retrieval tasks
- Fast inference on Apple Silicon
- 1024-dimensional vectors (good balance)

### Why Citation Markers?
- Solves the attribution problem at the source
- AI knows exactly what text it's paraphrasing
- Enables accurate contributor lookup
- No need for post-hoc text matching

---

## Future Improvements

### Planned Features
1. **Streaming Responses**: Real-time token streaming
2. **Multi-Modal Support**: Image understanding for item icons
3. **Voice Interface**: Speech-to-text queries
4. **Mobile App**: Native iOS/Android apps
5. **Collaborative Filtering**: User feedback for better results

### Known Limitations
1. **No Real-Time Data**: Wiki data is static (updated manually)
2. **English Only**: No multi-language support
3. **Text Only**: No image or video understanding
4. **Single User**: No multi-user session management
5. **Local Only**: No cloud deployment (yet)

---

## Credits

**Developed by**: Brandon Inkel
**Framework**: LangGraph by LangChain
**Models**: gpt-oss:20b by OpenAI, mxbai-embed-large by MixedBread
**Data Source**: Old School RuneScape Wiki
**License**: Private/Internal Use

---

## Support

For issues or questions:
1. Check logs in terminal
2. Review this documentation
3. Check Ollama status: `ollama list`
4. Verify data files exist
5. Restart all services

---

**Last Updated**: October 5, 2025
**Version**: 3.1 (Agentic RAG with Automatic Citation Injection)

