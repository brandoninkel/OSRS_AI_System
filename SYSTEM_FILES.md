# OSRS AI System - Clean Architecture File Tracker

## 🎯 SYSTEM LOCATION: `/Users/brandon/Documents/projects/GE/OSRS_AI_SYSTEM/`

## 📁 Directory Structure
```
OSRS_AI_SYSTEM/
├── api/                    # RAG API and embedding services
├── gui/                    # Web interface
├── data/                   # Wiki content and embeddings
├── scripts/                # Utilities and tools
├── config/                 # Configuration files
├── package.json           # Node.js dependencies
├── requirements.txt       # Python dependencies
├── start.sh              # Quick startup script
└── README.md             # Documentation
```

## 🧠 AI System Core Files
- `api/osrs_rag_service.py` - Main RAG service with LLaMA 3.1 + mxbai-embed-large embeddings, chat session isolation, entity extraction, contextual reference resolution
- `api/osrs_api_server.py` - Flask API server (localhost:5001) with streaming endpoints, chat session management, health/stats/context endpoints
- `api/wiki_template_parser.py` - MediaWiki template parser for processing OSRS wiki templates into readable text

## 🖥️ GUI System Files
- `gui/osrs-rag-gui.html` - Main GUI interface with chat session management, streaming progress bars, context gauges, real-time Server-Sent Events
- `gui/serve-rag-gui.py` - Simple HTTP server (localhost:3002) to serve the GUI HTML file

## 👁️ Watchdog System Files
- `scripts/streamlined-watchdog.js` - Streamlined wiki content monitoring with change detection, archiving, incremental updates, and proper MediaWiki API etiquette

## 🔗 Embedding System Files
- `scripts/create_osrs_embeddings.py` - Main embedding generation script using mxbai-embed-large with custom OSRS text parsing and enhancement
- `api/embeddings/embedding_service.py` - Embedding service using Ollama with mxbai-embed-large model, batch processing, caching

## 📚 Data Files - Wiki Content
- `data/osrs_wiki_content.jsonl` - Main OSRS wiki content file with full page text and metadata (211,000+ pages)
- `data/osrs_page_titles.txt` - Complete list of OSRS wiki page titles for monitoring and validation

## 🧮 Data Files - Embeddings
- `data/osrs_embeddings.jsonl` - 33,554 OSRS wiki page embeddings (1024D mxbai-embed-large vectors)


## 🧱 Knowledge Graph System
- `scripts/kg/build_kg.py` — KG builder (parallel workers supported with `--workers N`), snapshots inputs when `--snapshot` is used, and now auto-cleans snapshots on success.
- `scripts/knowledge-graph.command` — One-click runner; logs to `logs/kg/build_YYYYmmdd_HHMMSS.log`.

### Outputs
- `data/osrs_kg_triples.csv` — Triples (head, relation, tail, source_title, revid)
- `data/osrs_kg_nodes.jsonl` — Per-node summaries
- `data/osrs_kg_edges.jsonl` — Per-edge debug records
- `data/osrs_kg.meta.json` — Run metadata
- `data/tmp/` — Temporary parts and (when `--snapshot`) input snapshots; snapshots are auto-removed on success and retained on failure for debugging.

## 🧭 Admin Console Controls (DearPyGui)
- Start/Stop components:
  - API server (Flask): `api/osrs_api_server.py`
  - GUI server (static): `gui/serve-rag-gui.py` (or Vite dev: `frontend/` when enabled)
  - Watchdog: `scripts/streamlined-watchdog.js`
  - Embedder: `scripts/create_osrs_embeddings.py`
- Launchers used by the console:
  - `api/start-gui.command` — Starts API + GUI (+ optional Watchdog/Embedder via flags)
  - `api/start-data.command` — Starts Watchdog + Embedder (+ optional API)
  - `api/stop-all.command` — Stops all via PID files and port fallbacks
- Runtime logs and PIDs: `logs/osrs_ai/*.out` and `logs/osrs_ai/*.pid`
- Primary data inputs/outputs:
  - Inputs: `data/osrs_wiki_content.jsonl`, `data/osrs_wikitext_content.jsonl`, `data/osrs_embeddings.jsonl`, optional KG artifacts
  - Outputs: `data/osrs_embeddings.jsonl` (updates), `data/osrs_kg_*.{csv,jsonl}`, metadata and cache files under `data/`, snapshots under `data/tmp/`

## 🚀 Quick Start Commands
```bash
# Start everything
./start.sh

# Or manually:
cd api && python osrs_api_server.py     # Terminal 1
cd gui && python serve-rag-gui.py       # Terminal 2

# Access GUI: http://localhost:3002
# API Health: http://localhost:5001/health
```

## 🔧 Key Technologies Used
- **LLaMA 3.1 8B** - Language model for generating responses (via Ollama)
- **mxbai-embed-large** - 1024-dimensional embedding model for semantic similarity search
- **Flask** - Python web framework for RAG API server
- **Server-Sent Events (SSE)** - Real-time streaming communication for progress updates
- **Node.js** - JavaScript runtime for watchdog systems and wiki monitoring
- **NumPy** - Fast similarity computation with embedding matrices
- **Cosine Similarity** - Semantic search algorithm for finding relevant wiki content
- **JSONL Format** - Line-delimited JSON for efficient streaming and processing of large datasets

## ✅ System Status
- **API Server**: ✅ Working (localhost:5001)
- **GUI Server**: ✅ Working (localhost:3002)
- **Embeddings**: ✅ Loaded (33,554 pages)
- **Chat Sessions**: ✅ Isolated per chat ID
- **Dependencies**: ✅ Installed (Node.js + Python)
- **Watchdog Paths**: ✅ Updated for new structure
- **Embedding Creation**: ✅ Updated for new structure
- **Template Parser**: ✅ Copied to api/ directory
