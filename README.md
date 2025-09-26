# OSRS AI System

A comprehensive OSRS AI system with RAG (Retrieval-Augmented Generation) capabilities, Knowledge Graph integration, chat session isolation, and modern React frontend.

## 🚨 IMPORTANT: Directory Structure

**⚠️ IGNORE BACKUP DIRECTORIES**: Do not use files from `backup_old_projects/`, `old/`, or any nested `OSRS_AI_SYSTEM/` directories. These contain outdated code and configurations.

**✅ CURRENT STRUCTURE** (flat directory layout):
```
/Users/brandon/Documents/projects/GE/
├── api/                    # RAG API and services
│   ├── osrs_rag_service.py    # Main RAG service with LLaMA 3.1 8B
│   ├── osrs_api_server.py     # Flask API server
│   ├── attribution_service.py # Wiki attribution tracking
│   ├── kg_query_service.py    # Knowledge Graph queries
│   ├── reranker_service.py    # BGE reranker integration
│   ├── start-gui.command      # Start all services (recommended)
│   ├── start-data.command     # Start data services only
│   └── stop-all.command       # Stop all services
├── frontend/               # React + Vite frontend
│   ├── src/                   # React components and logic
│   ├── package.json           # Frontend dependencies
│   └── vite.config.js         # Vite configuration
├── admin/                  # Admin control panel
│   ├── admin_gui.py           # DearPyGui admin interface
│   └── start-admin.command    # Start admin panel
├── data/                   # Wiki content and embeddings
│   ├── osrs_wiki_content.jsonl     # Full wiki content (42K+ pages)
│   ├── osrs_embeddings.jsonl       # mxbai-embed-large embeddings
│   ├── osrs_kg_triples.csv         # Knowledge Graph triples
│   ├── osrs_kg_nodes.jsonl         # KG entity nodes
│   ├── kg_model/                   # PyKEEN KG embeddings
│   └── cache/                      # Attribution cache
├── scripts/                # Utilities and automation
│   ├── create_osrs_embeddings.py   # Generate embeddings
│   ├── streamlined-watchdog.js     # Wiki monitoring
│   ├── kg/                         # Knowledge Graph tools
│   │   ├── build_kg.py             # Build KG from wiki content
│   │   ├── train_kg_embeddings.py  # Train PyKEEN embeddings
│   │   └── eval_kg_embeddings.py   # Evaluate KG model
│   ├── train-kg-embeddings.command # KG training wrapper
│   └── kg-status.command           # Monitor KG training
├── gui/                    # Legacy GUI (use frontend/ instead)
├── logs/                   # System logs
│   ├── osrs_ai/               # API and service logs
│   └── kg/                    # Knowledge Graph training logs
├── config/                 # Configuration files
├── package.json           # Node.js dependencies
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Ollama** with LLaMA 3.1 8B model
- **Git** for version control

### 2. Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies
npm install

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 3. Start the Complete System
```bash
# Option A: Start everything (recommended)
./api/start-gui.command

# Option B: Start services individually
# Terminal 1: API server
cd api && python3 osrs_api_server.py --port 5002

# Terminal 2: Frontend dev server
cd frontend && npm run dev

# Terminal 3: Admin panel (optional)
cd admin && python3 admin_gui.py
```

### 4. Access the System
- **Frontend GUI**: http://localhost:3005 (Vite dev server)
- **API Health**: http://localhost:5002/health
- **API Stats**: http://localhost:5002/stats
- **Legacy GUI**: http://localhost:3002 (if using gui/ directory)

## 🔧 System Components

### Core Services
- **RAG API** (`api/osrs_rag_service.py`): Main retrieval and generation service
- **API Server** (`api/osrs_api_server.py`): Flask REST API with CORS
- **Frontend** (`frontend/`): Modern React interface with Vite
- **Admin Panel** (`admin/admin_gui.py`): DearPyGui control interface

### Data Processing
- **Wiki Watchdog** (`scripts/streamlined-watchdog.js`): Monitors OSRS wiki changes
- **Embedding Generator** (`scripts/create_osrs_embeddings.py`): Creates mxbai embeddings
- **Knowledge Graph Builder** (`scripts/kg/build_kg.py`): Extracts entities and relations

### Knowledge Graph
- **KG Training** (`scripts/kg/train_kg_embeddings.py`): PyKEEN TransE embeddings
- **KG Evaluation** (`scripts/kg/eval_kg_embeddings.py`): Model performance metrics
- **KG Query Service** (`api/kg_query_service.py`): Entity relationship queries

## 📊 System Specifications

### Data Scale
- **42,719** OSRS wiki page embeddings (1024D mxbai-embed-large)
- **148,003** Knowledge Graph entity embeddings (1024D)
- **509,382** KG relationship links
- **32,336** unique entities

### Ports & Services
- **API Server**: Port 5002 (Flask)
- **Frontend Dev**: Port 3005 (Vite)
- **Legacy GUI**: Port 3002 (Python HTTP server)
- **Ollama**: Port 11434 (LLaMA 3.1 8B)
- **Admin Panel**: Desktop application (DearPyGui)

### AI Models
- **LLM**: LLaMA 3.1 8B via Ollama
- **Embeddings**: mxbai-embed-large (1024D)
- **Reranker**: BAAI/bge-reranker-large (optional)
- **KG Embeddings**: PyKEEN TransE (100D)

## 🛠️ Development Workflow

### Making Changes
```bash
# Make your changes
git add <files>
git commit -m "Fix #N: Description of change"
git push origin main
```

### Testing
```bash
# Test API endpoints
cd api && python3 run_modality_tests.py

# Test search functionality
cd api && python3 run_search_battery.py

# Test HTTP endpoints
cd api && python3 run_http_battery.py
```

### Monitoring
```bash
# Check system status
./api/start-gui.command --status

# Monitor KG training
./scripts/kg-status.command

# View logs
tail -f logs/osrs_ai/api.out
```

## 🔍 Key Features

- **Chat Session Isolation**: Unique chat IDs prevent context bleeding
- **Real-time Streaming**: Server-Sent Events for live responses
- **Attribution System**: Track wiki sources with contributor info
- **Knowledge Graph**: Entity relationships and semantic search
- **Template Parsing**: Advanced wikitext template processing
- **Spell Correction**: OSRS-aware query enhancement
- **Progress Tracking**: Real-time processing feedback
- **Admin Controls**: GUI for system management

## 🚨 Troubleshooting

### Common Issues
1. **Port conflicts**: Check if services are already running
2. **Missing embeddings**: Run `scripts/create_osrs_embeddings.py`
3. **Ollama not found**: Ensure Ollama is installed and running
4. **Frontend build errors**: Delete `node_modules` and reinstall

### Log Locations
- **API logs**: `logs/osrs_ai/api.out`
- **Frontend logs**: `logs/osrs_ai/frontend.out`
- **KG training logs**: `logs/kg/train_*.log`

## 📚 Technologies

- **Backend**: Python 3.9, Flask, NumPy, PyKEEN
- **Frontend**: React 19, Vite, TailwindCSS, TypeScript
- **AI/ML**: Ollama, mxbai-embed-large, BGE reranker
- **Data**: JSONL, CSV, SQLite caching
- **Monitoring**: Node.js watchdog, DearPyGui admin
