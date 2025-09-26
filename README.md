# OSRS AI System - Clean Architecture

A streamlined, organized OSRS AI system with RAG capabilities, chat session isolation, and real-time GUI.

## Directory Structure

```
OSRS_AI_SYSTEM/
├── api/                    # RAG API and embedding services
│   ├── osrs_rag_service.py    # Main RAG service with LLaMA 3.1
│   ├── osrs_api_server.py     # Flask API server (localhost:5001)
│   └── embeddings/
│       └── embedding_service.py  # mxbai-embed-large service
├── gui/                    # Web interface
│   ├── osrs-rag-gui.html     # Main GUI with chat sessions
│   └── serve-rag-gui.py      # GUI server (localhost:3002)
├── data/                   # Wiki content and embeddings
│   ├── osrs_wiki_content.jsonl   # Full wiki content
│   ├── osrs_embeddings.jsonl     # 33,554 embeddings
│   └── osrs_page_titles.txt      # Page titles list
├── scripts/                # Utilities and tools
│   ├── optimized-watchdog.js     # Wiki monitoring
│   └── create_osrs_embeddings.py # Embedding generation
├── config/                 # Configuration files
├── package.json           # Node.js dependencies
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies (already installed)
npm install
```

### 2. Start the System
```bash
# Start API server (Terminal 1)
cd api && python osrs_api_server.py

# Start GUI server (Terminal 2)  
cd gui && python serve-rag-gui.py

# Access GUI at: http://localhost:3002
```

### 3. Optional Tools
```bash
# Start wiki watchdog
cd scripts && node optimized-watchdog.js

# Regenerate embeddings
cd scripts && python create_osrs_embeddings.py
```

## Features

- **Chat Session Isolation**: Each conversation has its own context
- **Streaming Progress Bars**: Real-time processing feedback
- **Context Gauges**: Monitor conversation memory usage
- **Entity Extraction**: Automatic context tracking
- **Wiki Monitoring**: Automated content updates
- **33,554 Embeddings**: Complete OSRS wiki knowledge

## Technologies

- **LLaMA 3.1 8B** via Ollama
- **mxbai-embed-large** embeddings
- **Flask** API server
- **Server-Sent Events** streaming
- **Node.js** watchdog systems
