# OSRS Agentic RAG - Quick Start Guide

## 🚀 Quick Start (3 Terminals Required)

### Terminal 1: Start Ollama
```bash
# Start Ollama server (required for LLM and embeddings)
ollama serve
```

**Expected Output**:
```
Ollama server is running on http://localhost:11434
```

**Leave this terminal running** - Do not close it!

---

### Terminal 2: Start API Server
```bash
# Navigate to API directory
cd /Users/brandon/Documents/projects/GE/api

# Start the Flask API server
python3 osrs_api_server.py --host 0.0.0.0
```

**Expected Output**:
```
INFO - Initializing OSRS Agentic RAG V3 service...
INFO - ✅ Embedding service initialized with mxbai-embed-large:latest
INFO - Loaded 35884 wiki embeddings
INFO - ✅ OSRS Agentic RAG V3 service initialized
INFO - ✅ Attribution service initialized
 * Running on http://0.0.0.0:5001
```

**Wait for "Loaded 35884 wiki embeddings"** before proceeding (takes ~30 seconds)

**Leave this terminal running** - Do not close it!

---

### Terminal 3: Start Frontend
```bash
# Navigate to frontend directory
cd /Users/brandon/Documents/projects/GE/frontend

# Start the Vite dev server
npm run dev
```

**Expected Output**:
```
VITE v7.1.6  ready in 105 ms

➜  Local:   http://localhost:3005/
➜  Network: use --host to expose
```

**Leave this terminal running** - Do not close it!

---

### ✅ Access the Application

**Open in browser**: http://localhost:3005

**Verify all services are running**:
- ✅ Ollama: http://localhost:11434
- ✅ API Server: http://localhost:5001/health
- ✅ Frontend: http://localhost:3005

---

## 📋 Prerequisites Checklist

- [ ] **Ollama installed** (`brew install ollama`)
- [ ] **Models pulled**:
  ```bash
  ollama pull llama3.1:8b
  ollama pull mxbai-embed-large:latest
  ```
- [ ] **Python 3.10+** installed
- [ ] **Node.js 18+** installed
- [ ] **Data files exist** in `/Users/brandon/Documents/projects/GE/data/`:
  - `osrs_embeddings.jsonl` (844MB)
  - `osrs_wiki_content.jsonl` (170MB)
  - `kg_entity_embeddings_mxbai.jsonl` (2.0GB)

---

## 🧪 Test the System

### 1. Test API Health
```bash
curl http://localhost:5001/health
```

Expected response:
```json
{"status": "healthy", "service": "OSRS RAG API"}
```

### 2. Test Chat Query
```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Zulrah?"}'
```

### 3. Test GUI
1. Open http://localhost:3005
2. Verify API status shows "online" (green badge)
3. Ask: "What is Zulrah's combat level?"
4. Wait for response (5-30 seconds)
5. Click "Show Attributions" button
6. Hover over highlighted text
7. Verify tooltip shows:
   - Source page link
   - Excerpt
   - Author name
   - Timestamp
   - Edit comment
   - Revision link

---

## 🔧 Common Issues

### "Cannot connect to Ollama"
```bash
# Start Ollama
ollama serve
```

### "Model not found"
```bash
# Pull models
ollama pull llama3.1:8b
ollama pull mxbai-embed-large:latest
```

### "Port 5001 already in use"
```bash
# Find and kill process
lsof -i :5001
kill -9 <PID>
```

### "Data files not found"
```bash
# Verify files exist
ls -lh /Users/brandon/Documents/projects/GE/data/*.jsonl
```

---

## 📊 System Status

### Check Ollama Models
```bash
ollama list
```

Should show:
- `llama3.1:8b`
- `mxbai-embed-large:latest`

### Check Running Processes
```bash
# Check API server (should show python3)
lsof -i :5001

# Check frontend (should show node)
lsof -i :3005

# Check Ollama (should show ollama)
lsof -i :11434
```

**All three ports should be LISTEN**ing for the system to work properly.

---

## 📖 Full Documentation

See [API.md](./API.md) for complete documentation including:
- Detailed architecture
- API endpoint specifications
- Component descriptions
- Configuration options
- Troubleshooting guide
- Development guide

---

## 🎯 What This System Does

1. **Intelligent Search**: AI agent decides what to search for
2. **Semantic Understanding**: Uses embeddings for context-aware retrieval
3. **Citation Tracking**: AI includes exact source text in responses
4. **Contributor Attribution**: Finds wiki authors for transparency
5. **Interactive UI**: Hover over highlighted text to see sources

---

## 🏗️ Architecture Overview

```
User Query → Frontend → API Server → Agentic RAG → Ollama (LLM)
                                    ↓
                              Wiki Embeddings (35K pages)
                                    ↓
                              Knowledge Graph (149K entities)
                                    ↓
                              Answer + Citations
                                    ↓
                              Attribution Service → MediaWiki API
                                    ↓
                              Contributors Found
```

---

## 💡 Example Queries

- "What is Zulrah's combat level?"
- "How do I start Dragon Slayer?"
- "What drops does Vorkath have?"
- "Where is the Grand Exchange?"
- "What are the requirements for Recipe for Disaster?"

---

## 🔄 Restart All Services

### Option 1: Kill and Restart (Recommended)

**Terminal 1: Restart Ollama**
```bash
# Kill existing Ollama
pkill -f "ollama serve"

# Start fresh
ollama serve
```

**Terminal 2: Restart API Server**
```bash
# Kill existing API server
pkill -f "osrs_api_server.py"

# Start fresh
cd /Users/brandon/Documents/projects/GE/api
python3 osrs_api_server.py --host 0.0.0.0
```

**Terminal 3: Restart Frontend**
```bash
# Kill existing frontend
pkill -f "vite"

# Start fresh
cd /Users/brandon/Documents/projects/GE/frontend
npm run dev
```

### Option 2: Background Processes (Not Recommended)

```bash
# Kill all services
pkill -f "ollama serve"
pkill -f "osrs_api_server.py"
pkill -f "vite"

# Restart in background
ollama serve > /dev/null 2>&1 &
cd /Users/brandon/Documents/projects/GE/api && python3 osrs_api_server.py --host 0.0.0.0 > ../logs/osrs_ai/api.out 2>&1 &
cd /Users/brandon/Documents/projects/GE/frontend && npm run dev > ../logs/osrs_ai/frontend.out 2>&1 &
```

**Note**: Background processes make debugging harder. Use 3 separate terminals for easier troubleshooting.

---

## 📝 Key Files

- `api/osrs_api_server.py` - Flask API server
- `api/osrs_agentic_rag.py` - LangGraph agent
- `api/attribution_service.py` - Contributor lookup
- `api/embeddings/embedding_service.py` - Embedding generation
- `frontend/src/App.jsx` - React GUI
- `data/osrs_embeddings.jsonl` - Wiki embeddings
- `data/osrs_wiki_content.jsonl` - Wiki content

---

## 🎓 How It Works

1. **User asks question** in GUI
2. **Frontend sends query** to API server
3. **API server calls** agentic RAG
4. **Agent decides** which tools to use:
   - `search_osrs_wiki()` - Semantic search
   - `get_full_wiki_page()` - Full page retrieval
5. **Agent generates answer** with citation markers
6. **Citations parsed** and sent to frontend
7. **Frontend displays** answer with highlights
8. **User clicks "Show Attributions"**
9. **API fetches contributors** from MediaWiki
10. **Tooltip shows** source, excerpt, author, revision link

---

## 🚨 Important Notes

- **First startup takes ~30 seconds** (loading embeddings)
- **Queries take 5-30 seconds** depending on complexity
- **Requires ~8GB RAM** minimum
- **Data is static** (updated manually via wiki watchdog)
- **No fallbacks or fake data** - all AI responses are real

---

## 🔗 Useful Links

- **Frontend GUI**: http://localhost:3005
- **API Server**: http://localhost:5001
- **API Health Check**: http://localhost:5001/health
- **Ollama Server**: http://localhost:11434
- **OSRS Wiki**: https://oldschool.runescape.wiki

---

## 🔍 Startup Troubleshooting

### Issue: "API status shows offline" in GUI

**Diagnosis**:
```bash
# Check if API server is running
curl http://localhost:5001/health
```

**If connection refused**:
1. API server is not running
2. Check Terminal 2 for errors
3. Verify Ollama is running first
4. Restart API server

**If timeout**:
1. API server is loading embeddings (wait 30 seconds)
2. Check Terminal 2 for "Loaded 35884 wiki embeddings"

---

### Issue: "Cannot connect to Ollama"

**Diagnosis**:
```bash
# Check if Ollama is running
curl http://localhost:11434
```

**Solution**:
```bash
# Terminal 1: Start Ollama
ollama serve
```

**Verify models are pulled**:
```bash
ollama list
```

Should show:
- `llama3.1:8b`
- `mxbai-embed-large:latest`

---

### Issue: "Frontend shows blank page"

**Diagnosis**:
```bash
# Check if frontend is running
lsof -i :3005
```

**Solution**:
```bash
# Terminal 3: Restart frontend
pkill -f "vite"
cd /Users/brandon/Documents/projects/GE/frontend
npm run dev
```

**Check browser console** (F12) for errors

---

### Issue: "Port already in use"

**For API Server (port 5001)**:
```bash
# Find process using port
lsof -i :5001

# Kill it
kill -9 <PID>

# Restart API server
cd /Users/brandon/Documents/projects/GE/api
python3 osrs_api_server.py --host 0.0.0.0
```

**For Frontend (port 3005)**:
```bash
# Find process using port
lsof -i :3005

# Kill it
kill -9 <PID>

# Restart frontend
cd /Users/brandon/Documents/projects/GE/frontend
npm run dev
```

---

### Issue: "Data files not found"

**Verify files exist**:
```bash
ls -lh /Users/brandon/Documents/projects/GE/data/*.jsonl
```

**Required files**:
- `osrs_embeddings.jsonl` (844MB)
- `osrs_wiki_content.jsonl` (170MB)
- `kg_entity_embeddings_mxbai.jsonl` (2.0GB)
- `osrs_kg_nodes.jsonl` (6.1MB)
- `osrs_kg_edges.jsonl` (179MB)

**If missing**: Run wiki watchdog and embedding generation scripts (see [SCRIPTS.md](../scripts/SCRIPTS.md))

---

### Issue: "Queries are very slow"

**Normal behavior**:
- First query: 10-30 seconds (agent reasoning + LLM generation)
- Subsequent queries: 5-15 seconds

**If slower than 60 seconds**:
1. Check system resources (RAM, CPU)
2. Verify Ollama is using GPU acceleration
3. Check for other heavy processes

---

### Issue: "Attributions not showing"

**Diagnosis**:
1. Click "Show Attributions" button
2. Check browser console (F12) for errors
3. Verify API server can reach MediaWiki API

**Test attribution endpoint**:
```bash
curl -X POST http://localhost:5001/attributions \
  -H "Content-Type: application/json" \
  -d '{"citations": [{"source_title": "Zulrah", "source_text": "Combat Level: 725", "text": "Zulrah has a combat level of 725", "start": 0, "end": 33}]}'
```

---

## 📞 Support Checklist

**Before asking for help, verify**:

1. ✅ **All 3 terminals are running**:
   - Terminal 1: `ollama serve`
   - Terminal 2: `python3 osrs_api_server.py`
   - Terminal 3: `npm run dev`

2. ✅ **All 3 ports are listening**:
   ```bash
   lsof -i :11434  # Ollama
   lsof -i :5001   # API Server
   lsof -i :3005   # Frontend
   ```

3. ✅ **API health check passes**:
   ```bash
   curl http://localhost:5001/health
   ```

4. ✅ **Data files exist**:
   ```bash
   ls -lh /Users/brandon/Documents/projects/GE/data/*.jsonl
   ```

5. ✅ **Models are pulled**:
   ```bash
   ollama list
   ```

6. ✅ **Check terminal logs** for error messages

7. ✅ **Try restarting all services** (see "Restart All Services" section)

---

## 📚 Additional Documentation

- **[API.md](./API.md)** - Complete API documentation
- **[FRONTEND.md](../frontend/FRONTEND.md)** - Frontend documentation
- **[SCRIPTS.md](../scripts/SCRIPTS.md)** - Data pipeline scripts
- **[DATA.md](../data/DATA.md)** - Data files documentation
- **[ADMIN.md](../admin/ADMIN.md)** - Admin GUI documentation

---

**Version**: 3.0 (Agentic RAG with LangGraph)
**Last Updated**: October 3, 2025
**Frontend Port**: 3005
**API Port**: 5001
**Ollama Port**: 11434

