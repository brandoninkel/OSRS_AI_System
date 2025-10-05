# OSRS AI Agentic RAG System

An intelligent AI assistant for Old School RuneScape powered by LangGraph, LLaMA 3.1, and a comprehensive wiki knowledge base.

## 🚀 Quick Start (3 Terminals Required)

**Terminal 1: Ollama**
```bash
ollama serve
```

**Terminal 2: API Server**
```bash
cd api
python3 osrs_api_server.py --host 0.0.0.0
```
Wait for "Loaded 35884 wiki embeddings" (~30 seconds)

**Terminal 3: Frontend**
```bash
cd frontend
npm run dev
```

**Access**: http://localhost:3005

---

## 📖 Complete Documentation

| Document | Description | Lines |
|----------|-------------|-------|
| **[QUICKSTART.md](api/QUICKSTART.md)** | Startup guide + troubleshooting | 532 |
| **[API.md](api/API.md)** | API server documentation | 948 |
| **[FRONTEND.md](frontend/FRONTEND.md)** | Frontend documentation | 1,274 |
| **[SCRIPTS.md](scripts/SCRIPTS.md)** | Data pipeline scripts | 730 |
| **[DATA.md](data/DATA.md)** | Data files documentation | 964 |
| **[ADMIN.md](admin/ADMIN.md)** | Admin GUI documentation | 632 |

**Total**: 5,080 lines of comprehensive documentation

---

## ✅ Verify System is Running

```bash
# Check all 3 services
curl http://localhost:11434  # Ollama
curl http://localhost:5001/health  # API
curl http://localhost:3005  # Frontend
```

All should respond successfully.

---

## 🔗 Quick Links

- **Frontend GUI**: http://localhost:3005
- **API Health**: http://localhost:5001/health
- **Full Docs**: [QUICKSTART.md](api/QUICKSTART.md)

---

**Version**: 3.0 (Agentic RAG with LangGraph)  
**Last Updated**: October 3, 2025
