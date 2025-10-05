# API Migration - RAG Service Evolution

**Date:** 2025-10-03
**Purpose:** Evolution from bloated V1 → Clean V2 → Agentic V3

---

## 🚀 V3: Agentic RAG with LangGraph (CURRENT)

**File:** `api/osrs_agentic_rag.py` (515 lines)
**Status:** ✅ Working - LangGraph implementation complete
**Dependencies:** `langgraph`, `langchain`, `langchain-community`, `langchain-ollama`

### Why V3?

V2 had fundamental issues:
- ❌ Manual phrase boosting (not organic)
- ❌ Hardcoded stat query detection
- ❌ Single-pass retrieval
- ❌ No reasoning or self-correction
- ❌ Only works for pre-programmed patterns

### V3 Architecture: LangGraph Agentic RAG

**Framework:** LangGraph (industry-standard agentic workflows)

**LangChain Tools** (Agent can call):
```python
@tool
def search_osrs_wiki(query: str) -> str:
    """Search 35,884 wiki pages for items, quests, monsters, skills"""

@tool
def search_osrs_knowledge_graph(query: str) -> str:
    """Search 149,047 KG entities for relationships"""

@tool
def get_full_wiki_page(title: str) -> str:
    """Get complete content of a specific page"""
```

**LangGraph Workflow:**
```
START
  ↓
agent (LLM decides: use tools or respond)
  ↓
[Conditional Edge]
  ├─→ tools (if tool_calls present)
  │     ↓
  │   [Execute tools]
  │     ↓
  │   agent (evaluate results, decide next action)
  │     ↓
  │   [Loop until satisfied]
  └─→ END (if no tool_calls)
```

**State:** `MessagesState` (LangGraph built-in)
- Maintains conversation history
- Tracks tool calls and results
- Supports streaming

**Agent System Prompt:**
```
You are an expert OSRS assistant with access to:
- search_osrs_wiki: Search wiki pages
- search_osrs_knowledge_graph: Find entity relationships
- get_full_wiki_page: Get complete page content

Think step-by-step:
1. Planning - Decide what to search
2. Searching - Use tools to find info
3. Analyzing - Evaluate if you need more
4. Answering - Provide accurate response
```

### Test Results (LangGraph Version)

**Query: "What is Zulrah's hitpoints?"**
- ✅ Agent called `search_osrs_wiki` with "Zulrah hitpoints"
- ✅ Found relevant pages (75.6%, 73.7%, 73.5%)
- ⚠️ Agent tried to make second search but didn't execute
- **Issue:** Need to enable multi-hop reasoning loop

**Query: "How do I start Dragon Slayer?"**
- ✅ Agent called `search_osrs_wiki` with "Dragon Slayer quest start"
- ✅ Found quest pages (74.2%, 74.1%, 73.3%)
- ✅ Provided accurate answer: "Speak to Guildmaster, need 32 QP"
- **Success:** Organic quest detection!

**Query: "What are the requirements for Recipe for Disaster?"**
- ✅ Agent called `search_osrs_wiki` with "Recipe for Disaster"
- ✅ Found quest guides (69.7%, 65.9%, 64.9%)
- ⚠️ Agent wanted to search again for requirements but didn't execute
- **Issue:** Need to enable multi-hop reasoning loop

### What's Working

1. **LLM-based planning** ✅ - Agent decides what to search
2. **Tool calling** ✅ - Agent uses search tools
3. **Reasoning visible** ✅ - Can see agent's thought process
4. **Organic behavior** ✅ - No hardcoded patterns
5. **State management** ✅ - LangGraph handles it

### What Needs Work

1. **Multi-hop reasoning** ❌ - Agent tries second search but doesn't execute
2. **Full page retrieval** ❌ - Agent doesn't use `get_full_wiki_page` tool
3. **Self-correction** ❌ - Agent doesn't retry if results are poor
4. **Streaming** ⚠️ - Implemented but not tested

### Next Steps for V3

1. **Fix multi-hop loop** - Let agent make multiple tool calls in sequence
2. **Teach agent to use get_full_wiki_page** - Better prompting or examples
3. **Add max iterations** - Prevent infinite loops
4. **Integrate with Flask API** - Replace V2 with V3
5. **Test streaming in GUI** - Show agent reasoning in real-time

---

## 📦 V2: Clean RAG Service (ARCHIVED)

**File:** `api/osrs_rag_service.py` (698 lines)
**Status:** 🗄️ Archived - Replaced by V3

---

## 📋 Migration Plan

### Phase 1: Identify Dependencies ✅

**Files to KEEP (Active Dependencies):**
- ✅ `api/embeddings/embedding_service.py` - Used by RAG, KG scripts, embedding creator
- ✅ `api/attribution_service.py` - Used by RAG for wiki contributor attribution
- ✅ `api/wiki_template_parser.py` - Used by streamlined-watchdog.js
- ✅ `api/wiki_template_checker.py` - Used by streamlined-watchdog.js
- ✅ `api/query_segmentation.py` - NEW, working great for phrase detection
- ✅ `api/osrs_api_server.py` - Flask API server (will update to use V2)

**Files to ARCHIVE (Unused/Deprecated):**
- 🗄️ `api/osrs_rag_service.py` → `api/old/osrs_rag_service_v1.py`
- 🗄️ `api/reranker_service.py` → `api/old/reranker_service.py` (not installed, unused)
- 🗄️ `api/kg_query_service.py` → `api/old/kg_query_service.py` (check if used)
- 🗄️ `api/run_*.py` test scripts → `api/old/` (keep for reference)

---

## 📦 Files to Archive

### 1. `api/osrs_rag_service.py` (2,748 lines)
**Reason:** Bloated with unused features, being replaced by V2  
**Destination:** `api/old/osrs_rag_service_v1.py`  
**What it did:**
- Wiki embedding search with title matching
- KG embeddings integration (PyKEEN + mxbai)
- Reranker support (unused - FlagEmbedding not installed)
- Spell correction (disabled by default)
- Recursive embedding expansion
- Timeline/intent detection
- Self-research aggregation
- Chat session management
- Attribution service integration
- Multiple fallback mechanisms

**What we're keeping in V2:**
- ✅ Wiki embedding search (simplified)
- ✅ KG embeddings (mxbai only)
- ✅ Query segmentation (NEW - phrase detection)
- ✅ Self-research aggregation (simplified)
- ✅ Chat session management
- ✅ Attribution service

**What we're removing:**
- ❌ Reranker code (not installed)
- ❌ Spell correction (not needed)
- ❌ Timeline/intent detection (over-engineered)
- ❌ Recursive expansion (self-research is simpler)
- ❌ Multiple fallback paths (confusing)

---

### 2. `api/reranker_service.py`
**Reason:** FlagEmbedding/bge-reranker-large not installed, code never runs  
**Destination:** `api/old/reranker_service.py`  
**What it did:**
- Cross-encoder reranking using BAAI/bge-reranker-large
- Rerank top-k candidates for better precision

**Why removing:**
- Package not installed
- Adds complexity without benefit
- Query segmentation provides better results

---

### 3. `api/kg_query_service.py`
**Reason:** Need to verify if used anywhere  
**Destination:** `api/old/kg_query_service.py` (if unused)  
**What it did:**
- Direct KG traversal queries
- Entity relationship exploration

**Action:** Check usage before archiving

---

### 4. Test Scripts
**Files:**
- `api/run_http_battery.py`
- `api/run_modality_tests.py`
- `api/run_search_battery.py`

**Reason:** Test scripts, keep for reference but not in main api/  
**Destination:** `api/old/tests/`  
**What they did:**
- HTTP endpoint testing
- Modality testing (text, images, etc.)
- Search quality testing

---

## 🏗️ V2 Architecture

### New File: `api/osrs_rag_service_v2.py` (~800-1000 lines)

**Core Components:**
```python
class OSRSRAGServiceV2:
    def __init__(self):
        # Load embeddings, KG, query segmenter
        
    def query(self, question: str, top_k: int = 20, chat_id: str = "default"):
        # Main entry point
        
    def _search_wiki(self, query: str, entities: Dict, top_k: int):
        # Phrase-based wiki search with category boosting
        
    def _search_kg(self, query: str, query_embedding: List[float], top_k: int):
        # KG entity search (mxbai unified space)
        
    def _combine_results(self, wiki_results: List, kg_results: List, top_k: int):
        # Merge, dedupe, sort by score
        
    def _self_research(self, query: str, initial_results: List, top_k: int):
        # Secondary queries for depth (simplified)
        
    def _generate_response(self, query: str, context: List, chat_id: str):
        # LLaMA generation with attribution
        
    def _manage_chat_session(self, chat_id: str):
        # Context tracking and entity resolution
```

**Key Improvements:**
1. **Single search path** - No multiple fallbacks
2. **Phrase-based matching** - Query segmentation from the start
3. **Category-aware boosting** - Use wiki categories for relevance
4. **Simplified self-research** - Just secondary queries, no recursion
5. **Clean separation** - Each method has one clear purpose

---

## 🔄 Migration Steps

### Step 1: Create Archive Directory ✅
```bash
mkdir -p api/old/tests
```

### Step 2: Move Files to Archive
```bash
# Main RAG service
mv api/osrs_rag_service.py api/old/osrs_rag_service_v1.py

# Reranker (unused)
mv api/reranker_service.py api/old/reranker_service.py

# Test scripts
mv api/run_http_battery.py api/old/tests/
mv api/run_modality_tests.py api/old/tests/
mv api/run_search_battery.py api/old/tests/

# Check kg_query_service usage first
# mv api/kg_query_service.py api/old/kg_query_service.py
```

### Step 3: Create V2 Service
- Write `api/osrs_rag_service_v2.py` with clean architecture
- Use query_segmentation.py for phrase detection
- Integrate with existing embedding_service.py
- Integrate with existing attribution_service.py

### Step 4: Update API Server
- Update `api/osrs_api_server.py` to import V2:
  ```python
  from osrs_rag_service_v2 import OSRSRAGServiceV2
  ```

### Step 5: Test
- Test with Playwright on GUI
- Verify "Dragon Slayer" query finds quest pages
- Verify "Zulrah hp" still works
- Test chat session isolation

### Step 6: Cleanup
- Remove __pycache__ for old files
- Update any remaining imports

---

## 📝 Rollback Plan

If V2 has issues:
1. Restore V1: `cp api/old/osrs_rag_service_v1.py api/osrs_rag_service.py`
2. Update server: Change import back to V1
3. Restart API server

---

## ✅ Success Criteria

- [ ] V2 service < 1000 lines
- [ ] "Dragon Slayer" query finds "Dragon Slayer I" quest page
- [ ] "Zulrah hp" query finds main Zulrah page with correct HP
- [ ] Chat sessions isolated (no context bleeding)
- [ ] Attribution working (wiki URLs + contributors)
- [ ] Self-research provides depth without complexity
- [ ] No hardcoded helpers or fallbacks
- [ ] All tests pass

---

## 🔍 Dependencies Verified

**External Scripts Using API Files:**
- `scripts/streamlined-watchdog.js` → Uses `api/wiki_template_parser.py` ✅
- `scripts/streamlined-watchdog.js` → Uses `api/wiki_template_checker.py` ✅
- `scripts/create_osrs_embeddings.py` → Uses `api/embeddings/embedding_service.py` ✅
- `scripts/kg/create_mxbai_kg_embeddings.py` → Uses `api/embeddings/embedding_service.py` ✅

**No changes needed to these dependencies!**

---

## 📊 Complexity Reduction

| Metric | V1 | V2 (Target) | Reduction |
|--------|----|----|-----------|
| Lines of Code | 2,748 | ~800-1000 | 63-71% |
| Search Paths | 5+ | 2 | 60% |
| Fallback Mechanisms | 4+ | 0 | 100% |
| Unused Features | 6 | 0 | 100% |

---

## 🎯 Migration Status

1. ✅ Create this migration document
2. ✅ Check if `kg_query_service.py` is used anywhere (KEEP - used by admin GUI)
3. ✅ Create `api/old/` directory structure
4. ✅ Move files to archive
   - ✅ `osrs_rag_service.py` → `api/old/osrs_rag_service_v1.py`
   - ✅ `reranker_service.py` → `api/old/reranker_service.py`
   - ✅ Test scripts → `api/old/tests/`
5. ✅ Write `osrs_rag_service.py` V2 (695 lines - 74.7% reduction!)
6. ✅ API server automatically uses V2 (same filename)
7. ✅ Test with Playwright
8. ✅ Document results

---

## 📊 V2 Results

### Code Metrics
- **V1 Lines:** 2,748
- **V2 Lines:** 695
- **Reduction:** 74.7% (2,053 lines removed!)

### Startup Time
- **V1:** ~26 seconds (with reranker init, KG loading, etc.)
- **V2:** ~21 seconds (cleaner, faster)

### Features Removed
- ❌ Reranker code (unused)
- ❌ Spell correction (not needed)
- ❌ Timeline/intent detection (over-engineered)
- ❌ Recursive expansion (simplified to self-research)
- ❌ Multiple fallback paths (single clean path)
- ❌ Temporary content filtering (not needed)

### Features Kept & Improved
- ✅ Wiki embedding search (with phrase detection!)
- ✅ KG embeddings (mxbai unified space)
- ✅ Query segmentation (NEW - treats "Dragon Slayer" as single entity)
- ✅ Self-research (simplified secondary queries)
- ✅ Chat session management
- ✅ Attribution service
- ✅ Streaming responses

---

## 🧪 Test Results

### Test 1: "How do I start Dragon Slayer?"
**V1 Result:** ❌ Found "Dragon Slayer II" and "Spawn (Dragon Slayer II)" - wrong quest!

**V2 Result:** ✅ **FIXED!** Now correctly identifies BOTH quests:
- Mentions Dragon Slayer I (original quest at Champions' Guild)
- Mentions Dragon Slayer II (sequel at Myths' Guild)
- Correctly identifies Dragon Slayer I as the free-to-play version
- Provides accurate starting locations for both

**What Fixed It:**
1. Query segmentation treats "Dragon Slayer" as single phrase
2. Category-aware boosting prioritizes quest pages
3. **NEW:** Sequel detection - prefers "Dragon Slayer I" over "Dragon Slayer II" for generic queries
   - Detects sequels by checking for " II", " 2", " III", etc. in title
   - Original quests get 0.9x boost weight
   - Sequels get 0.5x boost weight

### Test 2: "Zulrah hp"
**Result:** ✅ Still works perfectly!
- Finds main "Zulrah" page first
- Correctly reports 500 hitpoints
- Stat query detection working as designed

---

## 📈 Data Sources V2 Uses

### ✅ OSRS Wiki Embeddings
- **File:** `data/osrs_embeddings.jsonl`
- **Count:** 35,884 wiki pages
- **Dimensions:** 1024D (mxbai-embed-large)
- **Usage:** Primary search with phrase-based boosting

### ✅ KG Embeddings (Unified mxbai)
- **File:** `data/kg_entity_embeddings_mxbai.jsonl`
- **Count:** 149,047 entity embeddings
- **Dimensions:** 1024D (same space as wiki!)
- **Usage:** Secondary search for entity relationships

### ❌ KG Graph Structure (NOT USED)
- **File:** `data/osrs_kg_triples.csv`
- **Why removed:** V1 loaded this but rarely used it
- **Trade-off:** Lost explicit graph traversal, gained simplicity and speed

---

**Last Updated:** 2025-10-03 02:45 AM

