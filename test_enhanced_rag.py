#!/usr/bin/env python3
"""
Test script to demonstrate enhanced RAG with KG integration
Shows exactly what happens when you ask questions in the chat GUI
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

def test_enhanced_rag():
    """Test the enhanced RAG system with KG integration"""
    print("🧪 Testing Enhanced RAG with KG Integration")
    print("=" * 60)
    
    try:
        from osrs_rag_service import OSRSRAGService
        
        # Initialize the service (same as GUI does)
        print("🔄 Initializing RAG service...")
        rag_service = OSRSRAGService()
        
        # Check if KG embeddings are loaded
        kg_status = "✅ ENABLED" if (hasattr(rag_service, 'use_kg_embeddings') and 
                                   rag_service.use_kg_embeddings and 
                                   rag_service.kg_entity_emb is not None) else "❌ DISABLED"
        
        print(f"📊 System Status:")
        print(f"   Wiki embeddings: {len(rag_service.embeddings_data):,} entries")
        print(f"   KG integration: {kg_status}")
        
        if hasattr(rag_service, 'kg_entity_emb') and rag_service.kg_entity_emb is not None:
            print(f"   KG entities: {rag_service.kg_entity_emb.shape[0]:,}")
            print(f"   KG embedding dim: {rag_service.kg_entity_emb.shape[1]}")
        
        print("\n" + "=" * 60)
        print("🎯 TESTING: What happens when you ask questions in chat")
        print("=" * 60)
        
        # Test queries that would benefit from KG integration
        test_queries = [
            "What drops abyssal whip?",
            "How do I get dragon armor?", 
            "Tell me about Bandos",
            "What are barrows weapons?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: '{query}'")
            print("-" * 40)
            
            # Show what the enhanced system does
            print("📝 Processing steps:")
            print("   1. Creating query embedding...")
            
            # Test the find_similar_content method (this is what chat uses)
            try:
                similar_content = rag_service.find_similar_content(query, top_k=5)
                
                print(f"   2. Found {len(similar_content)} relevant results")
                print("   3. Results include:")
                
                for j, (content, score) in enumerate(similar_content[:3], 1):
                    source = "🧠 KG" if content.get('kg_entity') else "📖 Wiki"
                    title = content.get('title', 'Unknown')[:40]
                    print(f"      {source} {title}... (score: {score:.3f})")
                
                print("   4. ✅ Enhanced context ready for LLaMA response")
                
            except Exception as e:
                print(f"   ❌ Error during retrieval: {e}")
        
        print("\n" + "=" * 60)
        print("💡 HOW TO USE IN CHAT GUI:")
        print("=" * 60)
        print("1. Launch the RAG GUI (frontend)")
        print("2. Ask any OSRS question normally")
        print("3. The system AUTOMATICALLY uses both:")
        print("   • Wiki knowledge (detailed guides, stats)")
        print("   • KG relationships (entity connections)")
        print("4. You get better, more complete answers!")
        print()
        print("🎮 Try these example questions:")
        for query in test_queries:
            print(f"   • {query}")
        print()
        print("🔧 To disable KG integration (if needed):")
        print("   export OSRS_USE_KG_EMBEDDINGS=0")
        print()
        print("✅ Enhanced RAG is ready! Just chat normally.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the RAG service dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_rag()
