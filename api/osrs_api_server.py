#!/usr/bin/env python3
"""
OSRS RAG API Server
Provides HTTP API endpoints for the OSRS RAG service to integrate with the GUI
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import sys
import os
import json
import time
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from osrs_rag_service import OSRSRAGService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OSRSAPIServer:
    def __init__(self, host='localhost', port=5001):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for GUI integration
        
        self.host = host
        self.port = port
        
        # Initialize RAG service
        logger.info("Initializing OSRS RAG service...")
        self.rag_service = OSRSRAGService()
        logger.info("✅ OSRS RAG service initialized")
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'service': 'OSRS RAG API',
                'timestamp': datetime.now().isoformat(),
                'embeddings_loaded': len(self.rag_service.embeddings_data)
            })
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Main chat endpoint for OSRS questions"""
            try:
                data = request.get_json()
                
                if not data or 'query' not in data:
                    return jsonify({
                        'error': 'Missing query parameter'
                    }), 400
                
                query = data['query'].strip()
                if not query:
                    return jsonify({
                        'error': 'Empty query'
                    }), 400
                
                # Get optional parameters
                top_k = data.get('top_k', 5)
                show_sources = data.get('show_sources', True)
                chat_id = data.get('chat_id', None)  # Optional chat session ID

                logger.info(f"Processing chat query: {query[:50]}... (chat_id: {chat_id or 'default'})")

                # Process query through RAG service with chat isolation
                result = self.rag_service.query(
                    question=query,
                    top_k=top_k,
                    show_sources=show_sources,
                    chat_id=chat_id
                )
                
                # Format response for GUI compatibility
                response = {
                    'response': result['response'],
                    'query': result['query'],
                    'timestamp': result['timestamp'],
                    'success': True
                }
                
                if show_sources and 'sources' in result:
                    response['sources'] = result['sources']
                    response['similarity_scores'] = result.get('similarity_scores', [])
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error processing chat request: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/search', methods=['POST'])
        def search():
            """Search for similar OSRS content"""
            try:
                data = request.get_json()
                
                if not data or 'query' not in data:
                    return jsonify({
                        'error': 'Missing query parameter'
                    }), 400
                
                query = data['query'].strip()
                if not query:
                    return jsonify({
                        'error': 'Empty query'
                    }), 400
                
                top_k = data.get('top_k', 10)
                
                logger.info(f"Processing search query: {query[:50]}...")
                
                # Find similar content without generating response
                similar_content = self.rag_service.find_similar_content(query, top_k)
                
                # Format results
                results = []
                for content_data, similarity_score in similar_content:
                    results.append({
                        'title': content_data['title'],
                        'categories': content_data.get('categories', []),
                        'similarity': float(similarity_score),
                        'text_preview': content_data['text'][:200] + "..." if len(content_data['text']) > 200 else content_data['text']
                    })
                
                return jsonify({
                    'results': results,
                    'query': query,
                    'total_results': len(results),
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Error processing search request: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/stats', methods=['GET'])
        def stats():
            """Get service statistics"""
            try:
                cache_stats = self.rag_service.embedding_service.get_cache_stats()
                
                total = len(self.rag_service.embeddings_data)
                return jsonify({
                    'embeddings_loaded': total,
                    'total_embeddings': total,  # alias for GUI compatibility
                    'embedding_dimension': self.rag_service.embeddings_matrix.shape[1] if self.rag_service.embeddings_matrix is not None else 0,
                    'cache_stats': cache_stats,
                    'llama_model': self.rag_service.llama_model,
                    'embedding_model': self.rag_service.embedding_service.config.model_name,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })

            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/chat/stream', methods=['POST'])
        def chat_stream():
            """Streaming chat endpoint with progress updates"""
            try:
                data = request.get_json()

                if not data or 'query' not in data:
                    return jsonify({'error': 'Missing query parameter'}), 400

                query = data['query'].strip()
                if not query:
                    return jsonify({'error': 'Empty query'}), 400

                top_k = data.get('top_k', 5)
                show_sources = data.get('show_sources', True)
                chat_id = data.get('chat_id', None)  # Optional chat session ID

                def generate_progress():
                    """Generator function for streaming REAL progress updates from RAG + LLM"""
                    try:
                        for evt in self.rag_service.query_stream(query, top_k=top_k, show_sources=show_sources, chat_id=chat_id):
                            try:
                                payload = json.dumps(evt)
                            except Exception:
                                payload = json.dumps({"stage": "error", "message": "Serialization error"})
                            yield f"data: {payload}\n\n"
                    except Exception as e:
                        logger.exception("Streaming chat error")
                        yield f"data: {json.dumps({'stage': 'error', 'progress': 0, 'message': f'Error: {str(e)}'})}\n\n"
                return Response(
                    generate_progress(),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type'
                    }
                )

            except Exception as e:
                logger.error(f"Streaming chat setup error: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/context', methods=['GET'])
        def context_info():
            """Get conversation context and window information"""
            try:
                # Get optional chat_id parameter
                chat_id = request.args.get('chat_id', None)

                # Get context information for the specific chat
                context_data = self.rag_service.get_chat_context(chat_id)

                return jsonify({
                    **context_data,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })

            except Exception as e:
                logger.error(f"Error getting context info: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500

    def run(self, debug=False):
        """Start the API server"""
        logger.info(f"🚀 Starting OSRS RAG API server on {self.host}:{self.port}")
        logger.info(f"📊 Loaded {len(self.rag_service.embeddings_data)} OSRS wiki embeddings")
        logger.info(f"🤖 Using LLaMA model: {self.rag_service.llama_model}")
        logger.info(f"🔍 Using embedding model: {self.rag_service.embedding_service.config.model_name}")
        
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=debug,
                threaded=True
            )
        except KeyboardInterrupt:
            logger.info("👋 Shutting down OSRS RAG API server")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OSRS RAG API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and run server
    server = OSRSAPIServer(host=args.host, port=args.port)
    server.run(debug=args.debug)

if __name__ == "__main__":
    main()
