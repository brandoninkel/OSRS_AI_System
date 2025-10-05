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
import re
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))
# Use V3 Agentic RAG with LangGraph
from osrs_agentic_rag import OSRSAgenticRAG
from attribution_service import WikiAttributionService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OSRSAPIServer:
    def __init__(self, host='localhost', port=5001):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for GUI integration

        self.host = host
        self.port = port

        # Initialize V3 Agentic RAG service
        logger.info("Initializing OSRS Agentic RAG V3 service...")
        self.rag_service = OSRSAgenticRAG()
        logger.info("✅ OSRS Agentic RAG V3 service initialized")

        # Initialize attribution service
        self.attribution_service = WikiAttributionService()
        logger.info("✅ Attribution service initialized")

        # Initialize price history service
        from price_history import get_price_history_service
        self.price_history_service = get_price_history_service()
        logger.info("✅ Price history service initialized")

        # Initialize API queue manager
        from api_queue_manager import get_api_queue_manager
        self.queue_manager = get_api_queue_manager()
        logger.info("✅ API queue manager initialized")

        # Initialize citation tool for wiki page search
        from citation_tool import get_citation_tool
        self.citation_tool = get_citation_tool()
        logger.info("✅ Citation tool initialized")

        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'service': 'OSRS Agentic RAG API V3',
                'timestamp': datetime.now().isoformat(),
                'version': 'v3-langgraph'
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

                logger.info(f"Processing chat query: {query[:50]}...")

                # Process query through V3 Agentic RAG service
                result = self.rag_service.query(
                    question=query,
                    show_reasoning=True  # Always show reasoning for debugging
                )

                # Format response for GUI compatibility
                response = {
                    'response': result['answer'],
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }

                if 'sources' in result:
                    response['sources'] = result['sources']

                # Add reasoning if available
                if 'reasoning' in result:
                    response['reasoning'] = result['reasoning']
                if 'tool_calls' in result:
                    response['tool_calls'] = result['tool_calls']

                # Add citations if available
                if 'citations' in result:
                    response['citations'] = result['citations']

                return jsonify(response)

            except Exception as e:
                logger.error(f"Error processing chat request: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/attributions', methods=['POST'])
        def get_attributions():
            """Generate attributions using citation data from AI response"""
            try:
                data = request.get_json()

                if not data or 'citations' not in data:
                    return jsonify({
                        'error': 'Missing citations parameter'
                    }), 400

                citations = data['citations']

                logger.info(f"Generating attributions for {len(citations)} citations...")

                attributions = []

                # For each citation, look up the contributor using the exact source text
                for citation in citations:
                    source_title = citation.get('source_title', '')
                    source_text = citation.get('source_text', '')  # Exact text from wiki
                    paraphrased_text = citation.get('text', '')
                    start = citation.get('start', 0)
                    end = citation.get('end', 0)

                    if not source_title or not source_text:
                        continue

                    # Get attribution info for the exact source text
                    logger.info(f"Finding attribution for: {source_text[:50]}... from {source_title}")
                    result = self.attribution_service.find_attribution(
                        page_title=source_title,
                        snippet=source_text
                    )

                    if result.get('found'):
                        attributions.append({
                            'text': paraphrased_text,
                            'start': start,
                            'end': end,
                            'source_title': source_title,
                            'source_url': f"https://oldschool.runescape.wiki/w/{source_title.replace(' ', '_')}",
                            'excerpt': result.get('snippet', source_text),  # The exact text from wiki
                            'author': result.get('author', 'Unknown'),
                            'timestamp': result.get('timestamp', ''),
                            'revision_url': result.get('wiki_url', ''),
                            'is_original_author': True,  # We now find the actual original author
                            'comment': result.get('comment', ''),
                            'revision_id': result.get('revision_id', ''),
                            'section': result.get('section', ''),
                            'line_number': result.get('line_number', ''),
                            'context': result.get('context', [])
                        })
                    else:
                        # Even if attribution not found, include the citation
                        logger.warning(f"Attribution not found for: {source_text[:50]}...")
                        attributions.append({
                            'text': paraphrased_text,
                            'start': start,
                            'end': end,
                            'source_title': source_title,
                            'source_url': f"https://oldschool.runescape.wiki/w/{source_title.replace(' ', '_')}",
                            'excerpt': source_text,
                            'author': 'Unknown',
                            'timestamp': '',
                            'revision_url': '',
                            'is_original_author': False,
                            'comment': '',
                            'revision_id': '',
                            'section': '',
                            'line_number': '',
                            'context': []
                        })

                return jsonify({
                    'attributions': attributions,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error generating attributions: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/search', methods=['POST'])
        def search():
            """Search for similar OSRS content (V3 uses tools internally)"""
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

                logger.info(f"Processing search query: {query[:50]}...")

                # V3 uses agent-based search, so just return query result
                result = self.rag_service.query(question=query, show_reasoning=False)

                return jsonify({
                    'results': result.get('sources', []),
                    'query': query,
                    'total_results': len(result.get('sources', [])),
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
                # V3 uses global search instance
                from osrs_agentic_rag import _search

                return jsonify({
                    'embeddings_loaded': len(_search.embeddings_data),
                    'total_embeddings': len(_search.embeddings_data),
                    'kg_embeddings_loaded': len(_search.kg_embeddings_data),
                    'embedding_dimension': _search.embeddings_matrix.shape[1] if _search.embeddings_matrix is not None else 0,
                    'llama_model': 'gpt-oss:20b',
                    'embedding_model': 'mxbai-embed-large:latest',
                    'version': 'v3-langgraph',
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
            """Streaming chat endpoint with V3 agentic reasoning"""
            try:
                data = request.get_json()

                if not data or 'query' not in data:
                    return jsonify({'error': 'Missing query parameter'}), 400

                query = data['query'].strip()
                if not query:
                    return jsonify({'error': 'Empty query'}), 400

                logger.info(f"[stream] Processing query: {query}")

                def generate_progress():
                    """Generator function for streaming agent reasoning and responses"""
                    import sys
                    try:
                        # Stream agent's reasoning and tool calls
                        for evt in self.rag_service.query_stream(query):
                            try:
                                # Map V3 events to GUI format
                                if evt['type'] == 'tool_call':
                                    payload = {
                                        'stage': 'searching',
                                        'progress': 30,
                                        'tool': evt['tool'],
                                        'args': evt['args'],
                                        'message': f"🔍 Calling {evt['tool']}..."
                                    }
                                elif evt['type'] == 'tool_result':
                                    payload = {
                                        'stage': 'analyzing',
                                        'progress': 60,
                                        'message': '✅ Tool execution complete'
                                    }
                                elif evt['type'] == 'answer':
                                    payload = {
                                        'stage': 'generating',
                                        'progress': 90,
                                        'message': '🤖 Generating answer...'
                                    }
                                elif evt['type'] == 'complete':
                                    # Final completion event with answer and sources
                                    payload = {
                                        'stage': 'complete',
                                        'progress': 100,
                                        'response': evt.get('answer', ''),
                                        'sources': evt.get('sources', []),
                                        'message': '✅ Complete'
                                    }
                                    logger.info(f"[stream] Sending completion event")
                                elif evt['type'] == 'error':
                                    payload = {
                                        'stage': 'error',
                                        'progress': 0,
                                        'message': f"❌ Error: {evt.get('message', 'Unknown error')}"
                                    }
                                else:
                                    payload = evt

                                data = f"data: {json.dumps(payload)}\n\n"
                                logger.info(f"[stream] Yielding: {payload.get('stage', 'unknown')}")
                                yield data
                                sys.stdout.flush()  # Force flush
                            except Exception as e:
                                logger.error(f"Event serialization error: {e}")
                                yield f"data: {json.dumps({'stage': 'error', 'progress': 0, 'message': 'Serialization error'})}\n\n"
                    except Exception as e:
                        logger.exception("Streaming chat error")
                        yield f"data: {json.dumps({'stage': 'error', 'progress': 0, 'message': f'Error: {str(e)}'})}\n\n"
                    finally:
                        logger.info(f"[stream] Generator complete")

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
            """Get conversation context (V3 doesn't have persistent chat sessions yet)"""
            try:
                return jsonify({
                    'message': 'V3 uses stateless agentic RAG - each query is independent',
                    'version': 'v3-langgraph',
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

        @self.app.route('/economic/price-history', methods=['GET'])
        def get_price_history():
            """Get price history for an item"""
            try:
                item_name = request.args.get('item')
                hours = int(request.args.get('hours', 24))

                if not item_name:
                    return jsonify({
                        'error': 'Missing item parameter',
                        'success': False
                    }), 400

                history = self.price_history_service.get_price_history(item_name, hours)
                trend = self.price_history_service.get_price_trend(item_name, hours)

                return jsonify({
                    'item': item_name,
                    'history': history,
                    'trend': trend,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting price history: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False
                }), 500

        @self.app.route('/economic/compare', methods=['POST'])
        def compare_items():
            """Compare price trends for multiple items"""
            try:
                data = request.get_json()
                items = data.get('items', [])
                hours = data.get('hours', 24)

                if not items:
                    return jsonify({
                        'error': 'Missing items array',
                        'success': False
                    }), 400

                trends = self.price_history_service.get_multiple_trends(items, hours)

                return jsonify({
                    'items': items,
                    'trends': trends,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error comparing items: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False
                }), 500

        @self.app.route('/economic/tracked_items', methods=['GET'])
        def get_tracked_items():
            """Get list of items that have been tracked"""
            try:
                limit = request.args.get('limit', 100, type=int)
                items = self.price_history_service.get_tracked_items(limit)

                return jsonify({
                    'items': items,
                    'count': len(items),
                    'success': True
                })

            except Exception as e:
                logger.error(f"Error getting tracked items: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False
                }), 500

        @self.app.route('/wiki/search', methods=['GET'])
        def search_wiki_pages():
            """Search wiki pages by title for autocomplete"""
            try:
                query = request.args.get('q', '')
                limit = request.args.get('limit', 100, type=int)

                if not query:
                    return jsonify({
                        'error': 'Missing query parameter',
                        'success': False
                    }), 400

                # Search wiki pages using citation tool
                results = self.citation_tool.search_page_titles(query, limit)

                return jsonify({
                    'query': query,
                    'results': results,
                    'count': len(results),
                    'success': True
                })

            except Exception as e:
                logger.error(f"Error searching wiki pages: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False
                }), 500

        @self.app.route('/watchdog/status', methods=['POST'])
        def set_watchdog_status():
            """Signal watchdog active/inactive status"""
            try:
                data = request.get_json()
                active = data.get('active', False)

                # Run async function in sync context
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.queue_manager.set_watchdog_active(active))
                loop.close()

                return jsonify({
                    'success': True,
                    'watchdog_active': active,
                    'message': 'Watchdog status updated',
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error setting watchdog status: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False
                }), 500

        @self.app.route('/queue/stats', methods=['GET'])
        def get_queue_stats():
            """Get API queue statistics"""
            try:
                stats = self.queue_manager.get_stats()

                return jsonify({
                    'success': True,
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting queue stats: {e}")
                return jsonify({
                    'error': str(e),
                    'success': False
                }), 500

    def run(self, debug=False):
        """Start the API server"""
        logger.info(f"🚀 Starting OSRS RAG API server on {self.host}:{self.port}")
        logger.info(f"📊 V3 Agentic RAG with LangGraph")
        logger.info(f"🤖 Using GPT-OSS model: gpt-oss:20b (OpenAI's open-source model for agentic tasks)")
        logger.info(f"🔍 Using embedding model: mxbai-embed-large:latest")

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
