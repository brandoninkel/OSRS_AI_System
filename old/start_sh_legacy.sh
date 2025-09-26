#!/bin/bash

echo "🚀 Starting OSRS AI System..."

# Check if Python dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Start API server in background
echo "🧠 Starting RAG API server on localhost:5001..."
cd api && python osrs_api_server.py &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start GUI server in background  
echo "🖥️  Starting GUI server on localhost:3002..."
cd ../gui && python serve-rag-gui.py &
GUI_PID=$!

echo ""
echo "✅ OSRS AI System is running!"
echo "🌐 GUI: http://localhost:3002"
echo "🔌 API: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop all servers..."

# Wait for interrupt
trap "echo '🛑 Stopping servers...'; kill $API_PID $GUI_PID; exit" INT
wait
