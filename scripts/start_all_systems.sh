#!/bin/bash

###############################################################################
# OSRS AI System - Complete System Startup Script
#
# This script starts all components of the OSRS AI system:
# 1. Streamlined Watchdog (wiki monitoring + GE updates with completion-based orchestration)
# 2. OSRS API Server (Flask API with RAG, embeddings, price history)
# 3. Frontend GUI (React PWA on port 3005)
#
# NOTE: GE updates are now integrated into the watchdog cycle for sequential
#       execution and better API coordination. No separate GE daemon needed!
#
# All processes run in the background with proper logging.
# Use stop_all_systems.sh to gracefully shut down all services.
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/Users/brandon/Documents/projects/GE"
cd "$PROJECT_ROOT"

# Log directory
LOG_DIR="$PROJECT_ROOT/logs/osrs_ai"
mkdir -p "$LOG_DIR"

# PID file directory
PID_DIR="$PROJECT_ROOT/logs/pids"
mkdir -p "$PID_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Starting OSRS AI System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

###############################################################################
# 1. Start Streamlined Watchdog (Wiki Monitoring + GE Updates)
###############################################################################
echo -e "${YELLOW}📡 Starting Streamlined Watchdog (with integrated GE updates)...${NC}"

if [ -f "$PID_DIR/watchdog.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/watchdog.pid")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}   ⚠️  Watchdog already running (PID: $OLD_PID)${NC}"
    else
        echo -e "${YELLOW}   Cleaning up stale PID file${NC}"
        rm "$PID_DIR/watchdog.pid"
    fi
fi

if [ ! -f "$PID_DIR/watchdog.pid" ]; then
    cd "$PROJECT_ROOT/scripts"
    # Run with completion-based orchestration (triggers embeddings after each cycle)
    # All components restored: template checker, KG embeddings, wiki embeddings
    nohup node streamlined-watchdog.js --completion-based > "$LOG_DIR/watchdog.out" 2>&1 &
    WATCHDOG_PID=$!
    echo $WATCHDOG_PID > "$PID_DIR/watchdog.pid"
    echo -e "${GREEN}   ✅ Watchdog started (PID: $WATCHDOG_PID)${NC}"
    echo -e "${GREEN}      Log: $LOG_DIR/watchdog.out${NC}"
    echo -e "${GREEN}      Includes: Wiki monitoring + GE price updates + Template validation + Embeddings${NC}"
else
    echo -e "${GREEN}   ✅ Watchdog already running${NC}"
fi

sleep 2

###############################################################################
# 2. Start OSRS API Server (Flask API)
###############################################################################
echo -e "${YELLOW}🔧 Starting OSRS API Server...${NC}"

if [ -f "$PID_DIR/api_server.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/api_server.pid")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}   ⚠️  API Server already running (PID: $OLD_PID)${NC}"
    else
        echo -e "${YELLOW}   Cleaning up stale PID file${NC}"
        rm "$PID_DIR/api_server.pid"
    fi
fi

# Check if port 5001 is in use
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}   ⚠️  Port 5001 already in use${NC}"
    EXISTING_PID=$(lsof -ti:5001)
    echo -e "${YELLOW}   Existing process PID: $EXISTING_PID${NC}"
    
    # Check if it's our API server
    if [ -f "$PID_DIR/api_server.pid" ]; then
        STORED_PID=$(cat "$PID_DIR/api_server.pid")
        if [ "$EXISTING_PID" == "$STORED_PID" ]; then
            echo -e "${GREEN}   ✅ API Server already running (PID: $EXISTING_PID)${NC}"
        else
            echo -e "${RED}   ❌ Port 5001 occupied by different process${NC}"
            echo -e "${RED}      Kill it manually: kill $EXISTING_PID${NC}"
        fi
    else
        echo -e "${GREEN}   ✅ API Server running (PID: $EXISTING_PID)${NC}"
        echo $EXISTING_PID > "$PID_DIR/api_server.pid"
    fi
else
    cd "$PROJECT_ROOT/api"
    nohup python3 osrs_api_server.py --host 0.0.0.0 --port 5001 > "$LOG_DIR/api.out" 2>&1 &
    API_PID=$!
    echo $API_PID > "$PID_DIR/api_server.pid"
    echo -e "${GREEN}   ✅ API Server started (PID: $API_PID)${NC}"
    echo -e "${GREEN}      Log: $LOG_DIR/api.out${NC}"
    echo -e "${GREEN}      URL: http://localhost:5001${NC}"
fi

sleep 3

###############################################################################
# 3. Start Frontend GUI (React PWA)
###############################################################################
echo -e "${YELLOW}🎨 Starting Frontend GUI...${NC}"

if [ -f "$PID_DIR/frontend.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/frontend.pid")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}   ⚠️  Frontend already running (PID: $OLD_PID)${NC}"
    else
        echo -e "${YELLOW}   Cleaning up stale PID file${NC}"
        rm "$PID_DIR/frontend.pid"
    fi
fi

# Check if port 3005 is in use
if lsof -Pi :3005 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}   ⚠️  Port 3005 already in use${NC}"
    EXISTING_PID=$(lsof -ti:3005 | head -1)
    echo -e "${YELLOW}   Existing process PID: $EXISTING_PID${NC}"
    
    if [ -f "$PID_DIR/frontend.pid" ]; then
        STORED_PID=$(cat "$PID_DIR/frontend.pid")
        if [ "$EXISTING_PID" == "$STORED_PID" ]; then
            echo -e "${GREEN}   ✅ Frontend already running (PID: $EXISTING_PID)${NC}"
        else
            echo -e "${GREEN}   ✅ Frontend running (PID: $EXISTING_PID)${NC}"
            echo $EXISTING_PID > "$PID_DIR/frontend.pid"
        fi
    else
        echo -e "${GREEN}   ✅ Frontend running (PID: $EXISTING_PID)${NC}"
        echo $EXISTING_PID > "$PID_DIR/frontend.pid"
    fi
else
    cd "$PROJECT_ROOT/frontend"
    nohup npm run dev > "$LOG_DIR/frontend.out" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$PID_DIR/frontend.pid"
    echo -e "${GREEN}   ✅ Frontend started (PID: $FRONTEND_PID)${NC}"
    echo -e "${GREEN}      Log: $LOG_DIR/frontend.out${NC}"
    echo -e "${GREEN}      URL: http://localhost:3005${NC}"
fi

sleep 2

###############################################################################
# Summary
###############################################################################
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}✅ All Systems Started!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Running Services:${NC}"
echo ""

if [ -f "$PID_DIR/watchdog.pid" ]; then
    PID=$(cat "$PID_DIR/watchdog.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "  📡 Streamlined Watchdog: ${GREEN}RUNNING${NC} (PID: $PID)"
        echo -e "     ${GREEN}(includes GE updates)${NC}"
    else
        echo -e "  📡 Streamlined Watchdog: ${RED}STOPPED${NC}"
    fi
fi

if [ -f "$PID_DIR/api_server.pid" ]; then
    PID=$(cat "$PID_DIR/api_server.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "  🔧 OSRS API Server:     ${GREEN}RUNNING${NC} (PID: $PID)"
    else
        echo -e "  🔧 OSRS API Server:     ${RED}STOPPED${NC}"
    fi
fi

if [ -f "$PID_DIR/frontend.pid" ]; then
    PID=$(cat "$PID_DIR/frontend.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "  🎨 Frontend GUI:        ${GREEN}RUNNING${NC} (PID: $PID)"
    else
        echo -e "  🎨 Frontend GUI:        ${RED}STOPPED${NC}"
    fi
fi

echo ""
echo -e "${BLUE}Access Points:${NC}"
echo -e "  🌐 Frontend:  ${GREEN}http://localhost:3005${NC}"
echo -e "  🔧 API:       ${GREEN}http://localhost:5001${NC}"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  📄 All logs:  ${GREEN}$LOG_DIR/${NC}"
echo ""
echo -e "${YELLOW}To stop all systems: ${GREEN}./scripts/stop_all_systems.sh${NC}"
echo -e "${YELLOW}To check status:     ${GREEN}./scripts/check_system_status.sh${NC}"
echo ""

