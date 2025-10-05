#!/bin/bash

###############################################################################
# OSRS AI System - Status Checker
# 
# Checks the status of all OSRS AI system components.
###############################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/Users/brandon/Documents/projects/GE"
PID_DIR="$PROJECT_ROOT/logs/pids"
LOG_DIR="$PROJECT_ROOT/logs/osrs_ai"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📊 OSRS AI System Status${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check process status
check_process() {
    local NAME=$1
    local PID_FILE=$2
    local PORT=$3
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "  $NAME: ${GREEN}RUNNING${NC} (PID: $PID)"
            
            # Check port if specified
            if [ ! -z "$PORT" ]; then
                if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
                    echo -e "    Port $PORT: ${GREEN}LISTENING${NC}"
                else
                    echo -e "    Port $PORT: ${RED}NOT LISTENING${NC}"
                fi
            fi
        else
            echo -e "  $NAME: ${RED}STOPPED${NC} (stale PID: $PID)"
        fi
    else
        echo -e "  $NAME: ${RED}NOT RUNNING${NC} (no PID file)"
    fi
}

# Check all services
echo -e "${YELLOW}Services:${NC}"
check_process "📡 Streamlined Watchdog" "$PID_DIR/watchdog.pid"
check_process "💰 GE Update Daemon   " "$PID_DIR/ge_daemon.pid"
check_process "🔧 OSRS API Server    " "$PID_DIR/api_server.pid" "5001"
check_process "🎨 Frontend GUI       " "$PID_DIR/frontend.pid" "3005"

echo ""
echo -e "${YELLOW}Recent Log Activity:${NC}"

if [ -f "$LOG_DIR/api.out" ]; then
    echo -e "  ${BLUE}API Server (last 3 lines):${NC}"
    tail -3 "$LOG_DIR/api.out" | sed 's/^/    /'
fi

if [ -f "$LOG_DIR/ge_daemon.out" ]; then
    echo -e "  ${BLUE}GE Daemon (last 3 lines):${NC}"
    tail -3 "$LOG_DIR/ge_daemon.out" | sed 's/^/    /'
fi

echo ""
echo -e "${YELLOW}Database Status:${NC}"
if [ -f "$PROJECT_ROOT/data/price_history.db" ]; then
    DB_SIZE=$(du -h "$PROJECT_ROOT/data/price_history.db" | cut -f1)
    echo -e "  💾 price_history.db: ${GREEN}EXISTS${NC} ($DB_SIZE)"
else
    echo -e "  💾 price_history.db: ${RED}NOT FOUND${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"

