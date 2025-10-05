#!/bin/bash

###############################################################################
# OSRS AI System - Stop All Systems Script
# 
# Gracefully stops all running OSRS AI system components.
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/Users/brandon/Documents/projects/GE"
PID_DIR="$PROJECT_ROOT/logs/pids"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🛑 Stopping OSRS AI System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to stop a process
stop_process() {
    local NAME=$1
    local PID_FILE=$2
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping $NAME (PID: $PID)...${NC}"
            kill $PID
            sleep 2
            
            # Check if still running
            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${YELLOW}   Force killing $NAME...${NC}"
                kill -9 $PID
            fi
            
            echo -e "${GREEN}   ✅ $NAME stopped${NC}"
        else
            echo -e "${YELLOW}   $NAME not running${NC}"
        fi
        rm "$PID_FILE"
    else
        echo -e "${YELLOW}   $NAME PID file not found${NC}"
    fi
}

# Stop all services
stop_process "Frontend GUI" "$PID_DIR/frontend.pid"
stop_process "OSRS API Server" "$PID_DIR/api_server.pid"
stop_process "GE Update Daemon" "$PID_DIR/ge_daemon.pid"
stop_process "Streamlined Watchdog" "$PID_DIR/watchdog.pid"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ All Systems Stopped${NC}"
echo -e "${GREEN}========================================${NC}"

