# OSRS AI System - Admin GUI Documentation

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [PyQt6 Admin GUI](#pyqt6-admin-gui)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Features](#features)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Admin GUI provides a professional control panel for managing the OSRS AI system. Built with PyQt6, it offers:

- **Modern Dark Theme**: Professional catppuccin-inspired color scheme
- **Real-Time Monitoring**: CPU, memory, and disk usage tracking
- **Process Management**: Start/stop all services with proper cleanup
- **Shell Script Integration**: Uses system scripts directly (no API calls for security)
- **Service Status Checking**: Real-time status of all running services
- **Log File Access**: Quick access to watchdog and API logs
- **Comprehensive Logging**: Tabbed interface for different log types
- **Responsive Layout**: Resizable sections with splitter bars
- **System Tray Integration**: Minimize to system tray
- **Intelligent Lifecycle**: Automatic cleanup of spawned processes

**Current Version**: PyQt6 Admin GUI (Professional v2.0)
**Security Model**: Direct shell script execution (no API exposure)

---

## Directory Structure

```
admin/
├── ADMIN.md                        # This documentation
├── pyqt6_admin_gui.py              # ✅ ACTIVE - Professional admin GUI (1,083 lines)
├── start-pyqt6-gui.command         # ✅ ACTIVE - Launch script
└── old/                            # Archived files
    ├── admin_gui.py                # ❌ Old DearPyGui version
    ├── modern_admin_gui.py         # ❌ Old DearPyGui modern version
    ├── compare_guis.py             # ❌ GUI comparison script
    ├── test_gui_cleanup.py         # ❌ Test script
    ├── test_responsive_gui.py      # ❌ Test script
    ├── start-admin.command         # ❌ Old launcher
    ├── start-modern-gui.command    # ❌ Old launcher
    ├── gui_button_fixes.md         # ❌ Old documentation
    ├── gui_improvements_summary.md # ❌ Old documentation
    └── pyqt6_gui_features.md       # ❌ Old documentation
```

**Active Files**: 2 files (1 Python + 1 shell script)  
**Archived Files**: 10 files in `admin/old/`

---

## PyQt6 Admin GUI

### Overview

**File**: `admin/pyqt6_admin_gui.py` (1,083 lines)

**Purpose**: Professional control panel for OSRS AI system management

**Technology Stack**:
- **PyQt6** - Modern Qt6 bindings for Python
- **psutil** - System monitoring (CPU, memory, disk)
- **subprocess** - Process management
- **threading** - Background tasks

---

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PyQt6 Admin GUI                           │
│                  (pyqt6_admin_gui.py)                        │
└───────┬─────────────────────────────────────────────────────┘
        │
        ├─► Shell Script Integration (Security: No API)
        │   ├─ scripts/start_all_systems.sh
        │   ├─ scripts/stop_all_systems.sh
        │   └─ scripts/check_system_status.sh
        │
        ├─► ProcessManager
        │   ├─ Track spawned processes
        │   ├─ Monitor PIDs
        │   └─ Cleanup on exit
        │
        ├─► SystemMonitor (QThread)
        │   ├─ CPU usage
        │   ├─ Memory usage
        │   └─ Disk usage
        │
        ├─► LogMonitor (QThread)
        │   ├─ API logs
        │   ├─ System logs
        │   └─ Error logs
        │
        └─► Services Managed
            ├─ Streamlined Watchdog (wiki + GE updates)
            ├─ OSRS API Server (Flask + RAG)
            └─ Frontend GUI (React PWA on port 3005)
```

---

### Key Components

#### 1. ProcessManager
**Purpose**: Manages process lifecycle with proper cleanup

**Features**:
- Start/stop processes by name
- Track PIDs for all spawned processes
- Automatic cleanup on exit (atexit handlers)
- Signal handling (SIGTERM, SIGINT)
- Prevents orphaned processes

**Methods**:
```python
start_process(name, command, cwd) -> bool
stop_process(name) -> bool
is_running(name) -> bool
cleanup_all_processes() -> None
```

---

#### 2. SystemMonitor (QThread)
**Purpose**: Real-time system resource monitoring

**Metrics**:
- **CPU Usage**: Overall system CPU percentage
- **Memory Usage**: RAM usage (used/total)
- **Disk Usage**: Storage usage (used/total)

**Update Frequency**: Every 2 seconds

**Signals**:
```python
stats_updated = pyqtSignal(dict)  # Emits: {cpu, memory, disk}
```

---

#### 3. LogMonitor (QThread)
**Purpose**: Monitor log files and display updates

**Log Types**:
- **API Logs**: `logs/osrs_ai/api.out`
- **System Logs**: `logs/osrs_ai/system.out`
- **Error Logs**: `logs/osrs_ai/error.out`

**Features**:
- Tail-like behavior (shows last 1000 lines)
- Auto-scroll to bottom
- Color-coded messages
- Tabbed interface

**Signals**:
```python
log_updated = pyqtSignal(str, str)  # (log_type, content)
```

---

### UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  🚀 OSRS AI System Control Center                [_][□][X]  │
├─────────────────────────────────────────────────────────────┤
│  File  Help                                                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │  📊 System Status                                    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │ CPU: 45% │ │ RAM: 8GB │ │ Disk:50% │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  🎮 Control Panel                                    │   │
│  │  [🚀 Start All Services]  [🛑 Stop All Services]    │   │
│  │  [📊 Check Status] [🌐 Open Frontend]               │   │
│  │  [📡 Watchdog Log] [🔧 API Log]                     │   │
│  │  Status: ✅ All services running                    │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  📋 Logs                                             │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │ [API] [System] [Errors]                     │    │   │
│  │  ├─────────────────────────────────────────────┤    │   │
│  │  │ [2025-10-05] 🚀 Starting all services...   │    │   │
│  │  │ [2025-10-05] ✅ Watchdog started (PID:1234)│    │   │
│  │  │ [2025-10-05] ✅ API Server started         │    │   │
│  │  │ [2025-10-05] ✅ Frontend started           │    │   │
│  │  │                                              │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Ready | CPU: 45% | Memory: 8.2GB | Disk: 50%              │
└─────────────────────────────────────────────────────────────┘
```

---

### Color Scheme

**Modern Dark Theme** (Catppuccin-inspired):

| Element | Color | Hex Code |
|---------|-------|----------|
| Background | Dark Blue-Gray | `#1e1e2e` |
| Surface | Medium Gray | `#313244` |
| Primary Text | Light Blue-Gray | `#cdd6f4` |
| Success | Green | `#a6e3a1` |
| Error | Red | `#f38ba8` |
| Warning | Orange | `#fab387` |
| Info | Blue | `#89b4fa` |
| Accent | Purple | `#cba6f7` |

---

### Features

#### 1. Service Control
- **Start All Services**: Launches all OSRS AI services via shell script
  - Streamlined Watchdog (wiki monitoring + GE price updates every 5 min)
  - OSRS API Server (Flask API with RAG on port 5001)
  - Frontend GUI (React PWA on port 3005)
- **Stop All Services**: Gracefully stops all services via shell script
- **Check Status**: Real-time status check of all running services
- **Status Indicator**: Live status display in GUI
- **Security**: Uses shell scripts directly (no API exposure for admin functions)

#### 2. System Monitoring
- **CPU Usage**: Real-time CPU percentage
- **Memory Usage**: RAM usage with total/used display
- **Disk Usage**: Storage usage with total/used display
- **Progress Bars**: Visual representation of usage
- **Auto-Update**: Refreshes every 2 seconds

#### 3. Log Viewing
- **Tabbed Interface**: Separate tabs for API, System, Errors
- **Auto-Scroll**: Automatically scrolls to latest logs
- **Color Coding**: Different colors for log levels
- **Search**: Find text in logs (Ctrl+F)
- **Clear**: Clear log display (Ctrl+L)

#### 4. Window Management
- **Resizable**: Drag splitter bars to resize sections
- **Minimize to Tray**: Minimize to system tray
- **Always on Top**: Keep window on top (optional)
- **Remember Size**: Saves window size/position

#### 5. Keyboard Shortcuts
- **Ctrl+Q**: Quit application
- **Ctrl+R**: Refresh system stats
- **Ctrl+L**: Clear logs
- **Ctrl+F**: Find in logs
- **F5**: Refresh all

---

## Installation & Setup

### Prerequisites

**Required**:
- Python 3.10+
- PyQt6
- psutil

**Optional**:
- System tray support (for minimize to tray)

---

### Installation Steps

#### 1. Install PyQt6
```bash
# Using system Python (recommended)
/usr/bin/python3 -m pip install PyQt6 psutil

# Or using project Python
python3 -m pip install PyQt6 psutil
```

#### 2. Verify Installation
```bash
python3 -c "from PyQt6.QtWidgets import QApplication; print('✅ PyQt6 installed')"
```

#### 3. Make Launch Script Executable
```bash
cd /Users/brandon/Documents/projects/GE/admin
chmod +x start-pyqt6-gui.command
```

---

### Quick Start

**Option 1: Double-click in Finder**
```
Navigate to: /Users/brandon/Documents/projects/GE/admin/
Double-click: start-pyqt6-gui.command
```

**Option 2: Terminal**
```bash
cd /Users/brandon/Documents/projects/GE/admin
./start-pyqt6-gui.command
```

**Option 3: Direct Python**
```bash
cd /Users/brandon/Documents/projects/GE/admin
python3 pyqt6_admin_gui.py
```

---

## Usage Guide

### Starting Services

1. **Launch the GUI**:
   ```bash
   cd /Users/brandon/Documents/projects/GE/admin
   ./start-pyqt6-gui.command
   ```

2. **Click "🚀 Start All Services"**:
   - Executes `scripts/start_all_systems.sh`
   - Starts Streamlined Watchdog (wiki monitoring + GE updates)
   - Starts OSRS API Server (Flask + RAG on port 5001)
   - Starts Frontend GUI (React PWA on port 3005)
   - Updates status indicator

3. **Monitor System**:
   - Watch CPU/Memory/Disk usage in real-time
   - Check logs in tabbed interface
   - Click "📊 Check Status" to verify all services

4. **Access Services**:
   - Click "🌐 Open Frontend" to open http://localhost:3005
   - Click "📡 Watchdog Log" to view wiki monitoring logs
   - Click "🔧 API Log" to view API server logs

---

### Stopping Services

1. **Click "🛑 Stop All Services"**:
   - Executes `scripts/stop_all_systems.sh`
   - Stops Frontend GUI
   - Stops OSRS API Server
   - Stops Streamlined Watchdog
   - Updates status indicator

2. **Or Close the GUI**:
   - Automatically stops all managed processes
   - Cleanup handlers ensure no orphans
   - Safe shutdown with proper cleanup

---

### Viewing Logs

1. **Select Log Tab**:
   - **API**: RAG API server logs
   - **System**: System-level logs
   - **Errors**: Error messages only

2. **Log Features**:
   - Auto-scrolls to latest
   - Color-coded by level
   - Search with Ctrl+F
   - Clear with Ctrl+L

---

### System Monitoring

**Real-Time Metrics**:
- **CPU Usage**: Overall system CPU percentage
- **Memory Usage**: RAM used/total (GB)
- **Disk Usage**: Storage used/total (GB)

**Update Frequency**: Every 2 seconds

**Visual Indicators**:
- Progress bars show usage percentage
- Colors change based on usage level:
  - Green: < 60%
  - Orange: 60-80%
  - Red: > 80%

---

## Troubleshooting

### Issue: "PyQt6 not found"

**Solution 1: Install with system Python**
```bash
/usr/bin/python3 -m pip install PyQt6 psutil
```

**Solution 2: Install with project Python**
```bash
cd /Users/brandon/Documents/projects/GE
source .venv/bin/activate
pip install PyQt6 psutil
```

**Solution 3: Use different Python**
```bash
# Try pyenv Python
~/.pyenv/shims/python3 -m pip install PyQt6 psutil
```

---

### Issue: "Permission denied" when starting

**Solution**: Make script executable
```bash
chmod +x /Users/brandon/Documents/projects/GE/admin/start-pyqt6-gui.command
```

---

### Issue: Services won't start

**Check 1: Verify shell scripts exist and are executable**
```bash
ls -lh /Users/brandon/Documents/projects/GE/scripts/start_all_systems.sh
ls -lh /Users/brandon/Documents/projects/GE/scripts/stop_all_systems.sh
chmod +x /Users/brandon/Documents/projects/GE/scripts/*.sh
```

**Check 2: Verify data files exist**
```bash
ls -lh /Users/brandon/Documents/projects/GE/data/osrs_embeddings.jsonl
ls -lh /Users/brandon/Documents/projects/GE/data/osrs_wiki_content.jsonl
```

**Check 3: Check logs**
```bash
tail -f /Users/brandon/Documents/projects/GE/logs/osrs_ai/watchdog.out
tail -f /Users/brandon/Documents/projects/GE/logs/osrs_ai/api.out
tail -f /Users/brandon/Documents/projects/GE/logs/osrs_ai/frontend.out
```

**Check 4: Manually test start script**
```bash
cd /Users/brandon/Documents/projects/GE
bash scripts/start_all_systems.sh
```

---

### Issue: GUI freezes or crashes

**Solution 1: Check system resources**
```bash
# Check memory
free -h  # Linux
vm_stat  # macOS

# Check CPU
top
```

**Solution 2: Clear logs**
```bash
rm /Users/brandon/Documents/projects/GE/logs/osrs_ai/*.out
```

**Solution 3: Restart GUI**
```bash
# Kill any stuck processes
pkill -f pyqt6_admin_gui

# Restart
./start-pyqt6-gui.command
```

---

### Issue: Logs not updating

**Check 1: Verify log files exist**
```bash
ls -lh /Users/brandon/Documents/projects/GE/logs/osrs_ai/
```

**Check 2: Check file permissions**
```bash
chmod 644 /Users/brandon/Documents/projects/GE/logs/osrs_ai/*.out
```

**Check 3: Restart log monitor**
- Close and reopen GUI
- Or click "Refresh" button

---

## Archived Files (admin/old/)

These files have been moved to `admin/old/` as they are no longer actively used:

### 1. admin_gui.py
**Status**: ❌ **ARCHIVED** - Old DearPyGui version

**Reason**: Replaced by PyQt6 version with better performance and features

---

### 2. modern_admin_gui.py
**Status**: ❌ **ARCHIVED** - Old DearPyGui modern version

**Reason**: Replaced by PyQt6 version with professional styling

---

### 3. compare_guis.py
**Status**: ❌ **ARCHIVED** - GUI comparison script

**Reason**: Comparison complete, PyQt6 chosen as final version

---

### 4. test_gui_cleanup.py & test_responsive_gui.py
**Status**: ❌ **ARCHIVED** - Test scripts

**Reason**: Testing complete, features integrated into main GUI

---

### 5. start-admin.command & start-modern-gui.command
**Status**: ❌ **ARCHIVED** - Old launch scripts

**Reason**: Replaced by `start-pyqt6-gui.command`

---

### 6. Documentation Files
**Status**: ❌ **ARCHIVED** - Old documentation

**Files**:
- `gui_button_fixes.md`
- `gui_improvements_summary.md`
- `pyqt6_gui_features.md`

**Reason**: Replaced by this comprehensive `ADMIN.md`

---

## Summary

### Active Files

**Production** (2 files):
- `pyqt6_admin_gui.py` (1,083 lines) - Professional admin GUI
- `start-pyqt6-gui.command` (82 lines) - Launch script

**Documentation** (1 file):
- `ADMIN.md` (this file) - Complete documentation

**Total Active**: 3 files

---

### Archived Files

**Old GUIs** (2 files):
- `admin_gui.py` - DearPyGui version
- `modern_admin_gui.py` - DearPyGui modern version

**Test Scripts** (3 files):
- `compare_guis.py`
- `test_gui_cleanup.py`
- `test_responsive_gui.py`

**Old Launchers** (2 files):
- `start-admin.command`
- `start-modern-gui.command`

**Old Documentation** (3 files):
- `gui_button_fixes.md`
- `gui_improvements_summary.md`
- `pyqt6_gui_features.md`

**Total Archived**: 10 files

---

## Technical Details

### Dependencies

**Python Packages**:
```
PyQt6>=6.6.0
psutil>=5.9.0
```

**System Requirements**:
- Python 3.10+
- macOS, Linux, or Windows
- 100MB free RAM
- Display with 1280x720+ resolution

---

### File Locations

**GUI File**: `/Users/brandon/Documents/projects/GE/admin/pyqt6_admin_gui.py`

**Launch Script**: `/Users/brandon/Documents/projects/GE/admin/start-pyqt6-gui.command`

**Logs Directory**: `/Users/brandon/Documents/projects/GE/logs/osrs_ai/`

**Data Directory**: `/Users/brandon/Documents/projects/GE/data/`

---

### Process Management

**PID Files** (managed by shell scripts):
- `logs/pids/watchdog.pid` - Streamlined Watchdog
- `logs/pids/api.pid` - OSRS API Server
- `logs/pids/frontend.pid` - Frontend GUI

**Shell Scripts**:
- `scripts/start_all_systems.sh` - Start all services
- `scripts/stop_all_systems.sh` - Stop all services
- `scripts/check_system_status.sh` - Check service status

**Security Model**:
- Admin GUI uses shell scripts directly (no API calls)
- Prevents API-based manipulation of admin functions
- Local-only access for system management
- PID-based process tracking and cleanup

**Cleanup Strategy**:
1. Shell scripts manage PID files
2. GUI tracks spawned processes in memory
3. On exit, GUI cleanup handlers run
4. Shell scripts handle graceful shutdown
5. All cleanup actions logged

---

### Performance

**Resource Usage**:
- **Memory**: ~50-100MB (GUI only)
- **CPU**: <1% (idle), ~5% (active monitoring)
- **Startup Time**: ~2-3 seconds

**Optimization**:
- Background threads for monitoring
- Efficient log tailing (last 1000 lines)
- Lazy loading of UI elements
- Minimal redraws

---

---

## Security Architecture

### Why No API for Admin Functions?

The Admin GUI uses **direct shell script execution** instead of API calls for security:

**Security Benefits**:
1. **No API Exposure**: Admin functions can't be triggered via HTTP requests
2. **Local-Only Access**: Only users with filesystem access can manage services
3. **No Authentication Bypass**: Can't manipulate admin functions through API vulnerabilities
4. **Process Isolation**: Shell scripts run with proper permissions and isolation
5. **Audit Trail**: All actions logged locally, not through API logs

**What Uses Shell Scripts**:
- ✅ Start All Services (`scripts/start_all_systems.sh`)
- ✅ Stop All Services (`scripts/stop_all_systems.sh`)
- ✅ Check Status (`scripts/check_system_status.sh`)

**What Uses API** (read-only, safe):
- Frontend GUI (user-facing features)
- RAG queries (AI chat)
- Price data queries (market intelligence)

**Attack Surface Reduction**:
```
❌ OLD: API endpoint → Admin function → System control
   Risk: Anyone with API access can control system

✅ NEW: GUI → Shell script → System control
   Risk: Only local users with filesystem access
```

---

**Last Updated**: October 5, 2025
**Version**: 2.0.0
**GUI Version**: PyQt6 Professional (Security-Hardened)
**Security Model**: Direct shell script execution (no API exposure)

