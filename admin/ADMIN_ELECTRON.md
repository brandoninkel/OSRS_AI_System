# OSRS AI System - Electron Admin GUI Documentation

## Table of Contents
1. [Overview](#overview)
2. [Why Electron?](#why-electron)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Features](#features)
6. [Architecture](#architecture)
7. [Building](#building)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The OSRS AI Admin Control Center is a **modern Electron-based desktop application** that provides centralized monitoring and control for all OSRS AI system components.

### Key Features

✅ **Real-Time Monitoring**
- Live system resource tracking (CPU, Memory, Disk)
- Circular progress rings with color-coded alerts
- Process status with PID display
- Current operation tracking with progress bars

✅ **System Control**
- Start/Stop all services with one click
- Quick status checks
- Direct frontend access
- Log file viewing

✅ **Live Log Streaming**
- Real-time log tailing from all services
- Tabbed interface (All, Watchdog, API, System)
- Auto-scroll with 1000-line buffer
- Timestamp prefixing

✅ **Modern UI**
- Futuristic dark theme with neon accents
- Smooth animations and transitions
- Glowing effects and shadows
- Responsive design (desktop and mobile)

✅ **Security**
- No API dependencies (uses shell scripts directly)
- Context isolation
- Secure IPC communication
- Process sandboxing

---

## Why Electron?

### Problems with PyQt6

| Issue | PyQt6 | Electron |
|-------|-------|----------|
| **File Watching** | Manual polling (2s delay) | Native `chokidar` (instant) |
| **Hot Reload** | ❌ No | ✅ Yes (dev mode) |
| **Debugging** | Limited | Chrome DevTools |
| **UI Framework** | Qt widgets (complex) | HTML/CSS (easy) |
| **Process Management** | subprocess (basic) | Node child_process (advanced) |
| **Real-time Updates** | Polling-based | Event-driven |
| **Cross-platform** | ⚠️ Platform issues | ✅ Excellent |
| **Packaging** | Complex, unsigned | electron-builder |
| **Updates** | Manual | auto-updater |
| **Modern UI** | Difficult | Easy (CSS/Tailwind) |
| **App Bundle** | Security restrictions | Works perfectly |

### Electron Advantages

1. **Native File Watching** - Instant updates when status files change
2. **Real-Time Log Streaming** - Live log tailing with `tail` library
3. **Chrome DevTools** - Full debugging capabilities
4. **Modern UI** - Easy to create futuristic designs with CSS
5. **Cross-Platform** - Works identically on macOS, Windows, Linux
6. **Easy Packaging** - One command to build native apps
7. **Auto-Updates** - Built-in update mechanism
8. **Better Process Management** - Advanced child process control

---

## Installation

### Prerequisites
- Node.js 16+ (check with `node --version`)
- npm (comes with Node.js)

### Install Dependencies

```bash
cd admin/electron-app
npm install
```

This installs:
- `electron` - Desktop app framework
- `chokidar` - File watching
- `systeminformation` - System metrics
- `tail` - Log tailing
- `tree-kill` - Process management
- `electron-builder` - App packaging

---

## Usage

### Development Mode

```bash
cd admin/electron-app
npm run dev
```

This opens the app with DevTools enabled for debugging.

### Production Mode

```bash
cd admin/electron-app
npm start
```

This runs the app in production mode (no DevTools).

### Quick Launch Script

For convenience, you can create a launch script:

```bash
# Create launcher
cat > admin/launch_electron_admin.command << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/electron-app"
npm start
EOF

chmod +x admin/launch_electron_admin.command
```

Then double-click `launch_electron_admin.command` to start the app.

---

## Features

### 1. System Control Panel

**Buttons:**
- **START ALL** - Launches watchdog, API server, and frontend
- **STOP ALL** - Gracefully shuts down all services
- **STATUS** - Displays detailed system status
- **OPEN GUI** - Opens frontend at http://localhost:3005

**How it works:**
- Executes shell scripts in `scripts/` directory
- No API calls (direct script execution for security)
- Shows output in system logs

### 2. System Resources

**Monitored Metrics:**
- **CPU** - Current load percentage
- **Memory** - Used/Total GB and percentage
- **Disk** - Used/Total GB and percentage

**Display:**
- Circular progress rings
- Color-coded alerts:
  - Green (0-70%): Normal
  - Yellow (70-90%): Warning
  - Red (90-100%): Critical
- Updates every 2 seconds

### 3. Process Status

**Monitored Processes:**
- **Watchdog** - Wiki monitoring + GE updates
- **API Server** - Flask API with RAG
- **Frontend** - React PWA

**Display:**
- Running/Stopped status
- PID when running
- Color-coded status dots
- Updates every 3 seconds

### 4. Current Operation

**Shows:**
- Current task name
- Progress percentage
- Status text (e.g., "150/300 items")
- ETA and rate

**Sources:**
- `logs/watchdog_status.json`
- `logs/attribution_status.json`
- `logs/ge_status.json`

**Updates:**
- Real-time via file watching
- Instant updates when files change

### 5. Live Logs

**Tabs:**
- **ALL** - Combined logs from all services
- **WATCHDOG** - Wiki monitoring and GE updates
- **API** - RAG API and backend operations
- **SYSTEM** - General system logs

**Features:**
- Real-time streaming (no polling)
- Auto-scroll to bottom
- 1000-line buffer per tab
- Timestamp prefixing
- Monospace font for readability

**Log Files:**
- Watchdog: `logs/osrs_ai/watchdog.out`
- API: `logs/osrs_ai/api.out`
- System: `logs/osrs_ai/system.out`

---

## Architecture

### Process Model

```
┌─────────────────────────────────────────────────────────────┐
│                     ELECTRON APP                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Main Process    │◄───IPC──►│ Renderer Process │          │
│  │  (Node.js)       │         │  (Browser/UI)    │          │
│  └──────────────────┘         └──────────────────┘          │
│         │                              │                     │
│         │                              │                     │
│    ┌────▼────┐                    ┌───▼────┐                │
│    │ System  │                    │   UI   │                │
│    │ Monitor │                    │ Logic  │                │
│    └────┬────┘                    └────────┘                │
│         │                                                    │
│    ┌────▼────┐                                              │
│    │  File   │                                              │
│    │ Watcher │                                              │
│    └────┬────┘                                              │
│         │                                                    │
│    ┌────▼────┐                                              │
│    │  Log    │                                              │
│    │ Tailer  │                                              │
│    └─────────┘                                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌─────────┐          ┌─────────┐         ┌─────────┐
   │Watchdog │          │   API   │         │Frontend │
   │ Process │          │ Process │         │ Process │
   └─────────┘          └─────────┘         └─────────┘
```

### File Structure

```
electron-app/
├── main.js          # Main process (Node.js backend)
│   ├── Window management
│   ├── System monitoring (systeminformation)
│   ├── Process checking (PID files)
│   ├── File watching (chokidar)
│   ├── Log tailing (tail)
│   └── IPC handlers
│
├── preload.js       # Secure IPC bridge
│   └── Exposes safe APIs to renderer
│
├── renderer.js      # Renderer process (UI logic)
│   ├── Event handlers
│   ├── Display updates
│   ├── Log management
│   └── State management
│
├── index.html       # UI structure
├── styles.css       # Futuristic styling
├── package.json     # Dependencies & build config
└── README.md        # Quick reference
```

### Security Model

1. **Context Isolation** - Renderer can't access Node.js directly
2. **Preload Script** - Only exposes whitelisted APIs
3. **No Remote Module** - Prevents arbitrary code execution
4. **Shell Script Execution** - No API calls (direct script execution)
5. **Process Sandboxing** - Each process runs in isolation

---

## Building

### Build for Current Platform

```bash
cd admin/electron-app
npm run build
```

### Build for Specific Platform

**macOS:**
```bash
npm run build:mac
```
Creates:
- `dist/OSRS AI Admin.app` - Application bundle
- `dist/OSRS AI Admin.dmg` - Installer

**Windows:**
```bash
npm run build:win
```
Creates:
- `dist/OSRS AI Admin Setup.exe` - Installer
- `dist/OSRS AI Admin.exe` - Portable

**Linux:**
```bash
npm run build:linux
```
Creates:
- `dist/OSRS AI Admin.AppImage` - Portable
- `dist/osrs-ai-admin.deb` - Debian package

### Build Configuration

Edit `package.json` to customize:
- App name and version
- Icon paths
- Build targets
- Installer options

---

## Troubleshooting

### App won't start

```bash
# Check Node.js version
node --version  # Should be 16+

# Reinstall dependencies
cd admin/electron-app
rm -rf node_modules package-lock.json
npm install
```

### Logs not showing

- Check that services are writing to `logs/osrs_ai/*.out`
- Verify log files exist: `ls -la logs/osrs_ai/`
- Check file permissions: `chmod 644 logs/osrs_ai/*.out`

### Process status not updating

- Verify PID files exist: `ls -la logs/pids/`
- Check PID file contents: `cat logs/pids/watchdog.pid`
- Ensure processes are running: `ps aux | grep watchdog`

### Status not updating

- Check status files exist: `ls -la logs/*.json`
- Verify file format: `cat logs/watchdog_status.json`
- Check file watcher errors in DevTools console

### DevTools not opening

```bash
# Run in development mode
NODE_ENV=development npm start
```

### Build fails

```bash
# Install electron-builder globally
npm install -g electron-builder

# Try building again
npm run build
```

---

## Comparison: PyQt6 vs Electron

| Feature | PyQt6 | Electron |
|---------|-------|----------|
| **Technology** | Python + Qt | Node.js + Chromium |
| **UI Framework** | Qt Widgets | HTML/CSS/JS |
| **File Watching** | Manual polling | Native (chokidar) |
| **Log Streaming** | File position tracking | Native (tail) |
| **Debugging** | Print statements | Chrome DevTools |
| **Hot Reload** | ❌ No | ✅ Yes |
| **Packaging** | Complex | electron-builder |
| **App Size** | ~50 MB | ~150 MB |
| **Startup Time** | Fast (~1s) | Medium (~2s) |
| **Memory Usage** | Low (~100 MB) | Medium (~200 MB) |
| **Cross-Platform** | ⚠️ Issues | ✅ Excellent |
| **Modern UI** | Difficult | Easy |
| **Updates** | Manual | auto-updater |
| **Community** | Small | Large |

**Verdict**: Electron is better for this use case due to real-time requirements, modern UI needs, and cross-platform support.

---

## Next Steps

1. **Install dependencies**: `cd admin/electron-app && npm install`
2. **Run in dev mode**: `npm run dev`
3. **Test all features**: Start/stop services, check logs, monitor resources
4. **Build standalone app**: `npm run build:mac` (or your platform)
5. **Move old PyQt6 files to `admin/old/`** for cleanup

---

## Support

For issues or questions:
1. Check this documentation
2. Check `electron-app/README.md`
3. Open DevTools (`Cmd+Shift+I`) and check Console
4. Check main process logs in terminal


