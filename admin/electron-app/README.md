# OSRS AI Admin Control Center

A modern, futuristic Electron-based admin GUI for monitoring and controlling the OSRS AI system.

## Features

### 🎮 System Control
- **Start All Systems** - Launch watchdog, API server, and frontend with one click
- **Stop All Systems** - Gracefully shut down all services
- **Status Check** - View detailed system status
- **Open Frontend** - Quick access to the main GUI

### 📊 Real-Time Monitoring
- **System Resources** - Live CPU, memory, and disk usage with circular progress rings
- **Process Status** - Monitor running processes (Watchdog, API, Frontend)
- **Current Operation** - See what the system is currently doing with progress tracking
- **Live Logs** - Real-time log streaming from all services

### 🎨 Modern UI
- **Futuristic Dark Theme** - Professional cyberpunk-inspired design
- **Responsive Layout** - Adapts to desktop and mobile screens
- **Smooth Animations** - Polished transitions and effects
- **Glowing Effects** - Neon accents and shadows

## Installation

```bash
cd admin/electron-app
npm install
```

## Usage

### Development Mode
```bash
npm run dev
```

### Production Mode
```bash
npm start
```

### Build Standalone App

**macOS:**
```bash
npm run build:mac
```

**Windows:**
```bash
npm run build:win
```

**Linux:**
```bash
npm run build:linux
```

Built apps will be in `admin/electron-app/dist/`

## Architecture

```
electron-app/
├── main.js          # Main process (Node.js)
│   ├── System monitoring (CPU, memory, disk)
│   ├── Process management
│   ├── File watching (status files)
│   └── Log tailing
│
├── preload.js       # Secure IPC bridge
│   └── Exposes safe APIs to renderer
│
├── renderer.js      # Renderer process (UI logic)
│   ├── Event handlers
│   ├── Display updates
│   └── Log management
│
├── index.html       # UI structure
├── styles.css       # Futuristic styling
└── package.json     # Dependencies
```

## How It Works

### 1. Main Process (main.js)
- Runs Node.js with full system access
- Monitors system resources using `systeminformation`
- Watches status files with `chokidar`
- Tails log files with `tail`
- Manages child processes
- Sends updates to renderer via IPC

### 2. Renderer Process (renderer.js)
- Runs in browser context (isolated)
- Receives updates from main process
- Updates UI in real-time
- Handles user interactions
- Manages log display

### 3. Security
- **Context Isolation** - Renderer can't access Node.js directly
- **Preload Script** - Only exposes safe APIs
- **No API Calls** - Uses shell scripts directly (no API manipulation)

## Monitored Services

### Watchdog
- **Purpose**: Wiki monitoring + GE price updates
- **PID File**: `logs/pids/watchdog.pid`
- **Log File**: `logs/osrs_ai/watchdog.out`
- **Status File**: `logs/watchdog_status.json`

### API Server
- **Purpose**: Flask API with RAG
- **Port**: 5001
- **PID File**: `logs/pids/api_server.pid`
- **Log File**: `logs/osrs_ai/api.out`

### Frontend
- **Purpose**: React PWA
- **Port**: 3005
- **PID File**: `logs/pids/frontend.pid`
- **Log File**: `logs/osrs_ai/system.out`

## Status File Format

The admin GUI monitors JSON status files written by services:

```json
{
  "active": true,
  "task": "Processing Main namespace",
  "progress": 45,
  "status": "150/300 items",
  "eta": "30s",
  "rate": "5.2/s"
}
```

## Log Tabs

- **ALL** - Combined logs from all services
- **WATCHDOG** - Wiki monitoring and GE updates
- **API** - RAG API and backend operations
- **SYSTEM** - General system logs

## Keyboard Shortcuts

- `Cmd/Ctrl + R` - Reload app
- `Cmd/Ctrl + Q` - Quit app
- `Cmd/Ctrl + Shift + I` - Open DevTools (development mode)

## Troubleshooting

### App won't start
```bash
# Check Node.js version (requires 16+)
node --version

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Logs not showing
- Check that services are writing to `logs/osrs_ai/*.out`
- Verify log files exist and have content
- Check file permissions

### Process status not updating
- Verify PID files exist in `logs/pids/`
- Check that processes are actually running
- Ensure PID files contain valid PIDs

### Status not updating
- Check that services write status files to `logs/*.json`
- Verify status file format matches expected structure
- Check file watcher is working (no errors in console)

## Development

### Enable DevTools
```bash
NODE_ENV=development npm start
```

### Debug Main Process
Add to `main.js`:
```javascript
console.log('Debug info:', data);
```

### Debug Renderer Process
Open DevTools and check Console tab

### Hot Reload
The app doesn't support hot reload. Restart after changes:
```bash
# Kill app
pkill -f "electron.*osrs-ai-admin"

# Restart
npm run dev
```

## Building for Distribution

### macOS
```bash
npm run build:mac
```
Creates:
- `dist/OSRS AI Admin.app` - Application bundle
- `dist/OSRS AI Admin.dmg` - Installer

### Windows
```bash
npm run build:win
```
Creates:
- `dist/OSRS AI Admin Setup.exe` - Installer
- `dist/OSRS AI Admin.exe` - Portable

### Linux
```bash
npm run build:linux
```
Creates:
- `dist/OSRS AI Admin.AppImage` - Portable
- `dist/osrs-ai-admin.deb` - Debian package

## License

MIT

