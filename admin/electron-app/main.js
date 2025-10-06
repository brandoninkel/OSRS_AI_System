const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, exec } = require('child_process');
const kill = require('tree-kill');
const chokidar = require('chokidar');
const si = require('systeminformation');
const { Tail } = require('tail');

// Project paths
const PROJECT_ROOT = path.join(__dirname, '..', '..');
const SCRIPTS_DIR = path.join(PROJECT_ROOT, 'scripts');
const LOGS_DIR = path.join(PROJECT_ROOT, 'logs');
const OSRS_LOGS_DIR = path.join(LOGS_DIR, 'osrs_ai');
const PIDS_DIR = path.join(LOGS_DIR, 'pids');

// Process tracking
const processes = new Map();
const logTails = new Map();

let mainWindow = null;

// Create main window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    backgroundColor: '#0a0e1a',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset',
    frame: false,
    show: false
  });

  mainWindow.loadFile('index.html');

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
    stopAllProcesses();
  });
}

app.whenReady().then(() => {
  createWindow();
  startMonitoring();
});

app.on('window-all-closed', () => {
  stopAllProcesses();
  app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ============================================================================
// SYSTEM MONITORING
// ============================================================================

function startMonitoring() {
  // Monitor system resources every 2 seconds
  setInterval(async () => {
    try {
      const cpu = await si.currentLoad();
      const mem = await si.mem();
      const disk = await si.fsSize();
      
      const stats = {
        cpu: Math.round(cpu.currentLoad),
        memory: {
          used: Math.round(mem.used / 1024 / 1024 / 1024 * 10) / 10,
          total: Math.round(mem.total / 1024 / 1024 / 1024 * 10) / 10,
          percent: Math.round(mem.used / mem.total * 100)
        },
        disk: disk[0] ? {
          used: Math.round(disk[0].used / 1024 / 1024 / 1024 * 10) / 10,
          total: Math.round(disk[0].size / 1024 / 1024 / 1024 * 10) / 10,
          percent: Math.round(disk[0].use)
        } : { used: 0, total: 0, percent: 0 }
      };

      if (mainWindow) {
        mainWindow.webContents.send('system-stats', stats);
      }
    } catch (error) {
      console.error('Error getting system stats:', error);
    }
  }, 2000);

  // Monitor process status every 3 seconds
  setInterval(() => {
    updateProcessStatus();
  }, 3000);

  // Watch status files
  watchStatusFiles();
  
  // Start log tailing
  startLogTailing();
}

// ============================================================================
// PROCESS MANAGEMENT
// ============================================================================

function updateProcessStatus() {
  const status = {
    watchdog: checkProcess('watchdog'),
    api: checkProcess('api_server'),
    frontend: checkProcess('frontend')
  };

  if (mainWindow) {
    mainWindow.webContents.send('process-status', status);
  }
}

function checkProcess(name) {
  const pidFile = path.join(PIDS_DIR, `${name}.pid`);
  
  try {
    if (!fs.existsSync(pidFile)) {
      return { running: false, pid: null };
    }

    const pid = parseInt(fs.readFileSync(pidFile, 'utf8').trim());
    
    // Check if process is actually running
    try {
      process.kill(pid, 0); // Signal 0 checks if process exists
      return { running: true, pid };
    } catch (e) {
      return { running: false, pid: null };
    }
  } catch (error) {
    return { running: false, pid: null };
  }
}

// ============================================================================
// STATUS FILE WATCHING
// ============================================================================

function watchStatusFiles() {
  const statusFiles = [
    path.join(LOGS_DIR, 'watchdog_status.json'),
    path.join(LOGS_DIR, 'attribution_status.json'),
    path.join(LOGS_DIR, 'ge_status.json')
  ];

  const watcher = chokidar.watch(statusFiles, {
    persistent: true,
    ignoreInitial: false,
    awaitWriteFinish: {
      stabilityThreshold: 100,
      pollInterval: 50
    }
  });

  watcher.on('change', (filePath) => {
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const type = path.basename(filePath, '.json').replace('_status', '');
      
      if (mainWindow) {
        mainWindow.webContents.send('status-update', { type, data });
      }
    } catch (error) {
      console.error(`Error reading status file ${filePath}:`, error);
    }
  });
}

// ============================================================================
// LOG TAILING
// ============================================================================

function startLogTailing() {
  const logFiles = {
    watchdog: path.join(OSRS_LOGS_DIR, 'watchdog.out'),
    api: path.join(OSRS_LOGS_DIR, 'api.out'),
    system: path.join(OSRS_LOGS_DIR, 'system.out')
  };

  Object.entries(logFiles).forEach(([type, filePath]) => {
    if (fs.existsSync(filePath)) {
      const tail = new Tail(filePath, {
        fromBeginning: false,
        follow: true,
        useWatchFile: true
      });

      tail.on('line', (line) => {
        if (mainWindow) {
          mainWindow.webContents.send('log-line', { type, line });
        }
      });

      tail.on('error', (error) => {
        console.error(`Error tailing ${type} log:`, error);
      });

      logTails.set(type, tail);
    }
  });
}

// ============================================================================
// IPC HANDLERS
// ============================================================================

ipcMain.handle('start-all', async () => {
  return executeScript('start_all_systems.sh');
});

ipcMain.handle('stop-all', async () => {
  return executeScript('stop_all_systems.sh');
});

ipcMain.handle('check-status', async () => {
  return executeScript('check_system_status.sh');
});

ipcMain.handle('open-frontend', async () => {
  exec('open http://localhost:3005');
  return { success: true };
});

ipcMain.handle('open-log', async (event, logType) => {
  const logFile = path.join(OSRS_LOGS_DIR, `${logType}.out`);
  exec(`open "${logFile}"`);
  return { success: true };
});

function executeScript(scriptName) {
  return new Promise((resolve) => {
    const scriptPath = path.join(SCRIPTS_DIR, scriptName);
    
    exec(`bash "${scriptPath}"`, { cwd: PROJECT_ROOT }, (error, stdout, stderr) => {
      if (error) {
        resolve({ success: false, error: error.message, output: stderr });
      } else {
        resolve({ success: true, output: stdout });
      }
    });
  });
}

function stopAllProcesses() {
  // Stop all log tails
  logTails.forEach((tail) => {
    tail.unwatch();
  });
  logTails.clear();

  // Kill tracked processes
  processes.forEach((proc, name) => {
    if (proc && proc.pid) {
      kill(proc.pid, 'SIGTERM');
    }
  });
  processes.clear();
}

