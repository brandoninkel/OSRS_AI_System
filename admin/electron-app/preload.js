const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // System control
  startAll: () => ipcRenderer.invoke('start-all'),
  stopAll: () => ipcRenderer.invoke('stop-all'),
  checkStatus: () => ipcRenderer.invoke('check-status'),
  openFrontend: () => ipcRenderer.invoke('open-frontend'),
  openLog: (logType) => ipcRenderer.invoke('open-log', logType),

  // Event listeners
  onSystemStats: (callback) => {
    ipcRenderer.on('system-stats', (event, stats) => callback(stats));
  },
  onProcessStatus: (callback) => {
    ipcRenderer.on('process-status', (event, status) => callback(status));
  },
  onStatusUpdate: (callback) => {
    ipcRenderer.on('status-update', (event, data) => callback(data));
  },
  onLogLine: (callback) => {
    ipcRenderer.on('log-line', (event, data) => callback(data));
  }
});

