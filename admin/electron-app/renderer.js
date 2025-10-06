// ============================================================================
// OSRS AI Admin - Renderer Process
// ============================================================================

// State
const state = {
  logs: {
    all: [],
    watchdog: [],
    api: [],
    system: []
  },
  currentTab: 'all',
  systemStats: null,
  processStatus: null,
  currentOperation: {
    task: 'System Idle',
    progress: 0,
    status: 'Ready'
  }
};

// DOM Elements
const elements = {
  // Buttons
  startAllBtn: document.getElementById('startAllBtn'),
  stopAllBtn: document.getElementById('stopAllBtn'),
  statusBtn: document.getElementById('statusBtn'),
  frontendBtn: document.getElementById('frontendBtn'),
  
  // Status
  systemStatus: document.getElementById('systemStatus'),
  currentTime: document.getElementById('currentTime'),
  
  // Resources
  cpuValue: document.getElementById('cpuValue'),
  cpuCircle: document.getElementById('cpuCircle'),
  cpuText: document.getElementById('cpuText'),
  memValue: document.getElementById('memValue'),
  memCircle: document.getElementById('memCircle'),
  memText: document.getElementById('memText'),
  diskValue: document.getElementById('diskValue'),
  diskCircle: document.getElementById('diskCircle'),
  diskText: document.getElementById('diskText'),
  
  // Processes
  watchdogProcess: document.getElementById('watchdogProcess'),
  apiProcess: document.getElementById('apiProcess'),
  frontendProcess: document.getElementById('frontendProcess'),
  
  // Operation
  operationTask: document.getElementById('operationTask'),
  operationProgress: document.getElementById('operationProgress'),
  operationPercent: document.getElementById('operationPercent'),
  operationStatus: document.getElementById('operationStatus'),
  
  // Logs
  logAll: document.getElementById('logAll'),
  logWatchdog: document.getElementById('logWatchdog'),
  logApi: document.getElementById('logApi'),
  logSystem: document.getElementById('logSystem'),
  tabBtns: document.querySelectorAll('.tab-btn')
};

// ============================================================================
// INITIALIZATION
// ============================================================================

function init() {
  setupEventListeners();
  setupElectronListeners();
  startClock();
  updateSystemStatus();
}

function setupEventListeners() {
  // Control buttons
  elements.startAllBtn.addEventListener('click', handleStartAll);
  elements.stopAllBtn.addEventListener('click', handleStopAll);
  elements.statusBtn.addEventListener('click', handleCheckStatus);
  elements.frontendBtn.addEventListener('click', handleOpenFrontend);
  
  // Tab buttons
  elements.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
    });
  });
}

function setupElectronListeners() {
  // System stats updates
  window.electronAPI.onSystemStats((stats) => {
    state.systemStats = stats;
    updateResourceDisplay();
  });
  
  // Process status updates
  window.electronAPI.onProcessStatus((status) => {
    state.processStatus = status;
    updateProcessDisplay();
    updateSystemStatus();
  });
  
  // Status updates (watchdog, attribution, etc.)
  window.electronAPI.onStatusUpdate(({ type, data }) => {
    if (data.active) {
      state.currentOperation = {
        task: data.task || 'Processing',
        progress: data.progress || 0,
        status: data.status || ''
      };
      updateOperationDisplay();
    } else {
      state.currentOperation = {
        task: 'System Idle',
        progress: 0,
        status: 'Ready'
      };
      updateOperationDisplay();
    }
  });
  
  // Log lines
  window.electronAPI.onLogLine(({ type, line }) => {
    addLogLine(type, line);
  });
}

// ============================================================================
// BUTTON HANDLERS
// ============================================================================

async function handleStartAll() {
  elements.startAllBtn.disabled = true;
  elements.startAllBtn.textContent = 'STARTING...';
  
  try {
    const result = await window.electronAPI.startAll();
    if (result.success) {
      addLogLine('system', '✅ All systems started successfully');
    } else {
      addLogLine('system', `❌ Error starting systems: ${result.error}`);
    }
  } catch (error) {
    addLogLine('system', `❌ Error: ${error.message}`);
  } finally {
    elements.startAllBtn.disabled = false;
    elements.startAllBtn.innerHTML = '<span class="btn-icon">▶</span>START ALL';
  }
}

async function handleStopAll() {
  elements.stopAllBtn.disabled = true;
  elements.stopAllBtn.textContent = 'STOPPING...';
  
  try {
    const result = await window.electronAPI.stopAll();
    if (result.success) {
      addLogLine('system', '✅ All systems stopped successfully');
    } else {
      addLogLine('system', `❌ Error stopping systems: ${result.error}`);
    }
  } catch (error) {
    addLogLine('system', `❌ Error: ${error.message}`);
  } finally {
    elements.stopAllBtn.disabled = false;
    elements.stopAllBtn.innerHTML = '<span class="btn-icon">■</span>STOP ALL';
  }
}

async function handleCheckStatus() {
  try {
    const result = await window.electronAPI.checkStatus();
    if (result.success) {
      addLogLine('system', result.output);
    }
  } catch (error) {
    addLogLine('system', `❌ Error checking status: ${error.message}`);
  }
}

async function handleOpenFrontend() {
  try {
    await window.electronAPI.openFrontend();
    addLogLine('system', '🌐 Opening frontend GUI...');
  } catch (error) {
    addLogLine('system', `❌ Error opening frontend: ${error.message}`);
  }
}

// ============================================================================
// DISPLAY UPDATES
// ============================================================================

function updateResourceDisplay() {
  if (!state.systemStats) return;
  
  const { cpu, memory, disk } = state.systemStats;
  
  // CPU
  elements.cpuValue.textContent = `${cpu}%`;
  elements.cpuText.textContent = `${cpu}%`;
  updateProgressRing(elements.cpuCircle, cpu);
  
  // Memory
  elements.memValue.textContent = `${memory.used}/${memory.total} GB`;
  elements.memText.textContent = `${memory.percent}%`;
  updateProgressRing(elements.memCircle, memory.percent);
  
  // Disk
  elements.diskValue.textContent = `${disk.used}/${disk.total} GB`;
  elements.diskText.textContent = `${disk.percent}%`;
  updateProgressRing(elements.diskCircle, disk.percent);
}

function updateProgressRing(circle, percent) {
  const circumference = 2 * Math.PI * 45; // radius = 45
  const offset = circumference - (percent / 100) * circumference;
  circle.style.strokeDashoffset = offset;
  
  // Change color based on usage
  if (percent > 90) {
    circle.style.stroke = 'var(--accent-danger)';
  } else if (percent > 70) {
    circle.style.stroke = 'var(--accent-warning)';
  } else {
    circle.style.stroke = 'var(--accent-primary)';
  }
}

function updateProcessDisplay() {
  if (!state.processStatus) return;
  
  updateProcessItem(elements.watchdogProcess, state.processStatus.watchdog);
  updateProcessItem(elements.apiProcess, state.processStatus.api);
  updateProcessItem(elements.frontendProcess, state.processStatus.frontend);
}

function updateProcessItem(element, status) {
  const statusEl = element.querySelector('.process-status');
  const statusText = statusEl.querySelector('.status-text');
  
  if (status && status.running) {
    element.classList.add('running');
    statusText.textContent = `RUNNING (PID ${status.pid})`;
  } else {
    element.classList.remove('running');
    statusText.textContent = 'STOPPED';
  }
}

function updateOperationDisplay() {
  elements.operationTask.textContent = state.currentOperation.task;
  elements.operationProgress.style.width = `${state.currentOperation.progress}%`;
  elements.operationPercent.textContent = `${state.currentOperation.progress}%`;
  elements.operationStatus.textContent = state.currentOperation.status;
}

function updateSystemStatus() {
  if (!state.processStatus) return;
  
  const anyRunning = Object.values(state.processStatus).some(p => p && p.running);
  
  if (anyRunning) {
    elements.systemStatus.classList.add('online');
    elements.systemStatus.classList.remove('offline');
    elements.systemStatus.querySelector('.status-text').textContent = 'ONLINE';
  } else {
    elements.systemStatus.classList.remove('online');
    elements.systemStatus.classList.add('offline');
    elements.systemStatus.querySelector('.status-text').textContent = 'OFFLINE';
  }
}

// ============================================================================
// LOGS
// ============================================================================

function addLogLine(type, line) {
  const timestamp = new Date().toLocaleTimeString();
  const logLine = `[${timestamp}] ${line}\n`;
  
  // Add to appropriate log
  state.logs[type].push(logLine);
  state.logs.all.push(logLine);
  
  // Keep only last 1000 lines
  if (state.logs[type].length > 1000) {
    state.logs[type].shift();
  }
  if (state.logs.all.length > 1000) {
    state.logs.all.shift();
  }
  
  // Update display
  updateLogDisplay(type);
  updateLogDisplay('all');
}

function updateLogDisplay(type) {
  const logView = elements[`log${type.charAt(0).toUpperCase() + type.slice(1)}`];
  if (logView) {
    logView.textContent = state.logs[type].join('');
    logView.scrollTop = logView.scrollHeight;
  }
}

function switchTab(tab) {
  state.currentTab = tab;
  
  // Update tab buttons
  elements.tabBtns.forEach(btn => {
    if (btn.dataset.tab === tab) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  
  // Update log views
  document.querySelectorAll('.log-view').forEach(view => {
    view.classList.remove('active');
  });
  elements[`log${tab.charAt(0).toUpperCase() + tab.slice(1)}`].classList.add('active');
}

// ============================================================================
// UTILITIES
// ============================================================================

function startClock() {
  function updateClock() {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });
    elements.currentTime.textContent = time;
  }
  
  updateClock();
  setInterval(updateClock, 1000);
}

// ============================================================================
// START
// ============================================================================

init();

