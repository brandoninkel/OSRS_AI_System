#!/usr/bin/env python3
"""
Professional OSRS AI Admin Control Panel - PyQt6 Version
Modern, responsive interface with proper process management and clean design
"""

import sys
import os
import json
import subprocess
import threading
import time
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Optional
import psutil

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QPushButton, QLabel, QTextEdit, QProgressBar,
    QGroupBox, QTabWidget, QFrame, QSplitter, QScrollArea,
    QStatusBar, QMenuBar, QSystemTrayIcon, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect
)
from PyQt6.QtGui import (
    QFont, QIcon, QPalette, QColor, QPixmap, QPainter, 
    QLinearGradient, QBrush, QAction
)

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs" / "osrs_ai"
DATA_DIR = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"
PID_DIR = REPO_ROOT / "logs" / "pids"

# System management scripts (use shell scripts directly, not API)
START_SCRIPT = SCRIPTS_DIR / "start_all_systems.sh"
STOP_SCRIPT = SCRIPTS_DIR / "stop_all_systems.sh"
STATUS_SCRIPT = SCRIPTS_DIR / "check_system_status.sh"

# Global process tracking for cleanup
spawned_processes: List[subprocess.Popen] = []

class ModernColors:
    """Modern dark theme color palette"""
    # Background colors
    DARK_BG = "#1e1e2e"           # Main background
    DARKER_BG = "#181825"         # Darker sections
    SURFACE = "#313244"           # Surface elements
    
    # Accent colors
    BLUE = "#89b4fa"              # Primary blue
    GREEN = "#a6e3a1"             # Success green
    RED = "#f38ba8"               # Error red
    ORANGE = "#fab387"            # Warning orange
    PURPLE = "#cba6f7"            # Purple accent
    YELLOW = "#f9e2af"            # Yellow accent
    
    # Text colors
    TEXT = "#cdd6f4"              # Primary text
    TEXT_DIM = "#9399b2"          # Secondary text
    TEXT_BRIGHT = "#ffffff"       # Bright text
    
    # Border and separator
    BORDER = "#45475a"            # Borders
    SEPARATOR = "#6c7086"         # Separators

class ProcessManager:
    """Handles process lifecycle management with proper cleanup"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.setup_cleanup_handlers()
    
    def setup_cleanup_handlers(self):
        """Register cleanup handlers for graceful shutdown"""
        atexit.register(self.cleanup_all_processes)
        signal.signal(signal.SIGTERM, lambda s, f: self.cleanup_all_processes())
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup_all_processes())
        print("✅ Process cleanup handlers registered")
    
    def start_process(self, name: str, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Start a managed process"""
        try:
            # Kill existing process if running
            self.stop_process(name)
            
            print(f"🚀 Starting process: {name}")
            process = subprocess.Popen(
                command,
                cwd=cwd or REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[name] = process
            print(f"✅ Started {name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False
    
    def stop_process(self, name: str) -> bool:
        """Stop a managed process gracefully"""
        try:
            process = self.processes.get(name)
            if process and process.poll() is None:
                print(f"🛑 Stopping process: {name}")
                process.terminate()
                
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Force killing {name}")
                    process.kill()
                    process.wait()
                
                print(f"✅ Stopped {name}")
            
            if name in self.processes:
                del self.processes[name]
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop {name}: {e}")
            return False
    
    def cleanup_all_processes(self):
        """Clean up all managed processes"""
        print("🧹 Cleaning up all processes...")
        for name in list(self.processes.keys()):
            self.stop_process(name)
        print("✅ Process cleanup completed")
    
    def get_process_status(self, name: str) -> Dict:
        """Get status of a managed process"""
        process = self.processes.get(name)
        if not process:
            return {"status": "stopped", "pid": None, "cpu": 0, "memory": 0}
        
        if process.poll() is not None:
            return {"status": "stopped", "pid": None, "cpu": 0, "memory": 0}
        
        try:
            proc_info = psutil.Process(process.pid)
            return {
                "status": "running",
                "pid": process.pid,
                "cpu": proc_info.cpu_percent(),
                "memory": proc_info.memory_info().rss / 1024 / 1024  # MB
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"status": "stopped", "pid": None, "cpu": 0, "memory": 0}

class StatusUpdateThread(QThread):
    """Background thread for updating system status"""
    status_updated = pyqtSignal(dict)
    
    def __init__(self, process_manager: ProcessManager):
        super().__init__()
        self.process_manager = process_manager
        self.running = True
    
    def run(self):
        """Main thread loop for status updates"""
        while self.running:
            try:
                # Get system status
                status = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "processes": {
                        "orchestrator": self.process_manager.get_process_status("orchestrator"),
                        "api": self.process_manager.get_process_status("api"),
                        "frontend": self.process_manager.get_process_status("frontend"),
                        "watchdog": self.process_manager.get_process_status("watchdog")
                    },
                    "system": self.get_system_stats(),
                    "orchestrator_progress": self.get_orchestrator_status()
                }

                self.status_updated.emit(status)

            except Exception as e:
                print(f"Status update error: {e}")

            self.msleep(2000)  # Update every 2 seconds
    
    def get_system_stats(self) -> Dict:
        """Get system resource statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used": memory.used / 1024 / 1024 / 1024,  # GB
                "memory_total": memory.total / 1024 / 1024 / 1024,  # GB
                "disk_percent": disk.percent,
                "disk_free": disk.free / 1024 / 1024 / 1024  # GB
            }
        except Exception:
            return {"cpu": 0, "memory_percent": 0, "memory_used": 0,
                   "memory_total": 0, "disk_percent": 0, "disk_free": 0}

    def get_orchestrator_status(self) -> Dict:
        """Get orchestrator status from status file"""
        try:
            status_file = REPO_ROOT / "logs" / "orchestrator_status.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading orchestrator status: {e}")

        # Default status if file doesn't exist or can't be read
        return {
            "running": False,
            "current_stage": "idle",
            "queue_length": 0,
            "progress": {
                "current_stage": "idle",
                "stage_progress": 0.0,
                "overall_progress": 0.0,
                "eta_seconds": 0,
                "stages": {
                    "embeddings": {"status": "pending", "progress": 0.0, "eta": 0},
                    "kg_triples": {"status": "pending", "progress": 0.0, "eta": 0},
                    "kg_pykeen": {"status": "pending", "progress": 0.0, "eta": 0},
                    "kg_embeddings": {"status": "pending", "progress": 0.0, "eta": 0}
                }
            }
        }
    
    def stop(self):
        """Stop the status update thread"""
        self.running = False
        self.wait()

# Initialize global process manager
process_manager = ProcessManager()

class OSRSAdminMainWindow(QMainWindow):
    """Main application window with modern design and professional layout"""

    def __init__(self):
        super().__init__()
        self.process_manager = process_manager
        self.status_thread = None

        # Window configuration
        self.setWindowTitle("🚀 OSRS AI System Control Center")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Apply modern dark theme
        self.setup_theme()

        # Create UI components
        self.setup_ui()

        # Start status monitoring
        self.start_status_monitoring()

        # Setup window close handler
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def setup_theme(self):
        """Apply modern dark theme with professional styling"""
        self.setStyleSheet(f"""
            /* Main window styling */
            QMainWindow {{
                background-color: {ModernColors.DARK_BG};
                color: {ModernColors.TEXT};
            }}

            /* Group box styling */
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                color: {ModernColors.TEXT_BRIGHT};
                border: 2px solid {ModernColors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {ModernColors.SURFACE};
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: {ModernColors.BLUE};
            }}

            /* Button styling */
            QPushButton {{
                background-color: {ModernColors.BLUE};
                color: {ModernColors.TEXT_BRIGHT};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                min-height: 30px;
            }}

            QPushButton:hover {{
                background-color: #74c0fc;
            }}

            QPushButton:pressed {{
                background-color: #5c7cfa;
            }}

            QPushButton:disabled {{
                background-color: {ModernColors.BORDER};
                color: {ModernColors.TEXT_DIM};
            }}

            /* Success button variant */
            QPushButton[buttonType="success"] {{
                background-color: {ModernColors.GREEN};
                color: {ModernColors.DARK_BG};
            }}

            QPushButton[buttonType="success"]:hover {{
                background-color: #94d3a2;
            }}

            /* Danger button variant */
            QPushButton[buttonType="danger"] {{
                background-color: {ModernColors.RED};
                color: {ModernColors.TEXT_BRIGHT};
            }}

            QPushButton[buttonType="danger"]:hover {{
                background-color: #f5a3b7;
            }}

            /* Warning button variant */
            QPushButton[buttonType="warning"] {{
                background-color: {ModernColors.ORANGE};
                color: {ModernColors.DARK_BG};
            }}

            QPushButton[buttonType="warning"]:hover {{
                background-color: #fcc89b;
            }}

            /* Label styling */
            QLabel {{
                color: {ModernColors.TEXT};
                font-size: 12px;
            }}

            QLabel[labelType="title"] {{
                color: {ModernColors.TEXT_BRIGHT};
                font-size: 16px;
                font-weight: bold;
            }}

            QLabel[labelType="subtitle"] {{
                color: {ModernColors.BLUE};
                font-size: 14px;
                font-weight: bold;
            }}

            QLabel[labelType="status"] {{
                color: {ModernColors.TEXT_DIM};
                font-size: 11px;
            }}

            /* Progress bar styling */
            QProgressBar {{
                border: 2px solid {ModernColors.BORDER};
                border-radius: 6px;
                text-align: center;
                background-color: {ModernColors.DARKER_BG};
                color: {ModernColors.TEXT_BRIGHT};
                font-weight: bold;
            }}

            QProgressBar::chunk {{
                background-color: {ModernColors.BLUE};
                border-radius: 4px;
            }}

            /* Text edit styling */
            QTextEdit {{
                background-color: {ModernColors.DARKER_BG};
                color: {ModernColors.TEXT};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
                padding: 8px;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 11px;
            }}

            /* Tab widget styling */
            QTabWidget::pane {{
                border: 1px solid {ModernColors.BORDER};
                background-color: {ModernColors.SURFACE};
                border-radius: 6px;
            }}

            QTabBar::tab {{
                background-color: {ModernColors.DARKER_BG};
                color: {ModernColors.TEXT_DIM};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}

            QTabBar::tab:selected {{
                background-color: {ModernColors.BLUE};
                color: {ModernColors.TEXT_BRIGHT};
            }}

            QTabBar::tab:hover {{
                background-color: {ModernColors.SURFACE};
                color: {ModernColors.TEXT};
            }}

            /* Status bar styling */
            QStatusBar {{
                background-color: {ModernColors.DARKER_BG};
                color: {ModernColors.TEXT_DIM};
                border-top: 1px solid {ModernColors.BORDER};
            }}

            /* Frame styling */
            QFrame {{
                background-color: {ModernColors.SURFACE};
                border-radius: 6px;
            }}
        """)

    def setup_ui(self):
        """Create and layout all UI components"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Header section
        header_layout = self.create_header_section()
        main_layout.addLayout(header_layout)

        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section: Control panel and circular resource monitors
        top_widget = self.create_top_section()
        splitter.addWidget(top_widget)

        # Middle section: Unified progress display (installer-style)
        middle_widget = self.create_progress_section()
        splitter.addWidget(middle_widget)

        # Bottom section: Logs (only 2 tabs: Watchdog and System)
        bottom_widget = self.create_logs_section()
        splitter.addWidget(bottom_widget)

        # Set splitter proportions (more space for logs)
        splitter.setSizes([350, 100, 400])
        main_layout.addWidget(splitter)

        # Status bar
        self.setup_status_bar()

    def create_header_section(self) -> QHBoxLayout:
        """Create the header with title and system info"""
        layout = QHBoxLayout()

        # Title
        title_label = QLabel("🚀 OSRS AI System Control Center")
        title_label.setProperty("labelType", "title")
        layout.addWidget(title_label)

        # Spacer
        layout.addStretch()

        # System time and status
        self.time_label = QLabel()
        self.time_label.setProperty("labelType", "status")
        layout.addWidget(self.time_label)

        return layout

    def create_top_section(self) -> QWidget:
        """Create the top section with control panel and circular resource monitors"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(15)

        # Left side: Control Panel
        control_group = self.create_control_panel()
        layout.addWidget(control_group, stretch=2)

        # Right side: Circular Resource Monitors
        resources_panel = self.create_circular_resources()
        layout.addWidget(resources_panel, stretch=1)

        return widget

    def create_control_panel(self) -> QGroupBox:
        """Create the main control panel with action buttons"""
        group = QGroupBox("🎮 Control Panel")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Main action buttons (larger, more prominent)
        self.start_all_btn = QPushButton("🚀 Start All Services")
        self.start_all_btn.setProperty("buttonType", "success")
        self.start_all_btn.setMinimumHeight(45)
        self.start_all_btn.clicked.connect(self.start_all_services)
        layout.addWidget(self.start_all_btn, 0, 0, 1, 2)

        self.stop_all_btn = QPushButton("🛑 Stop All Services")
        self.stop_all_btn.setProperty("buttonType", "danger")
        self.stop_all_btn.setMinimumHeight(45)
        self.stop_all_btn.clicked.connect(self.stop_all_services)
        layout.addWidget(self.stop_all_btn, 0, 2, 1, 2)

        # Secondary action buttons
        self.status_btn = QPushButton("📊 Check Status")
        self.status_btn.setProperty("buttonType", "info")
        self.status_btn.clicked.connect(self.check_system_status)
        layout.addWidget(self.status_btn, 1, 0)

        self.frontend_btn = QPushButton("🌐 Open Frontend")
        self.frontend_btn.clicked.connect(self.open_frontend)
        layout.addWidget(self.frontend_btn, 1, 1)

        self.watchdog_log_btn = QPushButton("� Watchdog Log")
        self.watchdog_log_btn.clicked.connect(self.open_watchdog_log)
        layout.addWidget(self.watchdog_log_btn, 1, 2)

        self.api_log_btn = QPushButton("🔧 API Log")
        self.api_log_btn.clicked.connect(self.open_api_log)
        layout.addWidget(self.api_log_btn, 1, 3)

        # Status display
        self.control_status_label = QLabel("Ready")
        self.control_status_label.setProperty("labelType", "status")
        layout.addWidget(self.control_status_label, 2, 0, 1, 4)

        return group

    def create_circular_resources(self) -> QGroupBox:
        """Create circular resource monitors (CPU, Memory, Disk)"""
        group = QGroupBox("📊 System Resources")
        layout = QHBoxLayout(group)
        layout.setSpacing(20)

        # Create circular progress indicators for each resource
        # CPU
        cpu_widget = QWidget()
        cpu_layout = QVBoxLayout(cpu_widget)
        cpu_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.cpu_circle = QProgressBar()
        self.cpu_circle.setRange(0, 100)
        self.cpu_circle.setValue(0)
        self.cpu_circle.setTextVisible(True)
        self.cpu_circle.setFormat("%p%")
        self.cpu_circle.setMinimumSize(80, 80)
        self.cpu_circle.setMaximumSize(80, 80)
        cpu_layout.addWidget(self.cpu_circle, alignment=Qt.AlignmentFlag.AlignCenter)

        cpu_label = QLabel("CPU")
        cpu_label.setProperty("labelType", "subtitle")
        cpu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cpu_layout.addWidget(cpu_label)

        layout.addWidget(cpu_widget)

        # Memory
        mem_widget = QWidget()
        mem_layout = QVBoxLayout(mem_widget)
        mem_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.memory_circle = QProgressBar()
        self.memory_circle.setRange(0, 100)
        self.memory_circle.setValue(0)
        self.memory_circle.setTextVisible(True)
        self.memory_circle.setFormat("%p%")
        self.memory_circle.setMinimumSize(80, 80)
        self.memory_circle.setMaximumSize(80, 80)
        mem_layout.addWidget(self.memory_circle, alignment=Qt.AlignmentFlag.AlignCenter)

        mem_label = QLabel("Memory")
        mem_label.setProperty("labelType", "subtitle")
        mem_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mem_layout.addWidget(mem_label)

        self.memory_value_label = QLabel("0 GB")
        self.memory_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mem_layout.addWidget(self.memory_value_label)

        layout.addWidget(mem_widget)

        # Disk
        disk_widget = QWidget()
        disk_layout = QVBoxLayout(disk_widget)
        disk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.disk_circle = QProgressBar()
        self.disk_circle.setRange(0, 100)
        self.disk_circle.setValue(0)
        self.disk_circle.setTextVisible(True)
        self.disk_circle.setFormat("%p%")
        self.disk_circle.setMinimumSize(80, 80)
        self.disk_circle.setMaximumSize(80, 80)
        disk_layout.addWidget(self.disk_circle, alignment=Qt.AlignmentFlag.AlignCenter)

        disk_label = QLabel("Disk")
        disk_label.setProperty("labelType", "subtitle")
        disk_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        disk_layout.addWidget(disk_label)

        self.disk_value_label = QLabel("0 GB free")
        self.disk_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        disk_layout.addWidget(self.disk_value_label)

        layout.addWidget(disk_widget)

        return group

    def create_progress_section(self) -> QGroupBox:
        """Create unified progress display (installer-style)"""
        group = QGroupBox("⚙️ Current Operation")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Current task label (what's happening now)
        self.current_task_label = QLabel("System idle")
        self.current_task_label.setProperty("labelType", "subtitle")
        layout.addWidget(self.current_task_label)

        # Progress bar
        self.unified_progress = QProgressBar()
        self.unified_progress.setRange(0, 100)
        self.unified_progress.setValue(0)
        self.unified_progress.setTextVisible(True)
        self.unified_progress.setFormat("%p% - %v/%m")
        layout.addWidget(self.unified_progress)

        # Status line (details + ETA)
        self.progress_status_label = QLabel("Ready")
        self.progress_status_label.setProperty("labelType", "status")
        layout.addWidget(self.progress_status_label)

        return group

    def create_logs_section(self) -> QGroupBox:
        """Create logs section with 2 tabs: Watchdog and System"""
        group = QGroupBox("📋 System Logs")
        layout = QVBoxLayout(group)

        # Create tab widget for logs
        self.log_tabs = QTabWidget()

        # Watchdog logs tab (main - wiki monitoring + GE updates)
        self.watchdog_log = QTextEdit()
        self.watchdog_log.setReadOnly(True)
        self.watchdog_log.setFont(QFont("Menlo", 10))
        self.log_tabs.addTab(self.watchdog_log, "� Watchdog")

        # System logs tab (attribution + GE data processes + general system)
        self.system_log = QTextEdit()
        self.system_log.setReadOnly(True)
        self.system_log.setFont(QFont("Menlo", 10))
        self.log_tabs.addTab(self.system_log, "🖥️ System")

        layout.addWidget(self.log_tabs)

        return group

    def setup_status_bar(self):
        """Setup the status bar with useful information"""
        self.status_bar = self.statusBar()

        # Add permanent widgets to status bar
        self.connection_status = QLabel("🔴 Disconnected")
        self.status_bar.addPermanentWidget(self.connection_status)

        self.process_count = QLabel("Processes: 0")
        self.status_bar.addPermanentWidget(self.process_count)

        # Initial message
        self.status_bar.showMessage("OSRS AI Control Center ready")

    def start_status_monitoring(self):
        """Start the background status monitoring thread"""
        self.status_thread = StatusUpdateThread(self.process_manager)
        self.status_thread.status_updated.connect(self.update_status_display)
        self.status_thread.start()

        # Also start a timer for time updates
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time_display)
        self.time_timer.start(1000)  # Update every second

    def update_time_display(self):
        """Update the time display in the header"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)

    def update_status_display(self, status: Dict):
        """Update all status displays with new data"""
        try:
            # Update circular resource monitors
            system = status.get("system", {})

            # CPU
            cpu_percent = int(system.get("cpu", 0))
            self.cpu_circle.setValue(cpu_percent)

            # Set color based on usage
            if cpu_percent > 80:
                self.cpu_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.RED}; }}")
            elif cpu_percent > 60:
                self.cpu_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.ORANGE}; }}")
            else:
                self.cpu_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.GREEN}; }}")

            # Memory
            memory_percent = int(system.get("memory_percent", 0))
            memory_used = system.get("memory_used", 0)
            self.memory_circle.setValue(memory_percent)
            self.memory_value_label.setText(f"{memory_used:.1f} GB")

            if memory_percent > 80:
                self.memory_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.RED}; }}")
            elif memory_percent > 60:
                self.memory_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.ORANGE}; }}")
            else:
                self.memory_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.GREEN}; }}")

            # Disk
            disk_percent = int(system.get("disk_percent", 0))
            disk_free = system.get("disk_free", 0)
            self.disk_circle.setValue(disk_percent)
            self.disk_value_label.setText(f"{disk_free:.1f} GB free")

            if disk_percent > 80:
                self.disk_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.RED}; }}")
            elif disk_percent > 60:
                self.disk_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.ORANGE}; }}")
            else:
                self.disk_circle.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ModernColors.GREEN}; }}")

            # Update status bar with process count
            processes = status.get("processes", {})
            running_count = sum(1 for p in processes.values() if p.get("status") == "running")
            self.process_count.setText(f"Processes: {running_count}")

            if running_count > 0:
                self.connection_status.setText("🟢 Services Running")
                self.connection_status.setStyleSheet(f"color: {ModernColors.GREEN};")
            else:
                self.connection_status.setText("🔴 Services Stopped")
                self.connection_status.setStyleSheet(f"color: {ModernColors.RED};")

        except Exception as e:
            self.log_message(f"Status update error: {e}", "system")

    def log_message(self, message: str, log_type: str = "system"):
        """Add a message to the appropriate log tab (watchdog or system)"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Route to appropriate tab
        if log_type in ["watchdog", "wiki", "ge"]:
            # Watchdog tab: wiki monitoring, GE updates
            self.watchdog_log.append(formatted_message.strip())
            self.watchdog_log.verticalScrollBar().setValue(
                self.watchdog_log.verticalScrollBar().maximum()
            )
        else:
            # System tab: everything else (attribution, GE data, API, general)
            self.system_log.append(formatted_message.strip())
            self.system_log.verticalScrollBar().setValue(
                self.system_log.verticalScrollBar().maximum()
            )

    def update_orchestrator_logs(self):
        """Update orchestrator logs from log file"""
        try:
            orchestrator_log_file = REPO_ROOT / "logs" / "orchestrator.out"
            if orchestrator_log_file.exists():
                with open(orchestrator_log_file, 'r') as f:
                    content = f.read()

                # Only show last 50 lines to avoid overwhelming the GUI
                lines = content.strip().split('\n')
                recent_lines = lines[-50:] if len(lines) > 50 else lines

                # Clear and update orchestrator log
                self.orchestrator_log.clear()
                for line in recent_lines:
                    if line.strip():
                        self.orchestrator_log.append(line)

                # Auto-scroll to bottom
                self.orchestrator_log.verticalScrollBar().setValue(
                    self.orchestrator_log.verticalScrollBar().maximum()
                )
        except Exception as e:
            print(f"Error reading orchestrator logs: {e}")

    def update_pipeline_progress(self, orchestrator_status: Dict):
        """Update pipeline progress bars with orchestrator status"""
        try:
            progress_data = orchestrator_status.get("progress", {})
            stages = progress_data.get("stages", {})
            current_stage = orchestrator_status.get("current_stage", "idle")
            queue_length = orchestrator_status.get("queue_length", 0)

            # Update overall progress
            overall_progress = progress_data.get("overall_progress", 0.0)
            self.overall_progress.setValue(int(overall_progress * 100))

            # Update individual stage progress bars
            stage_mapping = {
                "prepare": "embeddings",  # Map prepare to first progress bar
                "embeddings": "kg_triples",
                "kg_update": "kg_model",
                "cleanup": "kg_embeddings"
            }

            for orchestrator_stage, gui_stage in stage_mapping.items():
                if orchestrator_stage in stages and gui_stage in self.stage_progress:
                    stage_data = stages[orchestrator_stage]
                    progress = stage_data.get("progress", 0.0)
                    status = stage_data.get("status", "pending")
                    eta = stage_data.get("eta", 0)

                    # Update progress bar
                    self.stage_progress[gui_stage].setValue(int(progress * 100))

                    # Update status label
                    if gui_stage in self.stage_labels:
                        if status == "running":
                            status_text = f"Running (ETA: {eta:.0f}s)" if eta > 0 else "Running"
                        elif status == "completed":
                            status_text = "✅ Complete"
                        elif status == "failed":
                            status_text = "❌ Failed"
                        elif status == "error":
                            status_text = "⚠️ Error"
                        else:
                            status_text = "Pending"

                        self.stage_labels[gui_stage].setText(status_text)

            # Update pipeline info label
            if current_stage == "idle":
                if queue_length > 0:
                    info_text = f"Queue: {queue_length} tasks pending"
                else:
                    info_text = "Pipeline idle - waiting for changes"
            else:
                info_text = f"Current: {current_stage.replace('_', ' ').title()}"

            self.pipeline_info_label.setText(info_text)

        except Exception as e:
            print(f"Error updating pipeline progress: {e}")

    # Action Methods - Connected to button clicks

    def start_all_services(self):
        """
        Start all OSRS AI services using shell script.
        Uses scripts/start_all_systems.sh directly (no API calls for security).

        Services started:
        - Streamlined Watchdog (wiki monitoring + GE updates)
        - OSRS API Server (Flask API with RAG)
        - Frontend GUI (React PWA)
        """
        self.log_message("🚀 Starting all OSRS AI services...", "system")
        self.control_status_label.setText("Starting services...")

        def run_start():
            try:
                # Use the unified start script
                self.log_message(f"� Executing: {START_SCRIPT}", "system")

                result = subprocess.run([
                    "bash", str(START_SCRIPT)
                ], capture_output=True, text=True, cwd=REPO_ROOT, timeout=60)

                if result.returncode == 0:
                    self.log_message("✅ All services started successfully", "system")
                    self.log_message(result.stdout, "system")
                    self.control_status_label.setText("✅ All services running")
                else:
                    self.log_message(f"❌ Failed to start services: {result.stderr}", "system")
                    self.control_status_label.setText("❌ Failed to start")

            except subprocess.TimeoutExpired:
                self.log_message("⚠️ Start command timed out (services may still be starting)", "system")
                self.control_status_label.setText("⚠️ Timeout (check status)")
            except Exception as e:
                self.log_message(f"❌ Error starting services: {e}", "system")
                self.control_status_label.setText("❌ Error occurred")

        # Run in background thread to avoid blocking UI
        threading.Thread(target=run_start, daemon=True).start()

    def stop_all_services(self):
        """
        Stop all OSRS AI services using shell script.
        Uses scripts/stop_all_systems.sh directly (no API calls for security).

        Services stopped:
        - Frontend GUI (React PWA)
        - OSRS API Server (Flask API)
        - Streamlined Watchdog (wiki monitoring + GE updates)
        """
        self.log_message("🛑 Stopping all OSRS AI services...", "system")
        self.control_status_label.setText("Stopping services...")

        def run_stop():
            try:
                # Use the unified stop script
                self.log_message(f"📜 Executing: {STOP_SCRIPT}", "system")

                result = subprocess.run([
                    "bash", str(STOP_SCRIPT)
                ], capture_output=True, text=True, cwd=REPO_ROOT, timeout=30)

                if result.returncode == 0:
                    self.log_message("✅ All services stopped successfully", "system")
                    self.log_message(result.stdout, "system")
                    self.control_status_label.setText("✅ All services stopped")
                else:
                    self.log_message(f"⚠️ Stop command completed with warnings: {result.stderr}", "system")
                    self.control_status_label.setText("⚠️ Stopped with warnings")

            except subprocess.TimeoutExpired:
                self.log_message("⚠️ Stop command timed out", "system")
                self.control_status_label.setText("⚠️ Timeout")
            except Exception as e:
                self.log_message(f"❌ Error stopping services: {e}", "system")
                self.control_status_label.setText("❌ Error occurred")

        threading.Thread(target=run_stop, daemon=True).start()

    def check_system_status(self):
        """
        Check status of all services using shell script.
        Uses scripts/check_system_status.sh directly (no API calls).
        """
        self.log_message("📊 Checking system status...", "system")

        def run_status_check():
            try:
                result = subprocess.run([
                    "bash", str(STATUS_SCRIPT)
                ], capture_output=True, text=True, cwd=REPO_ROOT, timeout=10)

                if result.returncode == 0:
                    self.log_message("✅ Status check complete", "system")
                    self.log_message(result.stdout, "system")
                else:
                    self.log_message(f"⚠️ Status check had warnings: {result.stderr}", "system")

            except subprocess.TimeoutExpired:
                self.log_message("⚠️ Status check timed out", "system")
            except Exception as e:
                self.log_message(f"❌ Status check error: {e}", "system")

        threading.Thread(target=run_status_check, daemon=True).start()

    def open_frontend(self):
        """Open the frontend in default browser"""
        import webbrowser
        try:
            webbrowser.open('http://localhost:3005')
            self.log_message("🌐 Opened frontend in browser (port 3005)", "system")
        except Exception as e:
            self.log_message(f"❌ Failed to open frontend: {e}", "system")

    def open_watchdog_log(self):
        """Open watchdog log file in default text editor"""
        try:
            log_file = LOG_DIR / "watchdog.out"
            if log_file.exists():
                subprocess.run(["open", str(log_file)])
                self.log_message(f"� Opened watchdog log: {log_file}", "system")
            else:
                self.log_message(f"⚠️ Watchdog log not found: {log_file}", "system")
        except Exception as e:
            self.log_message(f"❌ Failed to open watchdog log: {e}", "system")

    def open_api_log(self):
        """Open API log file in default text editor"""
        try:
            log_file = LOG_DIR / "api.out"
            if log_file.exists():
                subprocess.run(["open", str(log_file)])
                self.log_message(f"🔧 Opened API log: {log_file}", "system")
            else:
                self.log_message(f"⚠️ API log not found: {log_file}", "system")
        except Exception as e:
            self.log_message(f"❌ Failed to open API log: {e}", "system")

    def closeEvent(self, event):
        """Handle window close event with proper cleanup"""
        self.log_message("🛑 Shutting down OSRS AI Control Center...", "system")

        # Stop status monitoring thread
        if self.status_thread:
            self.status_thread.stop()

        # Clean up all processes
        self.process_manager.cleanup_all_processes()

        self.log_message("✅ Cleanup completed", "system")
        event.accept()

def main():
    """Main application entry point"""
    # Create QApplication
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("OSRS AI Control Center")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("OSRS AI Systems")

    # Create and show main window
    window = OSRSAdminMainWindow()
    window.show()

    # Log startup
    window.log_message("🚀 OSRS AI Control Center started", "system")
    window.log_message("✅ PyQt6 GUI initialized successfully", "system")
    window.log_message("🔧 Process management system active", "system")

    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
