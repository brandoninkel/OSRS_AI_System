#!/usr/bin/env python3
"""
KG Auto-Updater Service

Monitors wiki content changes and automatically triggers incremental KG updates.
Runs in background and integrates with the streamlined watchdog system.

Features:
- File watching for wiki content changes
- Incremental KG updates (fast, preserves existing embeddings)
- Background processing without blocking
- Status reporting for admin GUI
- Signal handling for graceful shutdown

Usage:
    python3 scripts/kg_auto_updater.py --follow
"""

import os
import sys
import time
import json
import logging
import signal
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KGAutoUpdater:
    def __init__(self, progress_mode=False):
        self.repo_root = Path(__file__).parent.parent
        self.data_dir = self.repo_root / "data"
        self.scripts_dir = self.repo_root / "scripts"
        self.logs_dir = self.repo_root / "logs" / "kg"

        # Files to monitor
        self.wiki_content_file = self.data_dir / "osrs_wiki_content.jsonl"
        self.wikitext_file = self.data_dir / "osrs_wikitext_content.jsonl"
        self.kg_triples_file = self.data_dir / "osrs_kg_triples.csv"

        # Status tracking
        self.status_file = self.data_dir / "kg_updater_status.json"
        self.pid_file = self.logs_dir / "kg_updater.pid"

        # State
        self.running = False
        self.last_wiki_hash = None
        self.last_wikitext_hash = None
        self.update_in_progress = False
        self.progress_mode = progress_mode
        
        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Write PID file
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.info(f"KG Auto-Updater PID: {os.getpid()}")
        except Exception as e:
            logger.warning(f"Could not write PID file: {e}")
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._trigger_update_signal)
        
        # Load initial state
        self._load_status()
        self._update_file_hashes()

    def report_progress(self, progress_percent, status="processing"):
        """Report progress for orchestration monitoring"""
        if self.progress_mode:
            print(f"Progress: {progress_percent:.1f}%", flush=True)
            print(f"Status: {status}", flush=True)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _trigger_update_signal(self, signum, frame):
        """Handle SIGUSR1 signal to trigger immediate update"""
        logger.info("Received SIGUSR1 signal, triggering immediate KG update...")
        if not self.update_in_progress:
            self._trigger_kg_update()
        else:
            logger.info("Update already in progress, ignoring signal")

    def _load_status(self):
        """Load previous status from file"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                    self.last_wiki_hash = status.get('last_wiki_hash')
                    self.last_wikitext_hash = status.get('last_wikitext_hash')
                    logger.info("Loaded previous status")
        except Exception as e:
            logger.warning(f"Could not load status: {e}")

    def _save_status(self):
        """Save current status to file"""
        try:
            status = {
                'last_update': datetime.now().isoformat(),
                'last_wiki_hash': self.last_wiki_hash,
                'last_wikitext_hash': self.last_wikitext_hash,
                'update_in_progress': self.update_in_progress,
                'pid': os.getpid()
            }
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save status: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for change detection"""
        if not file_path.exists():
            return ""
        
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
            return ""

    def _update_file_hashes(self):
        """Update stored file hashes"""
        self.last_wiki_hash = self._get_file_hash(self.wiki_content_file)
        self.last_wikitext_hash = self._get_file_hash(self.wikitext_file)

    def _check_for_changes(self) -> bool:
        """Check if wiki content files have changed"""
        current_wiki_hash = self._get_file_hash(self.wiki_content_file)
        current_wikitext_hash = self._get_file_hash(self.wikitext_file)
        
        wiki_changed = current_wiki_hash != self.last_wiki_hash
        wikitext_changed = current_wikitext_hash != self.last_wikitext_hash
        
        if wiki_changed or wikitext_changed:
            logger.info(f"Changes detected - Wiki: {wiki_changed}, Wikitext: {wikitext_changed}")
            self.last_wiki_hash = current_wiki_hash
            self.last_wikitext_hash = current_wikitext_hash
            return True
        
        return False

    def _trigger_kg_update(self):
        """Trigger incremental KG update (runs synchronously when called from CLI)"""
        if self.update_in_progress:
            logger.info("KG update already in progress, skipping")
            return

        self.update_in_progress = True
        self._save_status()

        # Run update directly (not in thread) for better progress reporting
        try:
            logger.info("🔗 Starting incremental KG update...")
            self.report_progress(0, "starting KG update")

            # Step 1: Rebuild KG triples from wiki content changes
            logger.info("🔧 Building KG triples from updated wiki content...")
            self.report_progress(10, "building KG triples (this may take a while)")

            result = subprocess.run([
                "bash", str(self.scripts_dir / "knowledge-graph.command"),
                "--workers", "8", "--snapshot"
            ], cwd=self.repo_root)

            if result.returncode != 0:
                logger.error(f"KG triples build failed with return code: {result.returncode}")
                return

            logger.info("✅ KG triples built successfully")
            self.report_progress(40, "KG triples completed")

                # Step 2: Train PyKEEN model with new triples
                kg_model_dir = self.data_dir / "kg_model"
                logger.info("🧠 Training PyKEEN model with updated triples...")
                self.report_progress(50, "training PyKEEN model (1 epoch, ~1-2 min)")
                result = subprocess.run([
                    "bash", str(self.scripts_dir / "train-kg-embeddings.command"),
                    "--backend", "pykeen", "--no-eval", "--strict",
                    "--triples", str(self.kg_triples_file),
                    "--out", str(kg_model_dir),
                    "--model", "TransE", "--dimension", "100",
                    "--epochs", "1", "--num-workers", "0", "--batch-size", "512"
                ], cwd=self.repo_root)

                if result.returncode != 0:
                    logger.warning(f"PyKEEN training failed with return code: {result.returncode}")
                    logger.info("✅ Using existing KG model")
                else:
                    logger.info("✅ PyKEEN model trained successfully")

                self.report_progress(70, "PyKEEN training completed")
                
                # Step 3: Create high-performance unified mxbai embeddings
                if (kg_model_dir / "entity_to_id.json").exists():
                    logger.info("🚀 Creating high-performance unified mxbai KG embeddings...")
                    logger.info("⚡ Using async mode with max concurrency for full system utilization")
                    self.report_progress(80, "creating KG embeddings (~149k entities)")

                    # Use the same high-performance approach as the main embeddings system
                    cmd = [
                        "python3", "-u",  # Unbuffered output for real-time progress
                        str(self.scripts_dir / "create_osrs_embeddings.py"),
                        "--kg-entities-only",  # Special mode for KG entities
                        "--async",
                        "--max-concurrency", "16",
                        "--chunk-size", "100"
                    ]

                    # Add progress mode if enabled
                    if self.progress_mode:
                        cmd.append("--progress-mode")
                        # Stream output to parent process for progress monitoring
                        process = subprocess.Popen(cmd, cwd=self.repo_root,
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT,
                                                  text=True, bufsize=1)
                        # Forward output line by line
                        for line in process.stdout:
                            print(line, end='', flush=True)
                        result_code = process.wait()
                        result = type('obj', (object,), {'returncode': result_code})()
                    else:
                        result = subprocess.run(cmd, cwd=self.repo_root)

                    if result.returncode == 0:
                        logger.info("✅ High-performance unified KG embeddings created")

                        # Signal RAG service to reload embeddings
                        self._signal_rag_reload()
                        self.report_progress(95, "signaling RAG reload")
                    else:
                        logger.warning(f"KG embeddings creation failed with return code: {result.returncode}")
                        # Fallback to ultra-high-performance method
                        logger.info("🔄 Using ultra-high-performance direct embedding creation...")
                        result = subprocess.run([
                            "python3", str(self.scripts_dir / "kg" / "create_mxbai_kg_embeddings.py"),
                            "--workers", "128"
                        ], cwd=self.repo_root)

                        if result.returncode == 0:
                            logger.info("✅ Unified KG embeddings updated (fallback method)")
                            self._signal_rag_reload()
                            self.report_progress(95, "signaling RAG reload")
                        else:
                            logger.error(f"Both embedding methods failed")

            logger.info("🎉 KG auto-update complete!")
            self.report_progress(100, "completed")

        except Exception as e:
            logger.error(f"KG update failed: {e}")
        finally:
            self.update_in_progress = False
            self._save_status()

    def _signal_rag_reload(self):
        """Signal RAG service to reload embeddings"""
        try:
            rag_pid_file = self.data_dir / "rag_service.pid"
            if rag_pid_file.exists():
                with open(rag_pid_file, 'r') as f:
                    rag_pid = int(f.read().strip())
                os.kill(rag_pid, signal.SIGUSR1)
                logger.info(f"Signaled RAG service (PID {rag_pid}) to reload embeddings")
        except Exception as e:
            logger.warning(f"Could not signal RAG service: {e}")

    def run(self):
        """Main monitoring loop"""
        logger.info("🚀 KG Auto-Updater started")
        logger.info(f"Monitoring: {self.wiki_content_file}")
        logger.info(f"Monitoring: {self.wikitext_file}")
        
        self.running = True
        
        while self.running:
            try:
                if self._check_for_changes():
                    logger.info("📝 Wiki content changes detected, triggering KG update...")
                    self._trigger_kg_update()
                
                # Save status periodically
                self._save_status()
                
                # Check every 30 seconds
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("🛑 KG Auto-Updater stopped")
        
        # Cleanup
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except Exception:
            pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KG Auto-Updater Service")
    parser.add_argument("--follow", action="store_true", help="Run in follow mode (continuous monitoring)")
    parser.add_argument("--trigger-update", action="store_true", help="Trigger immediate update and exit")
    parser.add_argument("--progress-mode", action="store_true", help="Enable progress reporting for orchestration")
    args = parser.parse_args()

    updater = KGAutoUpdater(progress_mode=args.progress_mode)

    if args.trigger_update:
        logger.info("Triggering immediate KG update...")
        updater._trigger_kg_update()
        # Wait for update to complete
        while updater.update_in_progress:
            time.sleep(1)
        logger.info("Update complete, exiting")
        return

    if args.follow:
        updater.run()
    else:
        print("Use --follow to start continuous monitoring or --trigger-update for immediate update")
        return

if __name__ == "__main__":
    main()
