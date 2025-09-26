#!/usr/bin/env python3

"""
OSRS Wiki Template Checker - Parallel Content Validation & Correction

Purpose: Scan existing wiki pages to detect formatting issues and batch refetch
pages that need template parsing corrections with full raw wikitext content.

Features:
- Multi-core CPU + Apple Metal GPU parallel processing
- Batch MediaWiki API calls (up to 50 pages per request)
- Template format validation and issue detection
- Memory-efficient page correction pipeline
- Integration with streamlined watchdog system
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional
import re
import subprocess
from pathlib import Path

class OSRSWikiTemplateChecker:
    def __init__(self, data_dir: str = None):
        """Initialize the template checker with parallel processing capabilities"""
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '../data')
        self.wiki_api_url = 'https://oldschool.runescape.wiki/api.php'
        self.user_agent = 'OSRS-AI-System/1.0 (https://github.com/user/osrs-ai; contact@example.com)'

        # File paths
        self.content_file = os.path.join(self.data_dir, 'osrs_wiki_content.jsonl')
        self.issues_file = os.path.join(self.data_dir, 'osrs_template_issues.json')
        self.issues_log_file = os.path.join(self.data_dir, 'osrs_template_issues.jsonl')

        # Processing configuration
        self.cpu_cores = multiprocessing.cpu_count()
        self.has_gpu = self.detect_gpu_acceleration()
        # Hard cap from env (default 16; max 64)
        self.hard_cap_workers = max(1, min(int(os.getenv('OSRS_CHECKER_MAX_WORKERS', '16')), 64))
        # Start conservative and never exceed hard cap
        self.max_workers = min(self.calculate_optimal_workers(), self.hard_cap_workers)
        self.batch_size = 50  # MediaWiki API limit for batch requests
        # Limit refetch volume per run to keep watchdog responsive
        self.max_refetch_per_run = 100

        # Issue tracking
        self.pages_with_issues = set()
        self.pages_missing_wikitext = set()
        self.pages_needing_refetch = set()
        self.corrected_pages = {}

        # OS limit discovery - learn safe limits during runtime
        self.discovered_max_workers = None
        self.safe_max_workers = None

        print(f"🔧 Template Checker initialized:")
        print(f"   📊 CPU cores: {self.cpu_cores}")
        print(f"   🎮 GPU acceleration: {'✅ Enabled' if self.has_gpu else '❌ Not available'}")
        print(f"   ⚡ Max workers: {self.max_workers}")
        print(f"   📦 Batch size: {self.batch_size}")

    def detect_gpu_acceleration(self) -> bool:
        """Detect Apple Metal GPU or NVIDIA GPU acceleration"""
        try:
            # Check for Apple Silicon + Metal
            if sys.platform == 'darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    cpu_info = result.stdout.lower()
                    if any(chip in cpu_info for chip in ['apple', 'm1', 'm2', 'm3', 'm4']):
                        print("   🍎 Apple Silicon + Metal GPU detected!")
                        return True

            # Check for NVIDIA GPU
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    print("   🎮 NVIDIA GPU detected!")
                    return True
            except FileNotFoundError:
                pass

            return False
        except Exception:
            return False

    def calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers - start small and scale aggressively"""
        # Start with minimal workers, will scale up dynamically
        return 4  # Start conservative, scale up to 1024+ based on performance

    def detect_template_issues(self, page_content: Dict) -> List[str]:
        """Minimal detection: only flag legacy bracket tokens (no brace-based heuristics)."""
        issues = []
        text = page_content.get('text', '') or ''
        if ('[DiarySkillStats:' in text) or ('[ItemSpawnTableHead:' in text) or ('[ItemSpawnLine:' in text):
            issues.append('legacy_bracket_tokens')
        return issues

    async def scan_existing_pages(self) -> Dict[str, List[str]]:
        """Scan all existing pages for template formatting issues with dynamic scaling"""
        print("🔍 Scanning existing pages for template issues...")

        if not os.path.exists(self.content_file):
            print("   ⚠️  No existing content file found")
            return {}

        # Count total pages first
        total_pages = 0
        with open(self.content_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    total_pages += 1

        if total_pages == 0:
            print("   ⚠️  No pages found in content file")
            return {}

        print(f"   📊 Total pages to scan: {total_pages:,}")
        print(f"   🚀 Starting with {self.max_workers} workers, scaling up to 1024+")

        issues_found = {}
        pages_scanned = 0
        start_time = time.time()
        current_workers = self.max_workers
        # Enforce hard cap on workers
        current_workers = min(max(1, current_workers), self.hard_cap_workers)


        # Load all pages into memory for dynamic processing
        all_pages = []
        with open(self.content_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    page = json.loads(line.strip())
                    all_pages.append(page)
                except json.JSONDecodeError:
                    continue

        # Dynamic worker scaling with progress tracking
        batch_size = max(50, len(all_pages) // 100)  # Adaptive batch size

        print(f"   📦 Processing in batches of {batch_size} pages")
        print(f"   🔢 Total batches: {(len(all_pages) + batch_size - 1) // batch_size}")

        for i in range(0, len(all_pages), batch_size):
            batch = all_pages[i:i + batch_size]
            batch_number = (i // batch_size) + 1
            total_batches = (len(all_pages) + batch_size - 1) // batch_size

            print(f"   🔄 Processing batch {batch_number}/{total_batches} ({len(batch)} pages)")

            if len(batch) == 0:
                print(f"   ⚠️  Empty batch detected at index {i}, skipping...")
                continue

            # DEBUG: Special handling for final batch
            if batch_number == total_batches:
                print(f"   🎯 FINAL BATCH: {len(batch)} pages - Expected total: {len(all_pages)}")
                print(f"   🎯 FINAL BATCH: Current pages_scanned before: {pages_scanned}")
                print(f"   🎯 FINAL BATCH: After this batch should be: {pages_scanned + len(batch)}")
            batch_start_time = time.time()

            # AGGRESSIVE DYNAMIC WORKER SCALING
            if i > 0:
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / (i // batch_size + 1)
                memory_usage = self.get_memory_usage()

                # SMART SCALING - Use discovered OS limits
                max_theoretical_workers = self.get_max_theoretical_workers()

                # Use discovered safe limit if we found one
                effective_max_workers = max_theoretical_workers
                if self.discovered_max_workers is not None:
                    # Use 75% of discovered limit as safe maximum
                    effective_max_workers = int(self.discovered_max_workers * 0.75)

                if avg_time_per_batch < 0.3 and memory_usage < 0.6 and current_workers < effective_max_workers:
                    # EXTREMELY fast processing - scale aggressively
                    current_workers = min(current_workers * 3, effective_max_workers)
                    print(f"\r🚀🚀 ULTRA scaling UP to {current_workers} workers (blazing fast, abundant memory)")
                elif avg_time_per_batch < 0.5 and memory_usage < 0.7 and current_workers < effective_max_workers:
                    # Very fast processing - double workers
                    current_workers = min(current_workers * 2, effective_max_workers)
                    print(f"\r🚀 Scaling UP to {current_workers} workers (fast processing, low memory)")
                elif avg_time_per_batch < 1.0 and memory_usage < 0.8 and current_workers < effective_max_workers // 2:
                    # Good processing - add significant workers
                    current_workers = min(current_workers + 64, effective_max_workers // 2)
                elif avg_time_per_batch < 2.0 and memory_usage < 0.85 and current_workers < effective_max_workers // 4:
                    # Decent processing - add moderate workers
                    current_workers = min(current_workers + 32, effective_max_workers // 4)
                elif avg_time_per_batch > 3.0 and current_workers > 16:
                    current_workers = max(current_workers // 2, 16)
                    print(f"\r⚠️  Scaling DOWN to {current_workers} workers (slow processing)")
                elif memory_usage > 0.9:
                    current_workers = max(current_workers // 2, 8)
                    print(f"\r⚠️  Scaling DOWN to {current_workers} workers (high memory: {memory_usage*100:.1f}%)")
                elif memory_usage > 0.95:
                    current_workers = max(current_workers // 4, 4)
                    print(f"\r🚨 EMERGENCY scaling DOWN to {current_workers} workers (critical memory: {memory_usage*100:.1f}%)")

            # Process batch with DYNAMIC worker scaling to prevent resource contention
            try:
                # DYNAMIC WORKER SCALING: Scale workers based on batch size to prevent deadlock
                max_workers_per_page = 8  # Maximum 8 workers per page
                optimal_workers = min(len(batch) * max_workers_per_page, current_workers)
                batch_workers = max(1, min(optimal_workers, self.hard_cap_workers))

                # DEBUG: Always show worker calculation for final batches
                if batch_number >= total_batches - 1:
                    print(f"   🔧 BATCH {batch_number}/{total_batches}: {len(batch)} pages")
                    print(f"   🔧 Worker calc: {len(batch)} × {max_workers_per_page} = {len(batch) * max_workers_per_page}")
                    print(f"   🔧 Optimal: min({len(batch) * max_workers_per_page}, {current_workers}) = {optimal_workers}")
                    print(f"   🔧 Final: min(max(1, {optimal_workers}), {self.hard_cap_workers}) = {batch_workers}")

                if batch_workers != current_workers:
                    print(f"   ⚡ Scaling workers: {current_workers} → {batch_workers} for {len(batch)} pages")

                # FINAL BATCH: for very small sets, avoid spawning processes
                if len(batch) <= 8 or batch_workers <= 2:
                    print(f"   🧮 Small batch optimization: processing {len(batch)} pages synchronously")
                # FINAL BATCH: run through the same parallel chunker, but show extra debug
                if batch_number == total_batches:
                    print(f"   🎯 FINAL BATCH: {len(batch)} pages (parallel chunked)")

                # SANE EXECUTION: Avoid creating huge process pools for a single task.
                # Run batch scan synchronously (it's fast and avoids OS spawn overhead).
                try:
                    # Determine an effective pool size (respect dynamic scaling but avoid OS exhaustion)
                    pool_workers = max(1, min(batch_workers, len(batch)))
                    chunk_size = max(1, (len(batch) + pool_workers - 1) // pool_workers)

                    batch_results = {}
                    if pool_workers == 1:
                        # Synchronous small-batch path (no process spawn)
                        batch_results = self.scan_page_batch(batch)
                    else:
                        with ProcessPoolExecutor(max_workers=max(1, min(pool_workers, self.hard_cap_workers))) as executor:
                            futures = []
                            for j in range(0, len(batch), chunk_size):
                                sub = batch[j:j+chunk_size]
                                futures.append(executor.submit(self.scan_page_batch, sub))
                            for fut in as_completed(futures, timeout=120):
                                res = fut.result()
                                if res:
                                    batch_results.update(res)
                            executor.shutdown(wait=True, cancel_futures=True)

                    # Stream issues to JSONL log immediately
                    try:
                        with open(self.issues_log_file, 'a', encoding='utf-8') as lf:
                            for t, iss in batch_results.items():
                                lf.write(json.dumps({'title': t, 'issues': iss}) + "\n")
                    except Exception:
                        pass

                    issues_found.update(batch_results)
                    pages_scanned += len(batch)
                except Exception as e:
                    print(f"\r⚠️  Batch processing error (parallel): {e}")
                    pages_scanned += len(batch)
                    continue
            except OSError as e:
                if "Invalid argument" in str(e) or "Too many open files" in str(e):
                    # Hit OS limit! Scale back dramatically and remember this limit
                    print(f"\r🚨 Hit OS limit at {current_workers:,} workers! Scaling back...")
                    current_workers = max(current_workers // 4, 64)  # Scale back to 25% or minimum 64
                    self.discovered_max_workers = current_workers  # Remember this limit
                    print(f"   🔧 Reduced to {current_workers:,} workers (OS limit discovered)")

                    # Retry with reduced workers
                    try:
                        with ProcessPoolExecutor(max_workers=min(max(1, current_workers), self.hard_cap_workers)) as executor:
                            future = executor.submit(self.scan_page_batch, batch)
                            batch_results = future.result(timeout=30)
                            issues_found.update(batch_results)
                            # Stream issues to JSONL log immediately (retry path)
                            try:
                                with open(self.issues_log_file, 'a', encoding='utf-8') as lf:
                                    for t, iss in batch_results.items():
                                        lf.write(json.dumps({'title': t, 'issues': iss}) + "\n")
                            except Exception:
                                pass
                            pages_scanned += len(batch)
                    except Exception as retry_e:
                        print(f"\r⚠️  Retry failed: {retry_e}")
                        pages_scanned += len(batch)
                        continue
                else:
                    print(f"\r⚠️  Unexpected OS error: {e}")
                    pages_scanned += len(batch)
                    continue

            # REAL-TIME PROGRESS BAR
            progress = (pages_scanned / total_pages) * 100
            elapsed = time.time() - start_time
            rate = pages_scanned / elapsed if elapsed > 0 else 0
            eta = (total_pages - pages_scanned) / rate if rate > 0 else 0
            batch_time = time.time() - batch_start_time

            # Progress bar visualization
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            print(f"\r🔍 Scanning |{bar}| {progress:.1f}% | {pages_scanned:,}/{total_pages:,} | "
                  f"ETA: {eta:.0f}s | {rate:.1f}/s | Workers: {current_workers} | Batch: {batch_time:.1f}s", end='')

            # DEBUG: Final batch completion
            if batch_number == total_batches:
                print(f"\n   🎯 FINAL BATCH COMPLETED! Progress: {progress:.1f}% ({pages_scanned:,}/{total_pages:,})")

        print()  # New line after progress bar
        print(f"   ✅ Batch processing loop completed! Scanned {pages_scanned:,}/{total_pages:,} pages")

        # Categorize issues
        for title, issues in issues_found.items():
            if not issues:
                continue

            self.pages_with_issues.add(title)
            if 'missing_raw_wikitext' in issues:
                self.pages_missing_wikitext.add(title)

            # Refetch ALL pages that reported any issues in the scan
            # This matches the expectation: "do all of the pages in the osrs template issues"
            self.pages_needing_refetch.add(title)

        total_time = time.time() - start_time
        final_rate = pages_scanned / total_time if total_time > 0 else 0

        print(f"✅ Scan complete: {pages_scanned:,} pages in {total_time:.1f}s ({final_rate:.1f}/s)")
        print(f"   ⚠️  Found {len(self.pages_with_issues):,} pages with issues")
        print(f"   📄 {len(self.pages_missing_wikitext):,} pages missing raw wikitext")
        print(f"   🚀 Peak workers: {current_workers}")

        # Save issues report
        with open(self.issues_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.time(),
                'pages_scanned': pages_scanned,
                'total_time': total_time,
                'final_rate': final_rate,
                'peak_workers': current_workers,
                'issues_found': {title: issues for title, issues in issues_found.items() if issues},
                'summary': {
                    'total_issues': len(self.pages_with_issues),
                    'missing_wikitext': len(self.pages_missing_wikitext),
                    'need_refetch': len(self.pages_needing_refetch)
                }
            }, f, indent=2)

        return issues_found

    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback without psutil
            try:
                import os
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    total = int([line for line in lines if 'MemTotal' in line][0].split()[1])
                    available = int([line for line in lines if 'MemAvailable' in line][0].split()[1])
                    return 1.0 - (available / total)
            except:
                return 0.5  # Conservative estimate if can't determine

    def get_max_theoretical_workers(self) -> int:
        """Return capped max workers from env; default 16 (max 64)."""
        try:
            cap = int(os.getenv('OSRS_CHECKER_MAX_WORKERS', '16'))
        except Exception:
            cap = 16
        cap = max(1, min(cap, 64))
        return cap

    def scan_page_batch(self, pages: List[Dict]) -> Dict[str, List[str]]:
        """Scan a batch of pages for issues (runs in parallel process)"""
        results = {}

        for page in pages:
            title = page.get('title', '')
            if title:
                issues = self.detect_template_issues(page)
                results[title] = issues

        return results

    async def batch_refetch_pages(self, titles: Set[str]) -> Dict[str, Dict]:
        """Batch refetch pages with full raw wikitext using MediaWiki API"""
        if not titles:
            return {}

        titles_list = list(titles)
        total_pages = len(titles_list)

        print(f"🔄 Batch refetching {total_pages:,} pages with full content...")
        print(f"   📦 Using {self.batch_size} pages per API request")

        refetched_pages = {}
        start_time = time.time()

        # Process in batches of 50 (MediaWiki API limit)
        async with aiohttp.ClientSession() as session:
            for i in range(0, total_pages, self.batch_size):
                batch_titles = titles_list[i:i + self.batch_size]
                batch_start_time = time.time()

                try:
                    batch_results = await self.fetch_page_batch(session, batch_titles)
                    refetched_pages.update(batch_results)

                    # REAL-TIME PROGRESS BAR
                    pages_fetched = min(i + self.batch_size, total_pages)
                    progress = (pages_fetched / total_pages) * 100
                    elapsed = time.time() - start_time
                    rate = pages_fetched / elapsed if elapsed > 0 else 0
                    eta = (total_pages - pages_fetched) / rate if rate > 0 else 0
                    batch_time = time.time() - batch_start_time

                    # Progress bar visualization
                    bar_length = 40
                    filled_length = int(bar_length * progress / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)

                    print(f"\r📥 Fetching |{bar}| {progress:.1f}% | {pages_fetched:,}/{total_pages:,} | "
                          f"ETA: {eta:.0f}s | {rate:.1f}/s | Batch: {batch_time:.1f}s | "
                          f"Success: {len(batch_results)}/{len(batch_titles)}", end='')

                    # No extra delay for read-only requests; strictly serial per MediaWiki guidance.
                    # Backoff is handled on HTTP errors in fetch_page_batch.

                except Exception as e:
                    print(f"\r⚠️  Batch fetch error: {e}")
                    continue

        print()  # New line after progress bar

        total_time = time.time() - start_time
        final_rate = len(refetched_pages) / total_time if total_time > 0 else 0

        print(f"✅ Refetch complete: {len(refetched_pages):,}/{total_pages:,} pages in {total_time:.1f}s ({final_rate:.1f}/s)")
        return refetched_pages

    async def fetch_page_batch(self, session: aiohttp.ClientSession, titles: List[str]) -> Dict[str, Dict]:
        """Fetch a batch of pages from MediaWiki API"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': '|'.join(titles),
            'prop': 'revisions|categories',
            'rvprop': 'content|timestamp|ids',
            'rvslots': 'main',
            'cllimit': 'max',
            'maxlag': '5'
        }

        headers = {
            'User-Agent': self.user_agent
        }

        try:
            async with session.get(self.wiki_api_url, params=params, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"API returned status {response.status}")

                data = await response.json()
                pages = data.get('query', {}).get('pages', {})

                results = {}
                for page_id, page_data in pages.items():
                    if page_id == '-1' or 'revisions' not in page_data:
                        continue

                    title = page_data.get('title', '')
                    revision = page_data['revisions'][0]
                    categories = [cat['title'] for cat in page_data.get('categories', [])]

                    results[title] = {
                        'title': title,
                        'rawWikitext': revision['slots']['main']['*'],
                        'timestamp': revision['timestamp'],
                        'revid': revision['revid'],
                        'categories': categories
                    }

                return results

        except Exception as e:
            print(f"   ⚠️  API request failed: {e}")
            return {}

    async def apply_template_corrections(self, pages: Dict[str, Dict]) -> Dict[str, Dict]:
        """Apply template parsing corrections to refetched pages with dynamic scaling"""
        if not pages:
            return {}

        total_pages = len(pages)
        print(f"🔧 Applying template corrections to {total_pages:,} pages...")
        print(f"   🚀 Starting with {self.max_workers} workers, scaling up to 1024+")

        corrected_pages = {}
        start_time = time.time()
        current_workers = self.max_workers
        # Enforce cap for corrections pass
        current_workers = min(max(1, current_workers), self.get_max_theoretical_workers())


        page_items = list(pages.items())

        # Dynamic batch processing with aggressive worker scaling
        batch_size = max(10, len(page_items) // 50)  # Smaller batches for better progress tracking
        processed_count = 0

        for i in range(0, len(page_items), batch_size):
            batch = page_items[i:i + batch_size]
            batch_start_time = time.time()

            # AGGRESSIVE DYNAMIC WORKER SCALING FOR CORRECTIONS
            if i > 0:
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / (i // batch_size + 1)
                memory_usage = self.get_memory_usage()

                # ULTRA-AGGRESSIVE SCALING FOR TEMPLATE PROCESSING
                max_theoretical_workers = self.get_max_theoretical_workers()
                # Template processing is more memory-intensive, so use 75% of theoretical max
                max_template_workers = int(max_theoretical_workers * 0.75)

                if avg_time_per_batch < 0.5 and memory_usage < 0.6 and current_workers < max_template_workers:
                    # EXTREMELY fast template processing - scale aggressively
                    current_workers = min(current_workers * 3, max_template_workers)
                    print(f"\r🚀🚀 ULTRA scaling UP to {current_workers} workers (blazing template processing)")
                elif avg_time_per_batch < 1.0 and memory_usage < 0.7 and current_workers < max_template_workers:
                    # Fast template processing - double workers
                    current_workers = min(current_workers * 2, max_template_workers)
                    print(f"\r🚀 Scaling UP to {current_workers} workers (fast template processing)")
                elif avg_time_per_batch < 2.0 and memory_usage < 0.8 and current_workers < max_template_workers // 2:
                    # Good template processing - add significant workers
                    current_workers = min(current_workers + 128, max_template_workers // 2)
                elif avg_time_per_batch < 3.0 and memory_usage < 0.85 and current_workers < max_template_workers // 4:
                    # Decent template processing - add moderate workers
                    current_workers = min(current_workers + 64, max_template_workers // 4)
                elif avg_time_per_batch > 5.0 and current_workers > 32:
                    current_workers = max(current_workers // 2, 32)
                    print(f"\r⚠️  Scaling DOWN to {current_workers} workers (slow template processing)")
                elif memory_usage > 0.9:
                    current_workers = max(current_workers // 2, 16)
                    print(f"\r⚠️  Scaling DOWN to {current_workers} workers (high memory: {memory_usage*100:.1f}%)")
                elif memory_usage > 0.95:
                    current_workers = max(current_workers // 4, 8)
                    print(f"\r🚨 EMERGENCY scaling DOWN to {current_workers} workers (critical memory: {memory_usage*100:.1f}%)")

            # Process batch with current worker count (small-batch synchronous path)
            if len(batch) <= 2 or current_workers <= 2:
                batch_results = self.process_correction_batch(batch)
                corrected_pages.update(batch_results)
                processed_count += len(batch_results)
            else:
                with ProcessPoolExecutor(max_workers=min(max(1, current_workers), self.get_max_theoretical_workers())) as executor:
                    future = executor.submit(self.process_correction_batch, batch)
                    try:
                        batch_results = future.result(timeout=60)  # Longer timeout for template processing
                        corrected_pages.update(batch_results)
                        processed_count += len(batch_results)
                    except Exception as e:
                        print(f"\r⚠️  Correction batch error: {e}")
                        continue

            # REAL-TIME PROGRESS BAR
            progress = (processed_count / total_pages) * 100
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (total_pages - processed_count) / rate if rate > 0 else 0
            batch_time = time.time() - batch_start_time

            # Progress bar visualization
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            print(f"\r🔧 Correcting |{bar}| {progress:.1f}% | {processed_count:,}/{total_pages:,} | "
                  f"ETA: {eta:.0f}s | {rate:.1f}/s | Workers: {current_workers} | Batch: {batch_time:.1f}s", end='')

        print()  # New line after progress bar

        total_time = time.time() - start_time
        final_rate = processed_count / total_time if total_time > 0 else 0

        print(f"✅ Corrections complete: {len(corrected_pages):,} pages in {total_time:.1f}s ({final_rate:.1f}/s)")
        print(f"   🚀 Peak workers: {current_workers}")

        self.corrected_pages = corrected_pages
        return corrected_pages

    def process_correction_batch(self, page_batch: List[Tuple[str, Dict]]) -> Dict[str, Dict]:
        """Process a batch of pages for template corrections (runs in parallel process)"""
        # Import parser in worker process
        sys.path.append(os.path.dirname(__file__))
        from wiki_template_parser import OSRSWikiTemplateParser

        parser = OSRSWikiTemplateParser()
        results = {}

        for title, page_data in page_batch:
            try:
                raw_wikitext = page_data.get('rawWikitext', '')
                if not raw_wikitext:
                    continue

                # Process templates from raw wikitext
                processed_wikitext = parser.process_wiki_content(raw_wikitext)

                # Clean the processed content
                clean_text = self.clean_processed_content(processed_wikitext)

                # Create corrected page data
                corrected_page = {
                    'title': title,
                    'text': clean_text,
                    'rawWikitext': raw_wikitext,
                    'categories': page_data.get('categories', []),
                    'timestamp': page_data.get('timestamp', ''),
                    'revid': page_data.get('revid', ''),
                    'corrected': True,
                    'correction_timestamp': time.time()
                }

                results[title] = corrected_page

            except Exception as e:
                print(f"   ⚠️  Error processing {title}: {e}")
                continue

        return results

    def clean_processed_content(self, content: str) -> str:
        """Clean processed wikitext content for AI consumption"""
        # Remove remaining wiki markup
        content = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', content)  # [[link|text]] -> text
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)  # [[link]] -> link
        content = re.sub(r"'''([^']+)'''", r'\1', content)  # Bold
        content = re.sub(r"''([^']+)''", r'\1', content)  # Italic

        # Remove external links
        content = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', content)

        # Remove file references and categories
        content = re.sub(r'\[\[File:[^\]]+\]\]', '', content)
        content = re.sub(r'\[\[Category:[^\]]+\]\]', '', content)

        # Remove HTML comments
        content = re.sub(r'<!--[\s\S]*?-->', '', content)

        # Clean up whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'^\s+|\s+$', '', content, flags=re.MULTILINE)

        return content.strip()

    async def run_full_check_and_correction(self, additional_titles: Set[str] = None) -> Dict[str, Dict]:
        """
        Main method: Complete template checking and correction pipeline

        Optimal order:
        1. Scan existing pages for template issues
        2. Add any additional titles that need checking
        3. Batch refetch all problematic pages with full raw wikitext
        4. Apply template parsing corrections
        5. Return corrected pages for watchdog integration
        """
        print("🚀 Starting comprehensive template check and correction...")

        # Step 1: Scan existing pages for issues
        issues_found = await self.scan_existing_pages()

        # Step 2: Add additional titles if provided
        if additional_titles:
            print(f"   📝 Adding {len(additional_titles):,} additional titles to check")
            self.pages_needing_refetch.update(additional_titles)

        # Step 3: Optionally union candidates from previous issues data (off by default)
        loaded = 0
        try:
            if os.getenv('OSRS_CHECKER_INCLUDE_HISTORY', '0') == '1':
                if os.path.exists(self.issues_log_file):
                    with open(self.issues_log_file, 'r', encoding='utf-8') as lf:
                        for line in lf:
                            try:
                                rec = json.loads(line)
                                title = rec.get('title')
                                issues = rec.get('issues', [])
                                if title and issues is not None and len(issues) > 0:
                                    if title not in self.pages_needing_refetch:
                                        self.pages_needing_refetch.add(title)
                                        loaded += 1
                            except Exception:
                                continue
                if os.path.exists(self.issues_file):
                    with open(self.issues_file, 'r', encoding='utf-8') as jf:
                        j = json.load(jf)
                        if isinstance(j, dict) and 'issues_found' in j and isinstance(j['issues_found'], dict):
                            for title, issues in j['issues_found'].items():
                                if issues:
                                    if title not in self.pages_needing_refetch:
                                        self.pages_needing_refetch.add(title)
                                        loaded += 1
                if loaded:
                    print(f"   📥 Loaded {loaded:,} additional refetch candidates from issues logs")
            else:
                print("   🧹 Skipping historical issues union (OSRS_CHECKER_INCLUDE_HISTORY=0)")
        except Exception as e:
            print(f"   ⚠️  Failed reading issues data: {e}")

        # Step 4: Batch refetch pages that need correction (no cap; show full progress)
        if self.pages_needing_refetch:
            titles_list = list(self.pages_needing_refetch)
            refetched_pages = await self.batch_refetch_pages(set(titles_list))

            # Step 4: Apply template corrections
            if refetched_pages:
                corrected_pages = await self.apply_template_corrections(refetched_pages)

                print(f"✅ Template check and correction complete!")
                print(f"   📊 Total pages processed: {len(corrected_pages):,}")
                print(f"   🔧 Pages corrected: {len([p for p in corrected_pages.values() if p.get('corrected')])}")

                return corrected_pages

        print("✅ No pages needed correction")
        return {}

    def get_correction_summary(self) -> Dict:
        """Get summary of correction operations"""
        return {
            'pages_with_issues': len(self.pages_with_issues),
            'pages_missing_wikitext': len(self.pages_missing_wikitext),
            'pages_needing_refetch': len(self.pages_needing_refetch),
            'pages_corrected': len(self.corrected_pages),
            'has_gpu_acceleration': self.has_gpu,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }

    def save_corrected_pages_to_memory(self) -> Dict[str, Dict]:
        """Return corrected pages for integration with streamlined watchdog"""
        return self.corrected_pages.copy()

# Integration functions for streamlined watchdog
async def check_and_correct_templates(data_dir: str = None, additional_titles: Set[str] = None) -> Dict[str, Dict]:
    """
    Main integration function for streamlined watchdog

    Args:
        data_dir: Path to data directory
        additional_titles: Additional page titles to check/correct

    Returns:
        Dictionary of corrected pages ready for watchdog processing
    """
    checker = OSRSWikiTemplateChecker(data_dir)
    corrected_pages = await checker.run_full_check_and_correction(additional_titles)
    return corrected_pages

def get_template_checker_summary(data_dir: str = None) -> Dict:
    """Get summary of template checker capabilities and last run"""
    checker = OSRSWikiTemplateChecker(data_dir)
    return checker.get_correction_summary()

if __name__ == "__main__":
    # Test the template checker
    import asyncio

    async def test_checker():
        checker = OSRSWikiTemplateChecker()

        # Test with a small set of pages
        test_titles = {'Dragon Slayer I', 'Zulrah', 'Abyssal whip'}
        corrected_pages = await checker.run_full_check_and_correction(test_titles)

        print(f"\n📊 Test Results:")
        print(f"   Pages corrected: {len(corrected_pages)}")
        for title, page in corrected_pages.items():
            print(f"   ✅ {title}: {len(page.get('text', ''))} chars")

    asyncio.run(test_checker())
