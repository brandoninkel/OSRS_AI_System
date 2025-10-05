#!/usr/bin/env python3
"""
Centralized API Queue Manager

Coordinates all API calls across the system to prevent rate limit violations.
Implements priority-based queueing with the streamlined watchdog getting highest priority.

Priority Levels:
- CRITICAL (0): Streamlined watchdog - halts/slows other API calls
- HIGH (1): User-facing queries (chat, economic analysis)
- MEDIUM (2): Attribution lookups
- LOW (3): Background tasks

Rate Limits:
- MediaWiki API: Serial requests only, 1 request per second max
- Prices API: 100ms between requests (10 req/sec max)
"""

import asyncio
import time
import logging
from enum import IntEnum
from typing import Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Priority levels for API requests"""
    CRITICAL = 0  # Streamlined watchdog
    HIGH = 1      # User-facing queries
    MEDIUM = 2    # Attribution lookups
    LOW = 3       # Background tasks


@dataclass
class APIRequest:
    """Represents a queued API request"""
    priority: Priority
    api_type: str  # 'mediawiki' or 'prices'
    func: Callable
    args: tuple
    kwargs: dict
    timestamp: float
    request_id: str
    
    def __lt__(self, other):
        """Compare by priority (lower number = higher priority)"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class APIQueueManager:
    """
    Centralized manager for all API calls.
    
    Ensures:
    - Rate limits are respected
    - Watchdog gets priority
    - No API clogging
    - Proper coordination between all callers
    """
    
    def __init__(self):
        # Priority queues for each API type
        self.mediawiki_queue = asyncio.PriorityQueue()
        self.prices_queue = asyncio.PriorityQueue()
        
        # Rate limiting state
        self.mediawiki_last_request = 0
        self.prices_last_request = 0
        
        # Watchdog state
        self.watchdog_active = False
        self.watchdog_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'mediawiki_requests': 0,
            'prices_requests': 0,
            'mediawiki_queued': 0,
            'prices_queued': 0,
            'watchdog_activations': 0
        }
        
        # Worker tasks
        self.workers_started = False
        self.mediawiki_worker = None
        self.prices_worker = None
        
        logger.info("🚦 API Queue Manager initialized")
    
    async def start_workers(self):
        """Start background workers to process queues"""
        if self.workers_started:
            return
        
        self.mediawiki_worker = asyncio.create_task(self._mediawiki_worker())
        self.prices_worker = asyncio.create_task(self._prices_worker())
        self.workers_started = True
        logger.info("✅ API queue workers started")
    
    async def _mediawiki_worker(self):
        """Worker that processes MediaWiki API requests"""
        while True:
            try:
                # Get next request from queue
                priority, request = await self.mediawiki_queue.get()
                
                # Apply rate limiting
                await self._apply_mediawiki_rate_limit(request.priority)
                
                # Execute request
                try:
                    result = await request.func(*request.args, **request.kwargs)
                    self.stats['mediawiki_requests'] += 1
                    logger.debug(f"✅ MediaWiki request completed: {request.request_id}")
                except Exception as e:
                    logger.error(f"❌ MediaWiki request failed: {request.request_id} - {e}")
                    result = None
                
                self.mediawiki_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ MediaWiki worker error: {e}")
                await asyncio.sleep(1)
    
    async def _prices_worker(self):
        """Worker that processes Prices API requests"""
        while True:
            try:
                # Get next request from queue
                priority, request = await self.prices_queue.get()
                
                # Apply rate limiting
                await self._apply_prices_rate_limit(request.priority)
                
                # Execute request
                try:
                    result = await request.func(*request.args, **request.kwargs)
                    self.stats['prices_requests'] += 1
                    logger.debug(f"✅ Prices request completed: {request.request_id}")
                except Exception as e:
                    logger.error(f"❌ Prices request failed: {request.request_id} - {e}")
                    result = None
                
                self.prices_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Prices worker error: {e}")
                await asyncio.sleep(1)
    
    async def _apply_mediawiki_rate_limit(self, priority: Priority):
        """
        Apply rate limiting for MediaWiki API.
        
        Rules:
        - Serial requests only (1 at a time)
        - 1 second between requests normally
        - If watchdog is active:
          * CRITICAL: No delay
          * HIGH: 2 second delay
          * MEDIUM/LOW: 5 second delay
        """
        now = time.time()
        elapsed = now - self.mediawiki_last_request
        
        if self.watchdog_active:
            # Watchdog is running - apply stricter limits for non-critical
            if priority == Priority.CRITICAL:
                min_delay = 1.0  # Normal rate for watchdog
            elif priority == Priority.HIGH:
                min_delay = 2.0  # Slow down user queries
            else:
                min_delay = 5.0  # Heavily throttle low priority
        else:
            # Normal operation
            min_delay = 1.0
        
        if elapsed < min_delay:
            wait_time = min_delay - elapsed
            logger.debug(f"⏳ Rate limiting: waiting {wait_time:.2f}s (priority={priority.name})")
            await asyncio.sleep(wait_time)
        
        self.mediawiki_last_request = time.time()
    
    async def _apply_prices_rate_limit(self, priority: Priority):
        """
        Apply rate limiting for Prices API.
        
        Rules:
        - 100ms between requests normally
        - If watchdog is active, slow down non-critical requests
        """
        now = time.time()
        elapsed = now - self.prices_last_request
        
        if self.watchdog_active and priority != Priority.CRITICAL:
            min_delay = 0.5  # Slow down when watchdog is active
        else:
            min_delay = 0.1  # Normal 100ms delay
        
        if elapsed < min_delay:
            wait_time = min_delay - elapsed
            await asyncio.sleep(wait_time)
        
        self.prices_last_request = time.time()
    
    async def queue_mediawiki_request(self, func: Callable, priority: Priority = Priority.HIGH,
                                     request_id: Optional[str] = None, *args, **kwargs) -> None:
        """
        Queue a MediaWiki API request.
        
        Args:
            func: Async function to call
            priority: Request priority level
            request_id: Optional identifier for logging
            *args, **kwargs: Arguments to pass to func
        """
        if not self.workers_started:
            await self.start_workers()
        
        request_id = request_id or f"mw_{int(time.time() * 1000)}"
        
        request = APIRequest(
            priority=priority,
            api_type='mediawiki',
            func=func,
            args=args,
            kwargs=kwargs,
            timestamp=time.time(),
            request_id=request_id
        )
        
        await self.mediawiki_queue.put((priority, request))
        self.stats['mediawiki_queued'] += 1
        logger.debug(f"📥 Queued MediaWiki request: {request_id} (priority={priority.name})")
    
    async def queue_prices_request(self, func: Callable, priority: Priority = Priority.HIGH,
                                   request_id: Optional[str] = None, *args, **kwargs) -> None:
        """
        Queue a Prices API request.
        
        Args:
            func: Async function to call
            priority: Request priority level
            request_id: Optional identifier for logging
            *args, **kwargs: Arguments to pass to func
        """
        if not self.workers_started:
            await self.start_workers()
        
        request_id = request_id or f"prices_{int(time.time() * 1000)}"
        
        request = APIRequest(
            priority=priority,
            api_type='prices',
            func=func,
            args=args,
            kwargs=kwargs,
            timestamp=time.time(),
            request_id=request_id
        )
        
        await self.prices_queue.put((priority, request))
        self.stats['prices_queued'] += 1
        logger.debug(f"📥 Queued Prices request: {request_id} (priority={priority.name})")
    
    async def set_watchdog_active(self, active: bool):
        """
        Signal that the streamlined watchdog is active/inactive.
        
        When active, all other API calls are slowed down to give watchdog priority.
        """
        async with self.watchdog_lock:
            if active and not self.watchdog_active:
                self.watchdog_active = True
                self.stats['watchdog_activations'] += 1
                logger.warning("🚨 WATCHDOG ACTIVE - Slowing down other API calls")
            elif not active and self.watchdog_active:
                self.watchdog_active = False
                logger.info("✅ Watchdog finished - Resuming normal API rates")
    
    def get_stats(self) -> dict:
        """Get queue statistics"""
        return {
            **self.stats,
            'mediawiki_queue_size': self.mediawiki_queue.qsize(),
            'prices_queue_size': self.prices_queue.qsize(),
            'watchdog_active': self.watchdog_active
        }


# Global singleton instance
_api_queue_manager = None


def get_api_queue_manager() -> APIQueueManager:
    """Get or create the global API queue manager"""
    global _api_queue_manager
    if _api_queue_manager is None:
        _api_queue_manager = APIQueueManager()
    return _api_queue_manager

