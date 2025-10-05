#!/usr/bin/env python3
"""
Complete GE Historical Data Seed Script

Fetches complete historical price data for ALL OSRS items from the
Weird Gloop API and stores it in the database.

Features:
- Fetches all 4,307 items from /mapping
- Fetches complete history for each item (~14.8M data points)
- Handles both old (no volume) and new (with volume) formats
- Prevents duplicates automatically
- Shows progress with ETA
- Completes in ~7 minutes with parallel processing
- Stores ~1.23GB of data
"""

import sys
import os
import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.price_history import PriceHistoryService
from api.price_analytics import PriceAnalyticsService
from api.config import get_headers, PRICES_API_BASE

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"ge_seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API endpoints
MAPPING_URL = f"{PRICES_API_BASE}/mapping"
HISTORY_URL_TEMPLATE = "https://api.weirdgloop.org/exchange/history/osrs/all?id={item_id}"

# Configuration
CONCURRENT_REQUESTS = 10  # 10 requests per second (respects API guidelines)
BATCH_SIZE = 1000  # Insert 1000 records at a time
SKIP_EXISTING = True  # Skip items that already have data
COMPUTE_ANALYTICS = True  # Compute analytics after seeding
MA_PERIODS = [7, 14, 30, 50, 100, 200]  # Moving average periods for charting


class ProgressTracker:
    """Track and display progress"""

    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.total_records = 0
        self.items_with_data = 0
        self.items_without_data = 0
        self.items_for_analytics = []  # Track items that need analytics
        self.start_time = time.time()
        
    def update(self, has_data: bool, record_count: int = 0, item_info: tuple = None):
        """Update progress"""
        self.completed_items += 1
        if has_data:
            self.items_with_data += 1
            self.total_records += record_count
            if item_info and record_count > 0:
                self.items_for_analytics.append(item_info)
        else:
            self.items_without_data += 1
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.completed_items > 0:
            avg_time_per_item = elapsed / self.completed_items
            remaining_items = self.total_items - self.completed_items
            eta_seconds = avg_time_per_item * remaining_items
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = "calculating..."
        
        # Display progress
        percent = (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
        print(f"\r📊 Progress: {self.completed_items}/{self.total_items} ({percent:.1f}%) | "
              f"Records: {self.total_records:,} | "
              f"With data: {self.items_with_data} | "
              f"No data: {self.items_without_data} | "
              f"ETA: {eta}", end='', flush=True)
    
    def finish(self):
        """Display final statistics"""
        elapsed = time.time() - self.start_time
        print(f"\n\n✅ Complete!")
        print(f"⏱️  Total time: {timedelta(seconds=int(elapsed))}")
        print(f"📦 Total items processed: {self.total_items}")
        print(f"✅ Items with data: {self.items_with_data}")
        print(f"❌ Items without data: {self.items_without_data}")
        print(f"📊 Total records inserted: {self.total_records:,}")
        print(f"💾 Estimated storage: {self.total_records * 80 / 1024 / 1024:.2f} MB")


async def fetch_item_mapping(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    """Fetch all item metadata from /mapping"""
    print("📥 Fetching item mapping...")
    
    async with session.get(MAPPING_URL, headers=get_headers()) as response:
        if response.status == 200:
            data = await response.json()
            print(f"✅ Fetched {len(data)} items from mapping")
            return data
        else:
            print(f"❌ Failed to fetch mapping: HTTP {response.status}")
            return []


async def fetch_item_history(session: aiohttp.ClientSession, item_id: int) -> Optional[Dict[str, Any]]:
    """Fetch complete history for a single item"""
    url = HISTORY_URL_TEMPLATE.format(item_id=item_id)
    
    try:
        async with session.get(url, headers=get_headers()) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                return None
    except Exception as e:
        print(f"\n⚠️  Error fetching item {item_id}: {e}")
        return None


async def process_item_batch(session: aiohttp.ClientSession, items: List[Dict[str, Any]],
                             price_service: PriceHistoryService, progress: ProgressTracker):
    """Process a batch of items"""
    tasks = []
    items_to_fetch = []

    for item in items:
        item_id = item['id']

        # Skip if already has data (optional optimization)
        if SKIP_EXISTING and price_service.has_complete_history(item_id):
            progress.update(has_data=True, record_count=0, item_info=(item_id, item['name']))
            continue

        items_to_fetch.append(item)
        tasks.append(fetch_item_history(session, item_id))

    # Fetch all histories in parallel
    if not tasks:
        return

    results = await asyncio.gather(*tasks)

    # Process results
    for i, result in enumerate(results):
        if result is None:
            progress.update(has_data=False)
            continue

        # Find corresponding item
        item = items_to_fetch[i] if i < len(items_to_fetch) else None
        if not item:
            continue

        item_id = item['id']
        item_name = item['name']

        # Extract history data
        if str(item_id) not in result:
            progress.update(has_data=False)
            continue

        history = result[str(item_id)]

        if not history or len(history) == 0:
            progress.update(has_data=False)
            continue

        # Prepare records for bulk insert
        records = []
        for data_point in history:
            records.append({
                'item_id': item_id,
                'item_name': item_name,
                'price': data_point['price'],
                'volume': data_point.get('volume'),
                'timestamp': data_point['timestamp']
            })

        # Bulk insert in batches
        for j in range(0, len(records), BATCH_SIZE):
            batch = records[j:j + BATCH_SIZE]
            price_service.bulk_record_complete_history(batch)

        progress.update(has_data=True, record_count=len(records), item_info=(item_id, item_name))


async def seed_complete_history():
    """Main function to seed complete historical data"""
    print("=" * 80)
    print("🚀 OSRS Complete GE Historical Data Seed")
    print("=" * 80)
    print(f"📝 Logging to: {log_file}")
    print()

    logger.info("=" * 80)
    logger.info("Starting complete GE historical data seed")
    logger.info(f"Configuration: CONCURRENT_REQUESTS={CONCURRENT_REQUESTS}, BATCH_SIZE={BATCH_SIZE}, SKIP_EXISTING={SKIP_EXISTING}")
    logger.info("=" * 80)
    
    # Initialize services
    price_service = PriceHistoryService()
    
    # Create session with proper headers
    async with aiohttp.ClientSession() as session:
        # Step 1: Fetch item mapping
        items = await fetch_item_mapping(session)
        
        if not items:
            print("❌ Failed to fetch item mapping. Exiting.")
            return
        
        # Step 2: Store item metadata
        print("📝 Storing item metadata...")
        for item in items:
            price_service.record_item_metadata(
                item_id=item['id'],
                name=item['name'],
                examine=item.get('examine'),
                members=item.get('members'),
                lowalch=item.get('lowalch'),
                highalch=item.get('highalch'),
                buy_limit=item.get('limit'),
                value=item.get('value'),
                icon=item.get('icon')
            )
        print(f"✅ Stored metadata for {len(items)} items")
        print()
        
        # Step 3: Fetch complete history for all items
        print("📥 Fetching complete historical data...")
        print(f"⚙️  Configuration:")
        print(f"   - Concurrent requests: {CONCURRENT_REQUESTS}")
        print(f"   - Batch size: {BATCH_SIZE}")
        print(f"   - Skip existing: {SKIP_EXISTING}")
        print()
        
        progress = ProgressTracker(len(items))
        
        # Process items in batches
        for i in range(0, len(items), CONCURRENT_REQUESTS):
            batch = items[i:i + CONCURRENT_REQUESTS]
            await process_item_batch(session, batch, price_service, progress)
            
            # Rate limiting: 10 requests per second
            await asyncio.sleep(1)

        progress.finish()

        # Step 4: Compute analytics
        if COMPUTE_ANALYTICS and progress.items_for_analytics:
            print()
            print("=" * 80)
            print("📊 Computing Analytics & Moving Averages")
            print("=" * 80)
            print(f"⚙️  MA Periods: {MA_PERIODS}")
            print(f"📦 Items to process: {len(progress.items_for_analytics)}")
            print()

            analytics_service = PriceAnalyticsService(price_service.db_path)

            for i, (item_id, item_name) in enumerate(progress.items_for_analytics):
                try:
                    analytics = analytics_service.compute_comprehensive_analytics(
                        item_id, item_name, ma_periods=MA_PERIODS
                    )
                    analytics_service.store_analytics(analytics)

                    # Progress
                    percent = ((i + 1) / len(progress.items_for_analytics) * 100)
                    print(f"\r📊 Analytics: {i+1}/{len(progress.items_for_analytics)} ({percent:.1f}%) | "
                          f"Current: {item_name}", end='', flush=True)
                except Exception as e:
                    print(f"\n⚠️  Error computing analytics for {item_name}: {e}")

            print()
            print()
            print("✅ Analytics computation complete!")
            print("=" * 80)


def main():
    """Entry point"""
    try:
        asyncio.run(seed_complete_history())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

