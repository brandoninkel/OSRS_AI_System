#!/usr/bin/env python3
"""
GE Update Daemon - Fixed Version

Continuously runs incremental updates every 5 minutes.
Includes database verification and repair on startup.

Features:
- Database schema verification and repair
- Recomputes missing/incomplete analytics
- Runs every 5 minutes (configurable)
- Robust error handling
- Comprehensive logging
- Graceful shutdown on Ctrl+C
- Auto-recovery from failures
"""

import sys
import os
import time
import logging
import sqlite3
from datetime import datetime
import signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.price_history import PriceHistoryService
from api.price_analytics import PriceAnalyticsService
from api.config import get_headers, GE_ENDPOINTS
import requests

# Configuration
UPDATE_INTERVAL = 300  # 5 minutes (300 seconds)
COMPUTE_ANALYTICS = True
MA_PERIODS = [7, 14, 30, 50, 100, 200]

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"ge_daemon_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    logger.info("Shutdown signal received. Finishing current update...")
    print("\n⚠️  Shutdown signal received. Finishing current update...")
    shutdown_flag = True


def verify_and_repair_database():
    """Verify database schema and repair if needed"""
    logger.info("=" * 80)
    logger.info("🔍 Verifying database schema...")
    print("\n🔍 Verifying database schema...")
    
    try:
        price_service = PriceHistoryService()
        conn = sqlite3.connect(price_service.db_path)
        cursor = conn.cursor()
        
        # Check if analytics table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='price_analytics'
        """)
        
        if not cursor.fetchone():
            logger.error("❌ price_analytics table not found!")
            print("❌ price_analytics table not found!")
            conn.close()
            return False
        
        # Check analytics table columns
        cursor.execute("PRAGMA table_info(price_analytics)")
        columns = {col[1] for col in cursor.fetchall()}
        
        required_columns = {
            'item_id', 'item_name', 'current_price',
            'price_change_1d', 'price_change_1d_pct',
            'price_change_7d', 'price_change_7d_pct',
            'price_change_30d', 'price_change_30d_pct',
            'volatility_7d', 'volatility_30d',
            'ma_7d', 'ma_14d', 'ma_30d', 'ma_50d', 'ma_100d', 'ma_200d',
            'avg_volume_24h', 'total_volume_24h',
            'open_price', 'high_price', 'low_price', 'close_price',
            'last_updated'
        }
        
        missing_columns = required_columns - columns
        
        if missing_columns:
            logger.warning(f"⚠️  Missing columns: {missing_columns}")
            print(f"⚠️  Missing columns: {missing_columns}")
            print("   Running schema fix...")
            conn.close()
            
            # Run schema fix
            import subprocess
            result = subprocess.run(
                ['python3', 'scripts/fix_analytics_schema.py'],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Schema fix failed: {result.stderr}")
                print(f"❌ Schema fix failed!")
                return False
            
            logger.info("✅ Schema fixed successfully")
            print("✅ Schema fixed successfully")
            
            # Reconnect
            conn = sqlite3.connect(price_service.db_path)
            cursor = conn.cursor()
        
        # Check for items with incomplete analytics
        cursor.execute("""
            SELECT COUNT(*) FROM price_analytics
            WHERE ma_7d IS NULL OR ma_14d IS NULL OR ma_30d IS NULL 
               OR ma_50d IS NULL OR ma_100d IS NULL OR ma_200d IS NULL
        """)
        incomplete_count = cursor.fetchone()[0]
        
        # Check total items with price data
        cursor.execute("SELECT COUNT(DISTINCT item_id) FROM price_history_complete")
        total_items = cursor.fetchone()[0]
        
        # Check items with analytics
        cursor.execute("SELECT COUNT(*) FROM price_analytics")
        analytics_count = cursor.fetchone()[0]
        
        missing_analytics = total_items - analytics_count
        
        logger.info(f"📊 Database Status:")
        logger.info(f"   - Total items with price data: {total_items:,}")
        logger.info(f"   - Items with analytics: {analytics_count:,}")
        logger.info(f"   - Missing analytics: {missing_analytics:,}")
        logger.info(f"   - Incomplete analytics: {incomplete_count:,}")
        
        print(f"\n📊 Database Status:")
        print(f"   - Total items with price data: {total_items:,}")
        print(f"   - Items with analytics: {analytics_count:,}")
        print(f"   - Missing analytics: {missing_analytics:,}")
        print(f"   - Incomplete analytics: {incomplete_count:,}")
        
        needs_repair = missing_analytics > 0 or incomplete_count > 0
        
        if needs_repair:
            logger.info(f"⚠️  Database needs repair: {missing_analytics + incomplete_count:,} items need analytics")
            print(f"\n⚠️  Database needs repair: {missing_analytics + incomplete_count:,} items need analytics")
            print("   Will recompute during first update cycle...")
        else:
            logger.info("✅ Database is healthy")
            print("✅ Database is healthy")
        
        conn.close()
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}", exc_info=True)
        print(f"❌ Database verification failed: {e}")
        return False


def recompute_missing_analytics():
    """Recompute analytics for items that are missing or incomplete"""
    logger.info("=" * 80)
    logger.info("🔧 Recomputing missing/incomplete analytics...")
    print("\n🔧 Recomputing missing/incomplete analytics...")
    
    try:
        price_service = PriceHistoryService()
        analytics_service = PriceAnalyticsService(price_service.db_path)
        
        conn = sqlite3.connect(price_service.db_path)
        cursor = conn.cursor()
        
        # Get items that need analytics (missing or incomplete)
        cursor.execute("""
            SELECT DISTINCT ph.item_id, ph.item_name
            FROM price_history_complete ph
            LEFT JOIN price_analytics pa ON ph.item_id = pa.item_id
            WHERE pa.item_id IS NULL
               OR pa.ma_7d IS NULL OR pa.ma_14d IS NULL OR pa.ma_30d IS NULL
               OR pa.ma_50d IS NULL OR pa.ma_100d IS NULL OR pa.ma_200d IS NULL
            ORDER BY ph.item_id
        """)
        
        items_to_fix = cursor.fetchall()
        conn.close()
        
        if not items_to_fix:
            logger.info("✅ No items need analytics recomputation")
            print("✅ No items need analytics recomputation")
            return True
        
        total = len(items_to_fix)
        logger.info(f"Found {total:,} items needing analytics")
        print(f"Found {total:,} items needing analytics")
        print()
        
        success_count = 0
        error_count = 0
        
        for i, (item_id, item_name) in enumerate(items_to_fix, 1):
            try:
                analytics = analytics_service.compute_comprehensive_analytics(
                    item_id, item_name, ma_periods=MA_PERIODS
                )
                analytics_service.store_analytics(analytics)
                success_count += 1
                
                if i % 100 == 0:
                    pct = (i / total) * 100
                    print(f"   Progress: {i:,}/{total:,} ({pct:.1f}%) - {item_name}")
                    logger.info(f"   Progress: {i:,}/{total:,} ({pct:.1f}%)")
                
            except Exception as e:
                error_count += 1
                logger.error(f"   Error for {item_name} (ID {item_id}): {e}")
        
        print()
        logger.info(f"✅ Recomputation complete: {success_count:,} success, {error_count:,} errors")
        print(f"✅ Recomputation complete: {success_count:,} success, {error_count:,} errors")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Recomputation failed: {e}", exc_info=True)
        print(f"❌ Recomputation failed: {e}")
        return False


def fetch_latest_prices():
    """Fetch latest prices for all items"""
    try:
        response = requests.get(GE_ENDPOINTS["latest"], headers=get_headers(), timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch latest prices: HTTP {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error fetching latest prices: {e}")
        return {}


def fetch_item_mapping():
    """Fetch item ID to name mapping"""
    try:
        response = requests.get(GE_ENDPOINTS["mapping"], headers=get_headers(), timeout=30)
        if response.status_code == 200:
            data = response.json()
            return {item['id']: item['name'] for item in data}
        else:
            logger.error(f"Failed to fetch mapping: HTTP {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error fetching mapping: {e}")
        return {}


def perform_update():
    """Perform a single incremental update"""
    logger.info("=" * 80)
    logger.info("Starting incremental update")
    
    try:
        price_service = PriceHistoryService()
        analytics_service = PriceAnalyticsService(price_service.db_path)
        
        logger.info("Fetching item mapping...")
        item_mapping = fetch_item_mapping()
        if not item_mapping:
            logger.error("Failed to fetch item mapping")
            return False
        logger.info(f"Fetched {len(item_mapping)} items")
        
        logger.info("Fetching latest prices...")
        latest_data = fetch_latest_prices()
        if not latest_data or 'data' not in latest_data:
            logger.error("Failed to fetch latest prices")
            return False
        
        prices = latest_data['data']
        logger.info(f"Fetched prices for {len(prices)} items")
        
        new_records = 0
        updated_items = []
        
        for item_id_str, price_info in prices.items():
            try:
                item_id = int(item_id_str)
                item_name = item_mapping.get(item_id, f"Unknown Item {item_id}")
                
                last_timestamp = price_service.get_last_timestamp(item_id)
                high_time = price_info.get('highTime', 0)
                low_time = price_info.get('lowTime', 0)
                new_timestamp = max(high_time, low_time) * 1000 if (high_time or low_time) else None

                # Skip if we have both timestamps and new one is not newer
                if last_timestamp is not None and new_timestamp is not None and new_timestamp <= last_timestamp:
                    continue

                # Skip if we don't have a valid new timestamp
                if new_timestamp is None:
                    continue
                
                price = price_info.get('high', 0)
                if price > 0:
                    price_service.record_complete_history(
                        item_id=item_id,
                        item_name=item_name,
                        price=price,
                        volume=None,
                        timestamp=new_timestamp
                    )
                    new_records += 1
                    updated_items.append((item_id, item_name))
            except Exception as e:
                # Silently skip items with bad timestamp data (non-critical)
                if "NoneType" in str(e) and "not supported between" in str(e):
                    continue
                logger.error(f"Error processing item {item_id_str}: {e}")
                continue
        
        logger.info(f"Inserted {new_records} new records")
        
        if COMPUTE_ANALYTICS and updated_items:
            logger.info(f"Updating analytics for {len(updated_items)} items...")
            for item_id, item_name in updated_items:
                try:
                    analytics = analytics_service.compute_comprehensive_analytics(
                        item_id, item_name, ma_periods=MA_PERIODS
                    )
                    analytics_service.store_analytics(analytics)
                except Exception as e:
                    logger.error(f"Error updating analytics for {item_name}: {e}")
            
            logger.info(f"Updated analytics for {len(updated_items)} items")
        
        logger.info(f"Update complete: {new_records} new records, {len(updated_items)} analytics updated")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"Error during update: {e}", exc_info=True)
        return False


def main():
    """Main daemon loop"""
    global shutdown_flag
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 80)
    print("🔄 OSRS GE Update Daemon (Fixed Version)")
    print("=" * 80)
    print(f"📝 Logging to: {log_file}")
    print(f"⏱️  Update interval: {UPDATE_INTERVAL} seconds ({UPDATE_INTERVAL // 60} minutes)")
    print(f"📊 Analytics: {'Enabled' if COMPUTE_ANALYTICS else 'Disabled'}")
    print(f"📈 MA Periods: {MA_PERIODS}")
    print()
    
    logger.info("=" * 80)
    logger.info("GE Update Daemon started (Fixed Version)")
    logger.info(f"Update interval: {UPDATE_INTERVAL} seconds")
    logger.info(f"Analytics: {'Enabled' if COMPUTE_ANALYTICS else 'Disabled'}")
    logger.info("=" * 80)
    
    # Verify and repair database on startup
    if not verify_and_repair_database():
        print("\n❌ Database verification failed. Exiting.")
        logger.error("Database verification failed. Exiting.")
        return 1
    
    # Recompute missing analytics
    if not recompute_missing_analytics():
        print("\n⚠️  Analytics recomputation failed, but continuing...")
        logger.warning("Analytics recomputation failed, but continuing...")
    
    print("\n" + "=" * 80)
    print("✅ Startup complete. Starting update loop...")
    print("Press Ctrl+C to stop gracefully")
    print("=" * 80)
    print()
    
    update_count = 0
    
    while not shutdown_flag:
        update_count += 1
        logger.info(f"Update #{update_count} starting...")
        print(f"🔄 Update #{update_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = perform_update()
        
        if success:
            print(f"✅ Update #{update_count} complete")
        else:
            print(f"⚠️  Update #{update_count} failed (will retry)")
        
        if not shutdown_flag:
            print(f"⏳ Next update in {UPDATE_INTERVAL // 60} minutes...")
            print()
            
            for _ in range(UPDATE_INTERVAL):
                if shutdown_flag:
                    break
                time.sleep(1)
    
    print()
    print("=" * 80)
    print("✅ Daemon stopped gracefully")
    print(f"📊 Total updates performed: {update_count}")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Daemon stopped gracefully")
    logger.info(f"Total updates performed: {update_count}")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='GE Update Daemon')
    parser.add_argument('--single-update', action='store_true',
                       help='Run a single update and exit (for watchdog integration)')
    args = parser.parse_args()

    if args.single_update:
        # Single update mode for watchdog integration
        try:
            logger.info("Running single GE update (watchdog mode)")

            # Verify database first
            if not verify_and_repair_database():
                logger.error("Database verification failed")
                sys.exit(1)

            # Recompute missing analytics if needed
            recompute_missing_analytics()

            # Run single update
            success = perform_update()

            if success:
                logger.info("✅ Single update completed successfully")
                sys.exit(0)
            else:
                logger.error("❌ Single update failed")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Fatal error in single update: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Normal daemon mode
        try:
            sys.exit(main())
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            print(f"\n❌ Fatal error: {e}")
            sys.exit(1)

