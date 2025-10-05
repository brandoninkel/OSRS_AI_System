#!/usr/bin/env python3
"""
Price History Tracking System

Stores historical price data for OSRS items to enable trend analysis
and economic forecasting.

Uses SQLite for efficient querying by item name and timestamp.
Integrates with API Queue Manager for coordinated price fetching.
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class PriceHistoryService:
    """Service for tracking and analyzing item price history"""

    def __init__(self, db_path: str = None):
        # Default to project root data directory
        if db_path is None:
            # Get the project root (parent of api directory)
            api_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(api_dir)
            db_path = os.path.join(project_root, "data", "price_history.db")

        self.db_path = db_path

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._init_database()
    
    def _init_database(self):
        """Initialize the price history database with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table 1: Latest prices (instant buy/sell)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history_latest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                high_price INTEGER,
                low_price INTEGER,
                high_time INTEGER,
                low_time INTEGER,
                timestamp INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(item_id, timestamp)
            )
        """)

        # Table 2: Timeseries data (historical averages with volume)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history_timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                item_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                timestep TEXT NOT NULL,
                avg_high_price INTEGER,
                avg_low_price INTEGER,
                high_price_volume INTEGER,
                low_price_volume INTEGER,
                margin INTEGER,
                margin_percent REAL,
                total_volume INTEGER,
                price_midpoint INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE(item_id, timestamp, timestep)
            )
        """)

        # Table 3: Item metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS item_metadata (
                item_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                examine TEXT,
                members BOOLEAN,
                lowalch INTEGER,
                highalch INTEGER,
                buy_limit INTEGER,
                value INTEGER,
                icon TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        # Table 4: Complete historical data (from /exchange/history/osrs/all)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history_complete (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                item_name TEXT NOT NULL,
                price INTEGER NOT NULL,
                volume INTEGER,
                timestamp INTEGER NOT NULL,
                data_source TEXT,
                has_volume BOOLEAN,
                created_at TEXT NOT NULL,
                UNIQUE(item_id, timestamp)
            )
        """)

        # Table 5: Price analytics (pre-computed metrics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                item_name TEXT NOT NULL,
                volatility_24h REAL,
                price_change_24h INTEGER,
                price_change_24h_percent REAL,
                avg_volume_24h INTEGER,
                total_volume_24h INTEGER,
                liquidity_score REAL,
                avg_margin_24h INTEGER,
                avg_margin_percent_24h REAL,
                trend_7d TEXT,
                trend_30d TEXT,
                ma_7d INTEGER,
                ma_30d INTEGER,
                risk_category TEXT,
                flip_score REAL,
                calculated_at TEXT NOT NULL,
                UNIQUE(item_id, calculated_at)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_latest_item_time
            ON price_history_latest(item_id, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timeseries_item_time
            ON price_history_timeseries(item_id, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timeseries_timestep
            ON price_history_timeseries(timestep, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timeseries_volume
            ON price_history_timeseries(total_volume DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_item
            ON price_analytics(item_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complete_item_time
            ON price_history_complete(item_id, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complete_timestamp
            ON price_history_complete(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_complete_volume
            ON price_history_complete(volume DESC) WHERE volume IS NOT NULL
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Price history database initialized with comprehensive schema")
    
    def record_latest_price(self, item_name: str, item_id: int, high_price: int,
                           low_price: int, high_time: int, low_time: int):
        """Record a latest price data point"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = int(time.time())
        created_at = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO price_history_latest
                (item_name, item_id, high_price, low_price, high_time, low_time, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (item_name, item_id, high_price, low_price, high_time, low_time, timestamp, created_at))
            
            conn.commit()
            logger.debug(f"📊 Recorded price for {item_name}: {high_price} GP")
        except Exception as e:
            logger.error(f"❌ Error recording price: {e}")
        finally:
            conn.close()
    
    def get_price_history(self, item_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get price history for an item over the last N hours"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert hours to milliseconds for timestamp comparison
        cutoff_time = int((time.time() - (hours * 3600)) * 1000)

        cursor.execute("""
            SELECT item_name, item_id, price, volume, timestamp, data_source
            FROM price_history_complete
            WHERE item_name = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (item_name, cutoff_time))

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                'item_name': row[0],
                'item_id': row[1],
                'price': row[2],
                'volume': row[3],
                'timestamp': row[4],
                'data_source': row[5]
            })

        return history
    
    def get_price_trend(self, item_name: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze price trend for an item"""
        history = self.get_price_history(item_name, hours)
        
        if len(history) < 2:
            return {
                'item_name': item_name,
                'trend': 'insufficient_data',
                'data_points': len(history),
                'message': 'Need at least 2 data points to determine trend'
            }
        
        # Calculate trend
        first_price = history[0]['price']
        last_price = history[-1]['price']

        price_change = last_price - first_price
        percent_change = (price_change / first_price * 100) if first_price > 0 else 0

        # Determine trend direction
        if percent_change > 5:
            trend = 'rising'
        elif percent_change < -5:
            trend = 'falling'
        else:
            trend = 'stable'

        # Calculate volatility (standard deviation of prices)
        prices = [h['price'] for h in history]
        avg_price = sum(prices) // len(prices)
        variance = sum((p - avg_price) ** 2 for p in prices) // len(prices)
        volatility = int(variance ** 0.5)

        return {
            'item_name': item_name,
            'trend': trend,
            'first_price': first_price,
            'last_price': last_price,
            'price_change': price_change,
            'percent_change': round(percent_change, 2),
            'avg_price': avg_price,
            'volatility': volatility,
            'data_points': len(history),
            'time_range_hours': hours,
            'highest_price': max(h['price'] for h in history),
            'lowest_price': min(h['price'] for h in history)
        }
    
    def get_multiple_trends(self, item_names: List[str], hours: int = 24) -> List[Dict[str, Any]]:
        """Get trends for multiple items"""
        return [self.get_price_trend(item, hours) for item in item_names]
    
    def cleanup_old_data(self, days: int = 30):
        """Remove price data older than N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = int(time.time()) - (days * 24 * 3600)
        
        cursor.execute("""
            DELETE FROM price_history_latest
            WHERE timestamp < ?
        """, (cutoff_time,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"🗑️ Cleaned up {deleted} old price records (older than {days} days)")
        return deleted

    def get_multiple_trends(self, item_names: List[str], hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get price trends for multiple items.

        Args:
            item_names: List of item names
            hours: Time range in hours

        Returns:
            List of trend dictionaries (one per item)
        """
        trends = []
        for item_name in item_names:
            trend = self.get_price_trend(item_name, hours)
            trend['item_name'] = item_name
            trends.append(trend)

        return trends

    def get_tracked_items(self, limit: int = 100) -> List[str]:
        """
        Get list of items that have been tracked (have price history).
        Returns items sorted by most recent activity.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of item names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT item_name
            FROM price_history_complete
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        items = [row[0] for row in cursor.fetchall()]
        conn.close()

        return items

    def search_item_names(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for item names in the GE database that match the query.
        Uses case-insensitive LIKE search for autocomplete functionality.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching item names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use LIKE with wildcards for fuzzy matching
        search_pattern = f"%{query}%"

        cursor.execute("""
            SELECT DISTINCT item_name
            FROM price_history_complete
            WHERE item_name LIKE ? COLLATE NOCASE
            ORDER BY
                CASE
                    WHEN item_name LIKE ? COLLATE NOCASE THEN 1
                    ELSE 2
                END,
                item_name
            LIMIT ?
        """, (search_pattern, f"{query}%", limit))

        items = [row[0] for row in cursor.fetchall()]
        conn.close()

        return items

    def record_timeseries(self, item_id: int, item_name: str, timestamp: int,
                         timestep: str, avg_high_price: int, avg_low_price: int,
                         high_price_volume: int, low_price_volume: int):
        """
        Record a timeseries data point with calculated metrics

        Args:
            item_id: Item ID
            item_name: Item name
            timestamp: Unix timestamp
            timestep: '5m', '1h', '6h', or '24h'
            avg_high_price: Average high price
            avg_low_price: Average low price
            high_price_volume: High price volume
            low_price_volume: Low price volume
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate derived metrics
        margin = avg_high_price - avg_low_price if avg_high_price and avg_low_price else 0
        margin_percent = (margin / avg_low_price * 100) if avg_low_price and avg_low_price > 0 else 0
        total_volume = (high_price_volume or 0) + (low_price_volume or 0)
        price_midpoint = (avg_high_price + avg_low_price) // 2 if avg_high_price and avg_low_price else 0

        created_at = datetime.now().isoformat()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO price_history_timeseries
                (item_id, item_name, timestamp, timestep, avg_high_price, avg_low_price,
                 high_price_volume, low_price_volume, margin, margin_percent, total_volume,
                 price_midpoint, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (item_id, item_name, timestamp, timestep, avg_high_price, avg_low_price,
                  high_price_volume, low_price_volume, margin, margin_percent, total_volume,
                  price_midpoint, created_at))

            conn.commit()
            logger.debug(f"📊 Recorded {timestep} timeseries for {item_name}")
        except Exception as e:
            logger.error(f"❌ Failed to record timeseries for {item_name}: {e}")
        finally:
            conn.close()

    def record_item_metadata(self, item_id: int, name: str, examine: str = None,
                            members: bool = None, lowalch: int = None, highalch: int = None,
                            buy_limit: int = None, value: int = None, icon: str = None):
        """
        Record item metadata

        Args:
            item_id: Item ID
            name: Item name
            examine: Examine text
            members: Members only
            lowalch: Low alchemy value
            highalch: High alchemy value
            buy_limit: GE buy limit
            value: Store value
            icon: Icon filename
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updated_at = datetime.now().isoformat()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO item_metadata
                (item_id, name, examine, members, lowalch, highalch, buy_limit, value, icon, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (item_id, name, examine, members, lowalch, highalch, buy_limit, value, icon, updated_at))

            conn.commit()
            logger.debug(f"📝 Recorded metadata for {name}")
        except Exception as e:
            logger.error(f"❌ Failed to record metadata for {name}: {e}")
        finally:
            conn.close()

    def record_complete_history(self, item_id: int, item_name: str, price: int,
                                volume: Optional[int], timestamp: int):
        """
        Record a complete historical data point

        Args:
            item_id: Item ID
            item_name: Item name
            price: Price in GP
            volume: Trade volume (None for pre-RuneLite data)
            timestamp: Unix timestamp in milliseconds
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Determine data source
        data_source = 'runelite' if volume is not None else 'jagex'
        has_volume = volume is not None
        created_at = datetime.now().isoformat()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO price_history_complete
                (item_id, item_name, price, volume, timestamp, data_source, has_volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (item_id, item_name, price, volume, timestamp, data_source, has_volume, created_at))

            conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to record complete history for {item_name}: {e}")
        finally:
            conn.close()

    def bulk_record_complete_history(self, records: List[Dict[str, Any]]):
        """
        Bulk insert complete historical data points for efficiency

        Args:
            records: List of dicts with keys: item_id, item_name, price, volume, timestamp
        """
        if not records:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        created_at = datetime.now().isoformat()

        try:
            data = []
            for record in records:
                volume = record.get('volume')
                data_source = 'runelite' if volume is not None else 'jagex'
                has_volume = volume is not None

                data.append((
                    record['item_id'],
                    record['item_name'],
                    record['price'],
                    volume,
                    record['timestamp'],
                    data_source,
                    has_volume,
                    created_at
                ))

            cursor.executemany("""
                INSERT OR IGNORE INTO price_history_complete
                (item_id, item_name, price, volume, timestamp, data_source, has_volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, data)

            conn.commit()
            logger.debug(f"📊 Bulk inserted {len(records)} historical records")
        except Exception as e:
            logger.error(f"❌ Failed to bulk insert historical data: {e}")
        finally:
            conn.close()

    def get_last_timestamp(self, item_id: int) -> Optional[int]:
        """
        Get the last timestamp for an item in complete history

        Args:
            item_id: Item ID

        Returns:
            Last timestamp in milliseconds, or None if no data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MAX(timestamp)
            FROM price_history_complete
            WHERE item_id = ?
        """, (item_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] else None

    def has_complete_history(self, item_id: int) -> bool:
        """
        Check if an item has complete historical data

        Args:
            item_id: Item ID

        Returns:
            True if item has data, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM price_history_complete
            WHERE item_id = ?
        """, (item_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] > 0 if result else False


# Global instance
_price_history_service = None


def get_price_history_service() -> PriceHistoryService:
    """Get or create the global price history service instance"""
    global _price_history_service
    if _price_history_service is None:
        _price_history_service = PriceHistoryService()
    return _price_history_service

