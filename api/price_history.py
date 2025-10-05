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
        """Initialize the price history database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create price_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL,
                item_id INTEGER,
                high_price INTEGER,
                low_price INTEGER,
                high_time INTEGER,
                low_time INTEGER,
                timestamp INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(item_name, timestamp)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_item_timestamp 
            ON price_history(item_name, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ Price history database initialized")
    
    def record_price(self, item_name: str, item_id: int, high_price: int, 
                    low_price: int, high_time: int, low_time: int):
        """Record a price data point"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = int(time.time())
        created_at = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO price_history 
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
        
        cutoff_time = int(time.time()) - (hours * 3600)
        
        cursor.execute("""
            SELECT item_name, item_id, high_price, low_price, high_time, low_time, timestamp, created_at
            FROM price_history
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
                'high_price': row[2],
                'low_price': row[3],
                'high_time': row[4],
                'low_time': row[5],
                'timestamp': row[6],
                'created_at': row[7]
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
        first_price = (history[0]['high_price'] + history[0]['low_price']) // 2
        last_price = (history[-1]['high_price'] + history[-1]['low_price']) // 2
        
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
        prices = [(h['high_price'] + h['low_price']) // 2 for h in history]
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
            'highest_price': max(h['high_price'] for h in history),
            'lowest_price': min(h['low_price'] for h in history)
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
            DELETE FROM price_history
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


# Global instance
_price_history_service = None


def get_price_history_service() -> PriceHistoryService:
    """Get or create the global price history service instance"""
    global _price_history_service
    if _price_history_service is None:
        _price_history_service = PriceHistoryService()
    return _price_history_service

