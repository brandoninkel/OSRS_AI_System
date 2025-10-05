#!/usr/bin/env python3
"""
Price Analytics Service

Computes comprehensive metrics and moving averages from historical price data.
Supports flexible moving average periods for technical analysis and charting.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import statistics

logger = logging.getLogger(__name__)


class PriceAnalyticsService:
    """Service for computing price analytics and moving averages"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def compute_daily_ohlc(self, item_id: int, date: str) -> Optional[Dict[str, Any]]:
        """
        Compute OHLC (Open, High, Low, Close) for a specific day
        
        Args:
            item_id: Item ID
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dict with OHLC data or None if no data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all prices for the day
        start_ts = int(datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = start_ts + (24 * 60 * 60 * 1000)
        
        cursor.execute("""
            SELECT price, volume, timestamp
            FROM price_history_complete
            WHERE item_id = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp ASC
        """, (item_id, start_ts, end_ts))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        prices = [row[0] for row in rows]
        volumes = [row[1] for row in rows if row[1] is not None]
        
        return {
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'avg': sum(prices) // len(prices),
            'total_volume': sum(volumes) if volumes else None,
            'avg_volume': sum(volumes) // len(volumes) if volumes else None,
            'data_points': len(prices)
        }
    
    def compute_moving_average(self, item_id: int, periods: int, 
                               end_timestamp: Optional[int] = None) -> Optional[float]:
        """
        Compute moving average for specified number of periods (days)
        
        Args:
            item_id: Item ID
            periods: Number of days to average
            end_timestamp: End timestamp (default: now)
            
        Returns:
            Moving average price or None if insufficient data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if end_timestamp is None:
            end_timestamp = int(datetime.now().timestamp() * 1000)
        
        start_timestamp = end_timestamp - (periods * 24 * 60 * 60 * 1000)
        
        cursor.execute("""
            SELECT AVG(price)
            FROM price_history_complete
            WHERE item_id = ? AND timestamp >= ? AND timestamp <= ?
        """, (item_id, start_timestamp, end_timestamp))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] else None
    
    def compute_multiple_moving_averages(self, item_id: int, 
                                        periods: List[int]) -> Dict[int, Optional[float]]:
        """
        Compute multiple moving averages at once
        
        Args:
            item_id: Item ID
            periods: List of periods (e.g., [7, 14, 30, 50, 100, 200])
            
        Returns:
            Dict mapping period to moving average
        """
        result = {}
        for period in periods:
            result[period] = self.compute_moving_average(item_id, period)
        return result
    
    def compute_volatility(self, item_id: int, days: int = 30) -> Optional[float]:
        """
        Compute price volatility (standard deviation) over specified days
        
        Args:
            item_id: Item ID
            days: Number of days to analyze
            
        Returns:
            Standard deviation of prices or None if insufficient data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
        
        cursor.execute("""
            SELECT price
            FROM price_history_complete
            WHERE item_id = ? AND timestamp >= ? AND timestamp <= ?
        """, (item_id, start_ts, end_ts))
        
        prices = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if len(prices) < 2:
            return None
        
        return statistics.stdev(prices)
    
    def compute_price_change(self, item_id: int, days: int = 1) -> Optional[Dict[str, Any]]:
        """
        Compute price change over specified days
        
        Args:
            item_id: Item ID
            days: Number of days to look back
            
        Returns:
            Dict with price change info or None if insufficient data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
        
        # Get first and last price in period
        cursor.execute("""
            SELECT price, timestamp
            FROM price_history_complete
            WHERE item_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            LIMIT 1
        """, (item_id, start_ts, end_ts))
        
        first = cursor.fetchone()
        
        cursor.execute("""
            SELECT price, timestamp
            FROM price_history_complete
            WHERE item_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (item_id, start_ts, end_ts))
        
        last = cursor.fetchone()
        conn.close()
        
        if not first or not last:
            return None
        
        first_price, first_ts = first
        last_price, last_ts = last
        
        change = last_price - first_price
        percent_change = (change / first_price * 100) if first_price > 0 else 0
        
        return {
            'first_price': first_price,
            'last_price': last_price,
            'change': change,
            'percent_change': percent_change,
            'days': days
        }
    
    def compute_comprehensive_analytics(self, item_id: int, item_name: str,
                                       ma_periods: List[int] = None) -> Dict[str, Any]:
        """
        Compute comprehensive analytics for an item
        
        Args:
            item_id: Item ID
            item_name: Item name
            ma_periods: List of MA periods (default: [7, 14, 30, 50, 100, 200])
            
        Returns:
            Dict with all computed metrics
        """
        if ma_periods is None:
            ma_periods = [7, 14, 30, 50, 100, 200]
        
        logger.info(f"📊 Computing analytics for {item_name} (ID {item_id})")
        
        # Moving averages
        mas = self.compute_multiple_moving_averages(item_id, ma_periods)
        
        # Price changes
        change_1d = self.compute_price_change(item_id, 1)
        change_7d = self.compute_price_change(item_id, 7)
        change_30d = self.compute_price_change(item_id, 30)
        
        # Volatility
        volatility_7d = self.compute_volatility(item_id, 7)
        volatility_30d = self.compute_volatility(item_id, 30)
        
        # Volume metrics (last 24h)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = end_ts - (24 * 60 * 60 * 1000)
        
        cursor.execute("""
            SELECT SUM(volume), AVG(volume), COUNT(*)
            FROM price_history_complete
            WHERE item_id = ? AND timestamp >= ? AND volume IS NOT NULL
        """, (item_id, start_ts))
        
        volume_result = cursor.fetchone()
        total_volume_24h = volume_result[0] if volume_result[0] else 0
        avg_volume_24h = volume_result[1] if volume_result[1] else 0
        
        conn.close()
        
        return {
            'item_id': item_id,
            'item_name': item_name,
            'moving_averages': mas,
            'price_change_1d': change_1d,
            'price_change_7d': change_7d,
            'price_change_30d': change_30d,
            'volatility_7d': volatility_7d,
            'volatility_30d': volatility_30d,
            'total_volume_24h': total_volume_24h,
            'avg_volume_24h': avg_volume_24h,
            'calculated_at': datetime.now().isoformat()
        }
    
    def store_analytics(self, analytics: Dict[str, Any]):
        """
        Store computed analytics in database
        
        Args:
            analytics: Analytics dict from compute_comprehensive_analytics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract data
        item_id = analytics['item_id']
        item_name = analytics['item_name']
        
        # Price changes
        change_1d = analytics.get('price_change_1d', {})
        change_7d = analytics.get('price_change_7d', {})
        change_30d = analytics.get('price_change_30d', {})
        
        # Moving averages
        mas = analytics.get('moving_averages', {})
        
        # OHLC
        ohlc = analytics.get('ohlc_today', {})

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO price_analytics
                (item_id, item_name,
                 current_price,
                 price_change_1d, price_change_1d_pct,
                 price_change_7d, price_change_7d_pct,
                 price_change_30d, price_change_30d_pct,
                 volatility_7d, volatility_30d,
                 ma_7d, ma_14d, ma_30d, ma_50d, ma_100d, ma_200d,
                 avg_volume_24h, total_volume_24h,
                 open_price, high_price, low_price, close_price,
                 last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item_id, item_name,
                analytics.get('current_price'),
                change_1d.get('change') if change_1d else None,
                change_1d.get('percent_change') if change_1d else None,
                change_7d.get('change') if change_7d else None,
                change_7d.get('percent_change') if change_7d else None,
                change_30d.get('change') if change_30d else None,
                change_30d.get('percent_change') if change_30d else None,
                analytics.get('volatility_7d'),
                analytics.get('volatility_30d'),
                mas.get(7),
                mas.get(14),
                mas.get(30),
                mas.get(50),
                mas.get(100),
                mas.get(200),
                analytics.get('avg_volume_24h'),
                analytics.get('total_volume_24h'),
                ohlc.get('open'),
                ohlc.get('high'),
                ohlc.get('low'),
                ohlc.get('close'),
                analytics['calculated_at']
            ))

            conn.commit()
            logger.debug(f"✅ Stored analytics for {item_name}")
        except Exception as e:
            logger.error(f"❌ Failed to store analytics for {item_name}: {e}")
        finally:
            conn.close()


# Global instance
_analytics_service = None


def get_analytics_service(db_path: str = None) -> PriceAnalyticsService:
    """Get or create the global analytics service instance"""
    global _analytics_service
    if _analytics_service is None:
        if db_path is None:
            import os
            api_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(api_dir)
            db_path = os.path.join(project_root, "data", "price_history.db")
        _analytics_service = PriceAnalyticsService(db_path)
    return _analytics_service

