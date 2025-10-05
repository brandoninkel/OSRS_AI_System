#!/usr/bin/env python3
"""
Fix Analytics Schema

Changes the price_analytics table to store only the LATEST analytics per item
instead of historical snapshots.

This will:
1. Create a new table with UNIQUE(item_id) constraint
2. Copy only the latest analytics for each item
3. Drop old table and rename new one
4. Add all the missing MA columns (14d, 50d, 100d, 200d)
"""

import sys
import os
import sqlite3
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_analytics_schema():
    """Fix the analytics schema to store only latest per item"""
    
    db_path = "data/price_history.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return False
    
    print("=" * 80)
    print("🔧 Fixing Analytics Schema")
    print("=" * 80)
    print()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current state
        cursor.execute("SELECT COUNT(*) FROM price_analytics")
        old_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT item_id) FROM price_analytics")
        unique_items = cursor.fetchone()[0]
        
        print(f"Current state:")
        print(f"  - Total analytics rows: {old_count:,}")
        print(f"  - Unique items: {unique_items:,}")
        print(f"  - Duplicates: {old_count - unique_items:,}")
        print()
        
        # Create new table with proper schema
        print("Creating new analytics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_analytics_new (
                item_id INTEGER PRIMARY KEY,
                item_name TEXT NOT NULL,
                
                -- Current price
                current_price INTEGER,
                
                -- Price changes
                price_change_1d INTEGER,
                price_change_1d_pct REAL,
                price_change_7d INTEGER,
                price_change_7d_pct REAL,
                price_change_30d INTEGER,
                price_change_30d_pct REAL,
                
                -- Volatility
                volatility_7d REAL,
                volatility_30d REAL,
                
                -- Moving averages (all periods)
                ma_7d REAL,
                ma_14d REAL,
                ma_30d REAL,
                ma_50d REAL,
                ma_100d REAL,
                ma_200d REAL,
                
                -- Volume metrics
                avg_volume_24h INTEGER,
                total_volume_24h INTEGER,
                
                -- OHLC
                open_price INTEGER,
                high_price INTEGER,
                low_price INTEGER,
                close_price INTEGER,
                
                -- Metadata
                last_updated TEXT NOT NULL
            )
        """)
        
        # Copy latest analytics for each item
        print("Copying latest analytics for each item...")
        cursor.execute("""
            INSERT INTO price_analytics_new
            (item_id, item_name, price_change_1d, price_change_1d_pct,
             volatility_7d, avg_volume_24h, total_volume_24h,
             ma_7d, ma_30d, last_updated)
            SELECT
                item_id,
                item_name,
                price_change_24h,
                price_change_24h_percent,
                volatility_24h,
                avg_volume_24h,
                total_volume_24h,
                ma_7d,
                ma_30d,
                calculated_at
            FROM price_analytics
            WHERE (item_id, calculated_at) IN (
                SELECT item_id, MAX(calculated_at)
                FROM price_analytics
                GROUP BY item_id
            )
        """)
        
        new_count = cursor.rowcount
        print(f"  ✅ Copied {new_count:,} latest analytics")
        print()
        
        # Drop old table
        print("Dropping old table...")
        cursor.execute("DROP TABLE price_analytics")
        
        # Rename new table
        print("Renaming new table...")
        cursor.execute("ALTER TABLE price_analytics_new RENAME TO price_analytics")
        
        # Create index
        print("Creating index...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_item ON price_analytics(item_id)")
        
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM price_analytics")
        final_count = cursor.fetchone()[0]
        
        print()
        print("=" * 80)
        print("✅ Schema Fixed Successfully!")
        print("=" * 80)
        print(f"Before: {old_count:,} rows ({old_count - unique_items:,} duplicates)")
        print(f"After: {final_count:,} rows (one per item)")
        print(f"Removed: {old_count - final_count:,} duplicate entries")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = fix_analytics_schema()
    sys.exit(0 if success else 1)

