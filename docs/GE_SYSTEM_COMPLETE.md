# OSRS GE Price History System - Complete Documentation

**Status**: ✅ PRODUCTION READY  
**Last Updated**: 2025-10-05  
**Version**: 1.0.0

---

## 📊 System Overview

The OSRS Grand Exchange Price History System provides comprehensive historical price data and analytics for all OSRS items.

### Key Features

- **13.8 million historical price records** from March 2015 to present
- **4,321 items** with complete price history
- **6 moving average periods**: 7d, 14d, 30d, 50d, 100d, 200d
- **Automatic database verification** and repair on startup
- **Real-time updates** every 5 minutes
- **Comprehensive analytics**: price changes, volatility, OHLC, volume metrics
- **API compliant**: Respects Weird Gloop API guidelines

---

## 🗄️ Database Structure

### Location
- **Path**: `data/price_history.db`
- **Size**: ~2.1 GB
- **Format**: SQLite3
- **Status**: ⚠️ **NOT IN GIT** (too large - see `.gitignore`)

### Tables

#### 1. `price_history_complete`
Complete historical price data for all items.

**Columns**:
- `id` - Auto-increment primary key
- `item_id` - OSRS item ID
- `item_name` - Item name
- `price` - Price in GP
- `volume` - Trading volume (NULL for pre-2020 data)
- `timestamp` - Unix timestamp (milliseconds)
- `data_source` - 'jagex' or 'runelite'
- `has_volume` - Boolean flag
- `created_at` - Record creation timestamp

**Indexes**:
- `idx_complete_item_time` on `(item_id, timestamp DESC)`
- `idx_complete_timestamp` on `(timestamp DESC)`
- `idx_complete_volume` on `(volume)`

**Constraints**:
- `UNIQUE(item_id, timestamp)` - Prevents duplicates

**Data Coverage**:
- Total records: 13,796,723
- Date range: 2015-02-26 to 2025-10-05
- Unique items: 4,321

#### 2. `price_analytics`
Pre-computed analytics for fast querying.

**Columns**:
- `item_id` - PRIMARY KEY
- `item_name` - Item name
- `current_price` - Latest price
- `price_change_1d`, `price_change_1d_pct` - 1-day change
- `price_change_7d`, `price_change_7d_pct` - 7-day change
- `price_change_30d`, `price_change_30d_pct` - 30-day change
- `volatility_7d`, `volatility_30d` - Price volatility
- `ma_7d`, `ma_14d`, `ma_30d`, `ma_50d`, `ma_100d`, `ma_200d` - Moving averages
- `avg_volume_24h`, `total_volume_24h` - Volume metrics
- `open_price`, `high_price`, `low_price`, `close_price` - OHLC data
- `last_updated` - Last analytics computation timestamp

**Indexes**:
- `idx_analytics_item` on `(item_id)`

**Data Coverage**:
- Total items: 4,321
- Complete analytics (all MAs): 4,293 (99.4%)
- Incomplete analytics (< 200d history): 28 (0.6%)

---

## 🚀 Usage

### Starting the Daemon

```bash
python3 scripts/ge_update_daemon.py
```

**What it does**:
1. Verifies database schema on startup
2. Detects and repairs missing/incomplete analytics
3. Runs incremental updates every 5 minutes
4. Updates analytics for changed items
5. Logs everything to `api/logs/ge_daemon_YYYYMMDD.log`

**Stopping**:
- Press `Ctrl+C` (finishes current update before exiting)

### Initial Seed (if database doesn't exist)

```bash
python3 scripts/seed_complete_ge_history.py
```

**What it does**:
- Fetches all 4,307 items from `/mapping`
- Fetches complete history for each item
- Inserts ~13.8M records
- Computes analytics for all items
- Takes ~40 minutes

### Schema Repair (if needed)

```bash
python3 scripts/fix_analytics_schema.py
```

**What it does**:
- Fixes analytics table schema
- Removes duplicate entries
- Adds missing MA columns
- Migrates data to new structure

---

## 📁 File Structure

### Active Files

```
scripts/
├── ge_update_daemon.py              # Main daemon (auto-repair + updates)
├── seed_complete_ge_history.py      # Initial seed script
├── fix_analytics_schema.py          # Schema migration tool
└── seed-complete-ge-history.command # macOS launcher

api/
├── price_history.py                 # Database service
├── price_analytics.py               # Analytics computation
├── config.py                        # API configuration
└── logs/                            # Log files (not in git)
    ├── ge_daemon_YYYYMMDD.log
    └── ge_seed_YYYYMMDD_HHMMSS.log

data/
├── price_history.db                 # Main database (not in git)
└── price_history.db-journal         # SQLite journal (not in git)

docs/
├── GE_SYSTEM_COMPLETE.md            # This file
├── API_COMPLIANCE.md                # API compliance documentation
├── COMPLETE_GE_SEED_GUIDE.md        # Seed guide
├── GE_API_COMPLETE_RESEARCH.md      # API research
└── SYSTEM_READY.md                  # Quick start guide
```

### Archived Files (not used in production)

```
scripts/old/ge_system/
├── ge_update_daemon.py              # Old daemon (no auto-repair)
├── incremental_ge_update.py         # One-shot update script
├── populate_price_history.py        # Old population script
├── seed_ge_prices.py                # Old seed script
├── recompute_all_analytics.py       # Manual analytics recompute
└── test_complete_history.py         # Test script
```

---

## 🔧 Configuration

### Daemon Settings

Edit `scripts/ge_update_daemon.py`:

```python
UPDATE_INTERVAL = 300  # 5 minutes (300 seconds)
COMPUTE_ANALYTICS = True
MA_PERIODS = [7, 14, 30, 50, 100, 200]
```

### API Settings

Edit `api/config.py`:

```python
USER_AGENT = "OSRS-AI-RAG-System/1.0 (brandoninkel@gmail.com) Python/requests"
PRICES_API_BASE = "https://prices.runescape.wiki/api/v1/osrs"
PRICES_API_RATE_LIMIT = 10  # requests per second
```

---

## 📈 Example Queries

### Get item analytics

```python
from api.price_history import PriceHistoryService

service = PriceHistoryService()
conn = service.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT item_name, current_price, 
           ma_7d, ma_30d, ma_200d,
           volatility_7d, price_change_1d_pct
    FROM price_analytics
    WHERE item_name = 'Dragon bones'
""")

result = cursor.fetchone()
print(f"Dragon bones: {result[1]:,} gp")
print(f"MA 7d: {result[2]:,.0f} gp")
print(f"MA 30d: {result[3]:,.0f} gp")
print(f"MA 200d: {result[4]:,.0f} gp")
```

### Get historical prices

```python
cursor.execute("""
    SELECT timestamp, price, volume
    FROM price_history_complete
    WHERE item_name = 'Twisted bow'
    ORDER BY timestamp DESC
    LIMIT 100
""")

for ts, price, volume in cursor.fetchall():
    date = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
    print(f"{date}: {price:,} gp")
```

### Find trending items

```python
cursor.execute("""
    SELECT item_name, price_change_7d_pct, ma_7d
    FROM price_analytics
    WHERE ma_7d > 10000
    ORDER BY price_change_7d_pct DESC
    LIMIT 10
""")

print("Top 10 trending items (7d):")
for name, change_pct, ma in cursor.fetchall():
    print(f"{name}: {change_pct:+.2f}% (MA: {ma:,.0f} gp)")
```

---

## ⚠️ Important Notes

### Data Not in Git

The database files are **NOT stored in git** due to size (2.1 GB). They are listed in `.gitignore`:

```gitignore
# GE Price History Database - TOO LARGE FOR GIT (2.1GB+)
# Contains 13.8M historical price records from 2015-present
# Database includes complete price history and pre-computed analytics
# with moving averages (7d, 14d, 30d, 50d, 100d, 200d)
data/price_history.db
data/price_history.db-journal
```

**To set up on a new machine**:
1. Clone the repository
2. Run `python3 scripts/seed_complete_ge_history.py` to rebuild the database
3. Start the daemon with `python3 scripts/ge_update_daemon.py`

### API Compliance

The system respects Weird Gloop API guidelines:
- Proper User-Agent with contact info
- Conservative rate limiting (10 requests/second)
- 1-second sleep between batches
- Uses bulk endpoints where available

### Automatic Repair

The daemon automatically detects and repairs:
- Missing analytics entries
- Incomplete analytics (missing MA columns)
- Schema issues
- Duplicate entries

---

## 📊 Verified Examples

### Dragon bones (popular item)
- Historical Data: 3,761 records from 2015-03-27 to 2025-10-05
- MA 7d: 2,498.11 gp
- MA 200d: 2,509.64 gp
- Volatility (7d): 13.33

### Twisted bow (high-value item)
- MA 7d: 1,528,254,522 gp
- MA 200d: 1,522,006,156 gp
- Volatility (7d): 4,485,718

### Shark (common consumable)
- MA 7d: 831 gp
- MA 30d: 796 gp
- Volatility (7d): 59.71

---

## ✅ System Status

- ✅ Database verified and operational
- ✅ All analytics computed (99.4% complete)
- ✅ Daemon tested and working
- ✅ API compliance verified
- ✅ Logging implemented
- ✅ Auto-repair functional
- ✅ Documentation complete

**The system is ready for production use!** 🚀

