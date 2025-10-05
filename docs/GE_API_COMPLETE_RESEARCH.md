# Complete GE API Research & Implementation Plan

## 🎉 BREAKTHROUGH DISCOVERY!

### **The `/exchange/history/osrs/all` Endpoint**

**URL Format:** `https://api.weirdgloop.org/exchange/history/osrs/all?id={item_id}`

**Example:** `https://api.weirdgloop.org/exchange/history/osrs/all?id=4151` (Abyssal whip)

---

## 📊 What We Get - COMPLETE HISTORICAL DATA!

### **Data Format:**
```json
{
  "4151": [
    {
      "id": "4151",
      "price": 2057864,
      "volume": null,           // ⚠️ NULL for old data (pre-RuneLite)
      "timestamp": 1427500800000  // March 28, 2015
    },
    {
      "id": "4151",
      "price": 1500228,
      "volume": 7521,            // ✅ VOLUME DATA (post-RuneLite)
      "timestamp": 1720008033000  // July 3, 2024
    }
  ]
}
```

### **Key Findings:**

1. ✅ **COMPLETE HISTORY** - Data goes back to **item inception** (March 2015 for Abyssal whip)
2. ✅ **ALL DATA POINTS** - Not limited to 365 points! (Abyssal whip has 3,500+ data points)
3. ✅ **VOLUME DATA** - Available for recent data (post-RuneLite integration ~2020)
4. ⚠️ **Volume is NULL** for older data (pre-RuneLite era)
5. ✅ **Daily granularity** - One data point per day
6. ✅ **Timestamps in milliseconds** - Unix timestamp format

---

## 🔍 Data Eras

### **Era 1: Pre-RuneLite (2015-2020)**
- **Source:** Jagex official GE data
- **Data:** Price only (volume = null)
- **Frequency:** Daily
- **Example:** `{"price": 2057864, "volume": null, "timestamp": 1427500800000}`

### **Era 2: Post-RuneLite (2020-Present)**
- **Source:** RuneLite plugin crowdsourced data
- **Data:** Price + Volume
- **Frequency:** More frequent (multiple updates per day)
- **Example:** `{"price": 1500228, "volume": 7521, "timestamp": 1720008033000}`

---

## 📈 Item Count Analysis

### **How Many Items?**

Testing the `/mapping` endpoint:
```bash
curl "https://prices.runescape.wiki/api/v1/osrs/mapping"
```

**Result:** **4,307 items** in the mapping ✅

**But not all are tradeable!**
- Some items don't have GE data (untradeable items)
- Some items are player-to-player only (not on GE)
- Some items are discontinued

**Testing Results:**
- ✅ Cannonball (ID 2): **3,763 data points** (March 2015 - Present)
- ✅ Abyssal whip (ID 4151): **3,772 data points** (March 2015 - Present)
- ✅ 3rd age amulet (ID 10344): **3,773 data points** (March 2015 - Present)
- ❌ Unknown item (ID 1): **No data** (not tradeable)

**Estimate:** ~3,500-4,000 items with actual price history

---

## 🚀 Complete Data Retrieval Strategy

### **Phase 1: Item Mapping**
```
GET https://prices.runescape.wiki/api/v1/osrs/mapping
```
- **ONE API call** for all items
- Get item ID, name, examine, alch values, GE limits
- Store in `item_metadata` table

### **Phase 2: Complete Historical Data**
```
For each item (~4,307 items):
    GET https://api.weirdgloop.org/exchange/history/osrs/all?id={item_id}

    For each data point:
        - Store price
        - Store volume (if not null)
        - Store timestamp
        - Calculate derived metrics
```

**Data Retrieved:**
- **~4,000 items** with price history (estimate)
- **~3,700 data points per item** (average, varies by item age)
- **Total: ~14,800,000 data points** 🎉

**Time Estimate:**
- 4,307 API calls (one per item)
- At 5 req/sec = 861 seconds = **~14 minutes**
- At 10 req/sec = 431 seconds = **~7 minutes**

**Storage:**
- ~14.8M records × ~80 bytes/record = **~1.18GB**

### **Phase 3: Real-Time Updates**
```
Every 5 minutes:
    GET https://prices.runescape.wiki/api/v1/osrs/latest
    
    For each item with new data:
        - Check if timestamp > last_timestamp in DB
        - If yes, insert new record
        - Calculate updated analytics
```

---

## 💾 Database Schema (Updated)

### **Table: `price_history_complete`**
```sql
CREATE TABLE IF NOT EXISTS price_history_complete (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    price INTEGER NOT NULL,
    volume INTEGER,              -- NULL for pre-RuneLite data
    timestamp INTEGER NOT NULL,  -- Unix timestamp (milliseconds)
    
    -- Metadata
    data_source TEXT,            -- 'jagex' or 'runelite'
    has_volume BOOLEAN,          -- TRUE if volume is not null
    created_at TEXT NOT NULL,
    
    UNIQUE(item_id, timestamp)
);

CREATE INDEX idx_complete_item_time ON price_history_complete(item_id, timestamp DESC);
CREATE INDEX idx_complete_timestamp ON price_history_complete(timestamp DESC);
CREATE INDEX idx_complete_volume ON price_history_complete(volume DESC) WHERE volume IS NOT NULL;
```

### **Table: `price_analytics_daily`** (Pre-computed)
```sql
CREATE TABLE IF NOT EXISTS price_analytics_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    date TEXT NOT NULL,          -- YYYY-MM-DD
    
    -- Price metrics
    open_price INTEGER,
    close_price INTEGER,
    high_price INTEGER,
    low_price INTEGER,
    avg_price INTEGER,
    
    -- Volume metrics (if available)
    total_volume INTEGER,
    avg_volume INTEGER,
    
    -- Calculated metrics
    price_change INTEGER,
    price_change_percent REAL,
    volatility REAL,
    
    -- Moving averages
    ma_7d INTEGER,
    ma_30d INTEGER,
    ma_90d INTEGER,
    
    calculated_at TEXT NOT NULL,
    UNIQUE(item_id, date)
);
```

---

## 🎯 Deduplication Strategy

### **Preventing Duplicate API Calls:**

1. **Check Last Timestamp**
   ```sql
   SELECT MAX(timestamp) FROM price_history_complete WHERE item_id = ?
   ```

2. **Only Fetch If Needed**
   - If no data exists: Fetch complete history
   - If data exists: Only fetch `/latest` for updates

3. **Idempotent Inserts**
   ```sql
   INSERT OR IGNORE INTO price_history_complete (...)
   ```
   - UNIQUE constraint on (item_id, timestamp) prevents duplicates

### **Preventing Duplicate Database Entries:**

1. **UNIQUE Constraint**
   - `UNIQUE(item_id, timestamp)` ensures no duplicates

2. **INSERT OR IGNORE**
   - Silently skips if record already exists

3. **Batch Validation**
   - Before bulk insert, check if item already has data
   - Skip items that are already complete

---

## 📊 Data Format Differences

### **Old Format (Jagex Official)**
```json
{
  "id": "4151",
  "price": 2057864,
  "volume": null,
  "timestamp": 1427500800000
}
```
- Daily snapshots
- Price only
- Consistent timestamps (midnight UTC)

### **New Format (RuneLite)**
```json
{
  "id": "4151",
  "price": 1500228,
  "volume": 7521,
  "timestamp": 1720008033000
}
```
- Multiple updates per day
- Price + Volume
- Variable timestamps (real-time updates)

### **Handling Both Formats:**

```python
def process_data_point(item_id, data_point):
    price = data_point['price']
    volume = data_point.get('volume')  # May be null
    timestamp = data_point['timestamp']
    
    # Determine data source
    data_source = 'runelite' if volume is not None else 'jagex'
    has_volume = volume is not None
    
    # Store in database
    record_price_history(
        item_id=item_id,
        price=price,
        volume=volume,
        timestamp=timestamp,
        data_source=data_source,
        has_volume=has_volume
    )
```

---

## ⚡ Fast Retrieval Implementation

### **Parallel Processing:**

```python
import asyncio
import aiohttp

async def fetch_item_history(session, item_id):
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id={item_id}"
    async with session.get(url, headers=headers) as response:
        return await response.json()

async def bulk_fetch_all_items(item_ids):
    async with aiohttp.ClientSession() as session:
        # Process in batches of 10 concurrent requests
        for i in range(0, len(item_ids), 10):
            batch = item_ids[i:i+10]
            tasks = [fetch_item_history(session, item_id) for item_id in batch]
            results = await asyncio.gather(*tasks)
            
            # Process results
            for result in results:
                process_and_store(result)
            
            # Rate limiting: 10 requests per second
            await asyncio.sleep(1)
```

**Speed:**
- 10 concurrent requests per second
- 4,307 items / 10 = 431 batches
- 431 seconds = **~7 minutes total**

---

## 🔄 Incremental Update Strategy

### **After Initial Bulk Seed:**

**Every 5 minutes:**
```python
def incremental_update():
    # Get latest prices for all items (ONE API call)
    latest_data = fetch_latest_prices()
    
    for item_id, price_data in latest_data.items():
        # Check if this is new data
        last_timestamp = get_last_timestamp(item_id)
        
        if price_data['timestamp'] > last_timestamp:
            # New data! Insert it
            record_price_history(
                item_id=item_id,
                price=price_data['high'],  # or 'low'
                volume=None,  # /latest doesn't have volume
                timestamp=price_data['highTime']
            )
```

**For volume data:**
```python
# Fetch complete history for items with recent updates
# This gets the volume data that /latest doesn't provide
for item_id in recently_updated_items:
    history = fetch_item_history(item_id)
    
    # Only insert new data points
    for data_point in history:
        if data_point['timestamp'] > last_timestamp:
            record_price_history(...)
```

---

## 📋 Complete Implementation Checklist

### **Phase 1: Database Setup**
- [ ] Create `price_history_complete` table
- [ ] Create `item_metadata` table
- [ ] Create `price_analytics_daily` table
- [ ] Create indexes for fast queries

### **Phase 2: Bulk Historical Seed**
- [ ] Fetch item mapping (ONE API call) - **4,307 items**
- [ ] Store item metadata
- [ ] Fetch complete history for all items (**4,307 API calls**)
- [ ] Process and store all data points (**~14.8M records**)
- [ ] Handle both old (no volume) and new (with volume) formats
- [ ] Implement deduplication (UNIQUE constraint)
- [ ] Show progress bar with ETA
- [ ] Skip items with no price history (untradeable items)

### **Phase 3: Analytics Calculation**
- [ ] Calculate daily OHLC (Open, High, Low, Close)
- [ ] Calculate moving averages (7d, 30d, 90d)
- [ ] Calculate volatility
- [ ] Calculate volume metrics (where available)
- [ ] Store in `price_analytics_daily` table

### **Phase 4: Incremental Updates**
- [ ] Fetch `/latest` every 5 minutes
- [ ] Check for new data (timestamp > last_timestamp)
- [ ] Insert only new records
- [ ] Recalculate analytics for updated items

### **Phase 5: Validation**
- [ ] Verify no duplicate records
- [ ] Verify data integrity (prices reasonable)
- [ ] Verify timestamp ordering
- [ ] Verify volume data where expected

---

## 🎯 Final Summary

### **What We Can Get:**

✅ **Complete historical data** back to item inception (March 2015+)
✅ **~14.8 million data points** across ~4,000 items
✅ **Volume data** for recent years (post-RuneLite ~2020+)
✅ **Price data** for all years (2015-present)
✅ **Fast retrieval** (~7 minutes with parallel processing)
✅ **No duplicates** (UNIQUE constraints + INSERT OR IGNORE)
✅ **Incremental updates** (check timestamps, only insert new data)
✅ **Both data formats** (old Jagex + new RuneLite)
✅ **4,307 items** in mapping (not all have price history)

### **API Endpoints Used:**

1. **`/mapping`** - Item metadata (ONE call) - **4,307 items**
2. **`/exchange/history/osrs/all?id={id}`** - Complete history per item (**4,307 calls**)
3. **`/latest`** - Real-time updates (ONE call every 5 min)

### **Storage:**

- **~1.18GB** for complete historical data (~14.8M records)
- **~50MB** for pre-computed analytics
- **Total: ~1.23GB** (very reasonable!)

### **Time:**

- **Initial seed: ~7 minutes** (with parallel processing at 10 req/sec)
- **Incremental updates: <1 second** (every 5 minutes)

---

## 🚀 Ready to Implement!

This gives your AI:
- **10 years of price history**
- **Volume data for liquidity analysis**
- **Complete market context**
- **Trend analysis capabilities**
- **Prediction model training data**

**Should I create the complete bulk historical seed script?** 🎉

