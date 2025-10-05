# Complete GE Data Analysis & Implementation Plan

## 🔍 API Limitations Discovery

### **Why Only 365 Data Points?**

The `/timeseries` endpoint has a **HARD LIMIT of 365 data points** per request.

**This is NOT a time limitation - it's a data point limitation:**
- `timestep=5m` → 365 points = ~30 hours of history
- `timestep=1h` → 365 points = ~15 days of history  
- `timestep=6h` → 365 points = ~91 days of history
- `timestep=24h` → 365 points = **~1 year of history**

**To get data back to item inception, we need multiple requests per item with different time ranges.**

However, the API does NOT support:
- ❌ `start_time` parameter
- ❌ `end_time` parameter
- ❌ Pagination
- ❌ Offset parameter

**Conclusion:** We can only get the **most recent 365 data points** for each timestep. We CANNOT fetch historical data beyond this window through the API.

---

## 📊 Complete Data Inventory

### **What We CAN Get:**

#### **1. Latest Prices (`/latest`)**
```json
{
  "itemId": {
    "high": 205,           // Instant-sell price (GP)
    "highTime": 1759666205, // Unix timestamp
    "low": 200,            // Instant-buy price (GP)
    "lowTime": 1759666176  // Unix timestamp
  }
}
```
- ✅ All ~3700 items in ONE API call
- ✅ Real-time instant buy/sell prices
- ✅ Timestamps for each price

#### **2. Item Mapping (`/mapping`)**
```json
{
  "id": 4151,
  "name": "Abyssal whip",
  "examine": "A weapon from the abyss.",
  "members": true,
  "lowalch": 72000,
  "highalch": 108000,
  "limit": 70,           // GE buy limit (4 hours)
  "value": 120001,       // Store value
  "icon": "Abyssal whip.png"
}
```
- ✅ All ~3700 items in ONE API call
- ✅ Item metadata (name, examine, alch values, GE limits)

#### **3. Timeseries Data (`/timeseries`)**

**For EACH timestep (5m, 1h, 6h, 24h):**
```json
{
  "timestamp": 1728086400,
  "avgHighPrice": 228,
  "avgLowPrice": 223,
  "highPriceVolume": 25454534,  // ⭐ TRADE VOLUME!
  "lowPriceVolume": 9400966     // ⭐ TRADE VOLUME!
}
```
- ✅ Average prices (not instant prices)
- ✅ **TRADE VOLUME** (number of transactions)
- ✅ 365 data points per item per timestep
- ✅ Requires 1 API call per item per timestep

#### **4. 5-Minute Averages (`/5m`)**
```json
{
  "timestamp": 1759666200,
  "data": {
    "itemId": {
      "avgHighPrice": 205,
      "avgLowPrice": 200,
      "highPriceVolume": 12345,
      "lowPriceVolume": 6789
    }
  }
}
```
- ✅ All items in ONE API call
- ✅ 5-minute average prices + volume
- ✅ Can specify `timestamp` parameter for historical 5m data

#### **5. 1-Hour Averages (`/1h`)**
- Same format as `/5m` but hourly averages
- ✅ All items in ONE API call
- ✅ Can specify `timestamp` parameter

---

## 💾 Complete Database Schema

### **Table 1: `price_history_latest`** (Current Prices)
```sql
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
);
```

### **Table 2: `price_history_timeseries`** (Historical Data)
```sql
CREATE TABLE IF NOT EXISTS price_history_timeseries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    timestep TEXT NOT NULL,  -- '5m', '1h', '6h', '24h'
    avg_high_price INTEGER,
    avg_low_price INTEGER,
    high_price_volume INTEGER,
    low_price_volume INTEGER,
    
    -- Calculated fields (computed on insert)
    margin INTEGER,              -- avg_high_price - avg_low_price
    margin_percent REAL,         -- (margin / avg_low_price) * 100
    total_volume INTEGER,        -- high_price_volume + low_price_volume
    price_midpoint INTEGER,      -- (avg_high_price + avg_low_price) / 2
    
    created_at TEXT NOT NULL,
    UNIQUE(item_id, timestamp, timestep)
);

CREATE INDEX idx_timeseries_item_time ON price_history_timeseries(item_id, timestamp DESC);
CREATE INDEX idx_timeseries_timestep ON price_history_timeseries(timestep, timestamp DESC);
CREATE INDEX idx_timeseries_volume ON price_history_timeseries(total_volume DESC);
```

### **Table 3: `item_metadata`** (Item Information)
```sql
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
);
```

### **Table 4: `price_analytics`** (Pre-computed Analytics)
```sql
CREATE TABLE IF NOT EXISTS price_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    
    -- Volatility metrics (24h)
    volatility_24h REAL,         -- Standard deviation of prices
    price_change_24h INTEGER,    -- Price change in GP
    price_change_24h_percent REAL,
    
    -- Volume metrics (24h)
    avg_volume_24h INTEGER,
    total_volume_24h INTEGER,
    
    -- Liquidity score (0-100)
    liquidity_score REAL,
    
    -- Margin metrics (24h)
    avg_margin_24h INTEGER,
    avg_margin_percent_24h REAL,
    
    -- Trend indicators
    trend_7d TEXT,               -- 'up', 'down', 'stable'
    trend_30d TEXT,
    
    -- Moving averages
    ma_7d INTEGER,               -- 7-day moving average
    ma_30d INTEGER,              -- 30-day moving average
    
    -- Risk/reward classification
    risk_category TEXT,          -- 'low', 'medium', 'high'
    flip_score REAL,             -- 0-100 score for flipping potential
    
    calculated_at TEXT NOT NULL,
    UNIQUE(item_id, calculated_at)
);
```

---

## 🎯 Calculated Metrics

### **Metrics We Can Calculate:**

1. **Margin Analysis**
   - `margin = avg_high_price - avg_low_price`
   - `margin_percent = (margin / avg_low_price) * 100`
   - `roi = (margin / avg_low_price) * 100`

2. **Volume Analysis**
   - `total_volume = high_price_volume + low_price_volume`
   - `volume_ratio = high_price_volume / low_price_volume`
   - `avg_volume_24h = SUM(volume) / 24`

3. **Volatility**
   - `price_std_dev = STDDEV(prices over time)`
   - `volatility_percent = (std_dev / mean_price) * 100`
   - `price_range = MAX(price) - MIN(price)`

4. **Liquidity Score** (0-100)
   ```python
   liquidity_score = (
       volume_weight * 0.5 +
       margin_stability_weight * 0.3 +
       price_stability_weight * 0.2
   ) * 100
   ```

5. **Trend Analysis**
   - `price_change_7d = current_price - price_7_days_ago`
   - `trend = 'up' if price_change > threshold else 'down'`
   - Moving averages (7d, 30d, 90d)

6. **Flip Score** (0-100)
   ```python
   flip_score = (
       margin_weight * 0.4 +
       volume_weight * 0.3 +
       stability_weight * 0.2 +
       liquidity_weight * 0.1
   ) * 100
   ```

7. **Risk Classification**
   - Low risk: High volume, low volatility, stable margins
   - Medium risk: Moderate volume/volatility
   - High risk: Low volume, high volatility, unstable margins

---

## 🚀 Implementation Strategy

### **Phase 1: Bulk Historical Seed (ALL Timesteps)**

**Goal:** Get maximum historical data for all items

**Strategy:**
```python
for each item (~3700 items):
    for each timestep in ['5m', '1h', '6h', '24h']:
        fetch_timeseries(item_id, timestep)
        # Gets 365 data points per timestep
        record_to_database()
        rate_limit_delay()
```

**Data Retrieved:**
- `5m`: 365 points × 3700 items = 1,350,500 records (~30 hours history)
- `1h`: 365 points × 3700 items = 1,350,500 records (~15 days history)
- `6h`: 365 points × 3700 items = 1,350,500 records (~91 days history)
- `24h`: 365 points × 3700 items = 1,350,500 records (~1 year history)
- **TOTAL: 5,402,000 records**

**Time Estimate:**
- 3700 items × 4 timesteps = 14,800 API calls
- At 5 req/sec = 2,960 seconds = **~50 minutes**
- At 10 req/sec = 1,480 seconds = **~25 minutes**

**Storage:**
- ~5.4M records × ~100 bytes/record = **~540MB**

### **Phase 2: Item Metadata Seed**

```python
fetch_item_mapping()  # ONE API call
store_in_item_metadata_table()
```

### **Phase 3: Calculate Analytics**

```python
for each item:
    calculate_volatility()
    calculate_liquidity_score()
    calculate_flip_score()
    calculate_trends()
    store_in_price_analytics_table()
```

### **Phase 4: Incremental Updates (Ongoing)**

**Every 5 minutes:**
```python
fetch_5m_averages()  # ONE API call for all items
update_timeseries_table()
recalculate_analytics_for_changed_items()
```

**Every hour:**
```python
fetch_1h_averages()  # ONE API call
update_timeseries_table()
```

**Daily:**
```python
fetch_24h_timeseries_for_all_items()  # 3700 API calls
update_timeseries_table()
recalculate_all_analytics()
```

---

## 📋 Missing Data & Workarounds

### **What We CANNOT Get:**

1. ❌ **Historical data beyond 365 points per timestep**
   - Workaround: Start collecting now, build history over time

2. ❌ **Completed trade prices** (only offers, not actual trades)
   - Workaround: Use average prices as proxy

3. ❌ **Order book depth** (how many offers at each price)
   - Workaround: Use volume as liquidity indicator

4. ❌ **Individual trade data**
   - Workaround: Use aggregated volume data

5. ❌ **Player inventory/wealth data**
   - Workaround: N/A - not available

### **What We CAN Infer:**

1. ✅ **Market manipulation** - Sudden volume/price spikes
2. ✅ **Bot activity** - Unusual volume patterns
3. ✅ **Update impacts** - Price changes after game updates
4. ✅ **Seasonal patterns** - Weekly/monthly trends
5. ✅ **Item popularity** - Volume trends over time

---

## 🎯 AI Analysis Capabilities

With this data, AI can answer:

1. **Price Prediction**
   - "Will Dragon bones go up next week?"
   - Uses: Historical trends, volume patterns, moving averages

2. **Flip Recommendations**
   - "What are the best items to flip right now?"
   - Uses: Margin, volume, liquidity score, flip score

3. **Risk Assessment**
   - "Is this item safe to invest in?"
   - Uses: Volatility, volume stability, price trends

4. **Market Analysis**
   - "How did the new update affect item prices?"
   - Uses: Price changes around specific dates

5. **Volume Analysis**
   - "Which items have the highest trading volume?"
   - Uses: Volume data across all timesteps

6. **Trend Detection**
   - "Show me items trending up this week"
   - Uses: Price changes, moving averages

7. **Correlation Analysis**
   - "What items move together in price?"
   - Uses: Price correlation across items

8. **Liquidity Scoring**
   - "Which items can I flip quickly?"
   - Uses: Volume, margin stability

---

## ⚡ Next Steps

1. ✅ Update database schema (add all tables)
2. ✅ Implement timeseries fetching (all timesteps)
3. ✅ Implement calculated metrics
4. ✅ Create bulk historical seed script
5. ✅ Add progress tracking and ETA
6. ✅ Implement incremental updater
7. ✅ Create analytics calculator
8. ✅ Add AI query tools

**Ready to implement?** This will give your AI comprehensive economic analysis capabilities with ~5.4M historical data points! 🚀

