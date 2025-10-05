# Complete GE Historical Data Seed - User Guide

## 🎯 Overview

This system fetches **complete historical price data** for ALL OSRS items from the Weird Gloop API and stores it in your local database.

---

## ✅ What You Get

- **Complete history** back to March 2015 (item inception)
- **~14.8 million data points** across ~4,000 items
- **Volume data** for recent years (2020+)
- **Price data** for all years (2015-present)
- **Fast retrieval** (~7 minutes with parallel processing)
- **No duplicates** (automatic deduplication)
- **~1.23GB storage** (very reasonable!)

---

## 🚀 Quick Start

### **Option 1: Command Line**
```bash
python3 scripts/seed_complete_ge_history.py
```

### **Option 2: macOS Double-Click**
Double-click: `scripts/seed-complete-ge-history.command`

---

## 📊 What Happens During Seed

### **Phase 1: Item Mapping (5 seconds)**
- Fetches all 4,307 items from `/mapping` endpoint
- Stores item metadata (name, examine, alch values, GE limits)

### **Phase 2: Complete Historical Data (~7 minutes)**
- Fetches complete history for each item
- Processes ~14.8M data points
- Handles both old (no volume) and new (with volume) formats
- Shows real-time progress with ETA

### **Phase 3: Verification**
- Displays final statistics
- Shows total records inserted
- Estimates storage used

---

## 📈 Progress Display

During the seed, you'll see:
```
📊 Progress: 1234/4307 (28.6%) | Records: 4,567,890 | With data: 1200 | No data: 34 | ETA: 0:05:23
```

**What it means:**
- **Progress:** Items processed / Total items
- **Records:** Total data points inserted
- **With data:** Items that have price history
- **No data:** Items without price history (untradeable)
- **ETA:** Estimated time remaining

---

## 🔧 Configuration

Edit `scripts/seed_complete_ge_history.py` to customize:

```python
CONCURRENT_REQUESTS = 10  # Requests per second (10 is safe)
BATCH_SIZE = 1000         # Records per database insert
SKIP_EXISTING = True      # Skip items that already have data
```

---

## 💾 Database Schema

### **Table: `price_history_complete`**
Stores all historical data points:
- `item_id` - Item ID
- `item_name` - Item name
- `price` - Price in GP
- `volume` - Trade volume (NULL for pre-RuneLite data)
- `timestamp` - Unix timestamp in milliseconds
- `data_source` - 'jagex' or 'runelite'
- `has_volume` - TRUE if volume is not NULL

### **Table: `item_metadata`**
Stores item information:
- `item_id` - Item ID (primary key)
- `name` - Item name
- `examine` - Examine text
- `members` - Members only
- `lowalch` - Low alchemy value
- `highalch` - High alchemy value
- `buy_limit` - GE buy limit
- `value` - Store value
- `icon` - Icon filename

---

## 🔍 Querying the Data

### **Get all data for an item:**
```sql
SELECT * FROM price_history_complete
WHERE item_id = 4151
ORDER BY timestamp ASC;
```

### **Get recent data with volume:**
```sql
SELECT * FROM price_history_complete
WHERE item_id = 4151
  AND volume IS NOT NULL
ORDER BY timestamp DESC
LIMIT 100;
```

### **Get price range for an item:**
```sql
SELECT 
    item_name,
    MIN(price) as lowest_price,
    MAX(price) as highest_price,
    AVG(price) as avg_price,
    COUNT(*) as data_points
FROM price_history_complete
WHERE item_id = 4151;
```

### **Get items with most data points:**
```sql
SELECT 
    item_name,
    COUNT(*) as data_points,
    MIN(timestamp) as first_date,
    MAX(timestamp) as last_date
FROM price_history_complete
GROUP BY item_id
ORDER BY data_points DESC
LIMIT 10;
```

---

## 🔄 Re-running the Seed

### **Skip Existing Data (Fast)**
Set `SKIP_EXISTING = True` in the script. This will:
- Skip items that already have data
- Only fetch new items
- Complete in seconds if most data exists

### **Full Re-seed (Slow)**
Set `SKIP_EXISTING = False` in the script. This will:
- Fetch all items again
- Use `INSERT OR IGNORE` to prevent duplicates
- Take ~7 minutes

---

## 🧪 Testing Before Full Seed

Run the test script first:
```bash
python3 scripts/test_complete_history.py
```

This will:
- Test with 3 items (Cannonball, Abyssal whip, 3rd age amulet)
- Verify database storage
- Show sample data
- Complete in ~5 seconds

---

## ⚠️ Troubleshooting

### **Error: "Failed to fetch mapping"**
- Check internet connection
- Verify API is accessible: `curl https://prices.runescape.wiki/api/v1/osrs/mapping`

### **Error: "Database locked"**
- Close any other programs accessing the database
- Wait a few seconds and try again

### **Slow performance**
- Reduce `CONCURRENT_REQUESTS` to 5
- Check internet speed
- Verify no rate limiting from API

### **Interrupted seed**
- Progress is saved automatically
- Re-run with `SKIP_EXISTING = True` to continue
- Already-inserted data won't be duplicated

---

## 📊 Expected Results

After completion, you should see:
```
✅ Complete!
⏱️  Total time: 0:07:23
📦 Total items processed: 4307
✅ Items with data: 3847
❌ Items without data: 460
📊 Total records inserted: 14,234,567
💾 Estimated storage: 1.18 GB
```

---

## 🎯 What This Enables

Your AI can now:
- ✅ Analyze 10 years of price history
- ✅ Calculate liquidity scores (volume data)
- ✅ Detect long-term trends
- ✅ Predict future prices (ML training data)
- ✅ Identify flip opportunities (margin analysis)
- ✅ Understand market context (game updates impact)
- ✅ Calculate moving averages (7d, 30d, 90d)
- ✅ Measure volatility (risk assessment)

---

## 📚 Related Documentation

- **API Research:** `docs/GE_API_COMPLETE_RESEARCH.md`
- **Database Schema:** `api/price_history.py`
- **Configuration:** `api/config.py`

---

## 🚀 Next Steps

After seeding:
1. **Verify data:** Run database queries to check data
2. **Set up incremental updates:** Use `/latest` endpoint for real-time updates
3. **Calculate analytics:** Pre-compute daily metrics
4. **Train AI models:** Use historical data for predictions

---

## 💡 Tips

- **Run overnight:** First seed takes ~7 minutes, but you can let it run unattended
- **Monitor progress:** Watch the progress bar to see ETA
- **Check logs:** Any errors will be displayed in real-time
- **Verify storage:** Ensure you have at least 2GB free space
- **Backup database:** Copy `data/price_history.db` after seeding

---

## 🎉 Success!

Once complete, you'll have the most comprehensive OSRS GE price database available, enabling powerful economic analysis and AI-driven insights!

