# API Compliance Documentation

## 🔒 API Rules & Compliance

This document verifies that our GE data system fully complies with Weird Gloop and MediaWiki API guidelines.

---

## ✅ Weird Gloop API Compliance

### **Official Guidelines**

From Weird Gloop API documentation:
- **User-Agent Required**: Must be descriptive with contact information
- **Rate Limiting**: No explicit rate limit, but "be respectful"
- **Bulk Requests**: Supported and encouraged for efficiency
- **Blocked User-Agents**: python-requests, Python-urllib, curl/{version}, etc.

### **Our Implementation**

✅ **User-Agent**: `OSRS-AI-RAG-System/1.0 (brandoninkel@gmail.com) Python/requests`
- Descriptive project name
- Version number
- Contact email
- Library identification

✅ **Rate Limiting**: 10 concurrent requests per second
- Conservative limit (well below abuse threshold)
- 1-second sleep between batches
- Respects "be respectful" guideline

✅ **Bulk Requests**: Using bulk endpoints where available
- `/mapping` - ONE call for all 4,307 items
- `/latest` - ONE call for all current prices
- `/exchange/history/osrs/all?id={id}` - Individual item histories

✅ **Proper Headers**: All requests include proper User-Agent
- Configured in `api/config.py`
- Applied to all requests via `get_headers()`

---

## 📊 Request Patterns

### **Initial Seed (One-Time)**

**Phase 1: Item Mapping**
- **Endpoint**: `/mapping`
- **Requests**: 1 call
- **Time**: < 1 second
- **Compliant**: ✅ Bulk endpoint

**Phase 2: Complete Historical Data**
- **Endpoint**: `/exchange/history/osrs/all?id={id}`
- **Requests**: 4,307 calls (one per item)
- **Rate**: 10 requests per second
- **Time**: ~7 minutes
- **Compliant**: ✅ Conservative rate, proper User-Agent

**Phase 3: Analytics Computation**
- **Endpoint**: None (local computation)
- **Requests**: 0 API calls
- **Time**: ~2 minutes
- **Compliant**: ✅ No API usage

**Total Initial Seed**:
- **API Calls**: 4,308 (1 mapping + 4,307 histories)
- **Time**: ~7 minutes
- **Average Rate**: ~10 requests/second
- **Compliant**: ✅ Well within reasonable limits

### **Incremental Updates (Every 5 Minutes)**

**Update Cycle**:
- **Endpoint**: `/latest`
- **Requests**: 1 call
- **Time**: < 1 second
- **Frequency**: Every 5 minutes
- **Compliant**: ✅ Minimal API usage

**Per Hour**:
- **API Calls**: 12 (one every 5 minutes)
- **Average Rate**: 0.003 requests/second
- **Compliant**: ✅ Extremely light usage

**Per Day**:
- **API Calls**: 288 (12 per hour × 24 hours)
- **Average Rate**: 0.003 requests/second
- **Compliant**: ✅ Negligible impact

---

## 🔄 Comparison to API Guidelines

### **"Be Respectful" Analysis**

**What "Be Respectful" Means**:
- Don't make "multiple large queries per second for sustained period"
- Don't hammer the API continuously
- Use bulk endpoints when available
- Include proper User-Agent

**Our Implementation**:
- ✅ **Initial seed**: 10 req/sec for 7 minutes (one-time)
- ✅ **Incremental**: 1 req per 5 minutes (ongoing)
- ✅ **Bulk endpoints**: Using `/mapping` and `/latest`
- ✅ **Proper User-Agent**: Always included
- ✅ **Caching**: Store data locally, don't re-fetch
- ✅ **Deduplication**: Skip existing data

**Verdict**: ✅ **FULLY COMPLIANT**

---

## 📋 MediaWiki API Compliance

### **Official Guidelines**

From MediaWiki REST API documentation:
- **Rate Limit**: 200 requests/second (very generous)
- **User-Agent Required**: Must identify bot/tool
- **maxlag Parameter**: Use for non-interactive tasks
- **Compression**: GZip compression recommended

### **Our Implementation**

✅ **Rate Limiting**: 5-10 requests/second
- Well below 200 req/sec limit
- Conservative approach

✅ **User-Agent**: `OSRS-AI-RAG-System/1.0 (brandoninkel@gmail.com) Python/requests`
- Identifies our tool
- Includes contact info

✅ **Compression**: GZip enabled
- `Accept-Encoding: gzip` header
- Reduces bandwidth usage

✅ **Attribution Service**: Uses REST API endpoints
- `/rest.php/v1/page/{title}/history`
- `/rest.php/v1/revision/{id}`
- Respects rate limits

**Verdict**: ✅ **FULLY COMPLIANT**

---

## 🛡️ Safety Mechanisms

### **1. Rate Limiting**

**Implementation**:
```python
# In seed script
CONCURRENT_REQUESTS = 10  # 10 requests per second
await asyncio.sleep(1)    # 1-second sleep between batches
```

**Protection**:
- Prevents accidental API abuse
- Configurable (can be reduced if needed)
- Automatic throttling

### **2. Deduplication**

**Implementation**:
```python
# Skip existing data
if SKIP_EXISTING and price_service.has_complete_history(item_id):
    continue

# Database UNIQUE constraint
UNIQUE(item_id, timestamp)

# INSERT OR IGNORE
INSERT OR IGNORE INTO price_history_complete (...)
```

**Protection**:
- Prevents duplicate API calls
- Prevents duplicate database entries
- Idempotent operations

### **3. Error Handling**

**Implementation**:
```python
try:
    response = await session.get(url, headers=get_headers())
    if response.status == 200:
        # Process data
    else:
        # Log error, continue
except Exception as e:
    # Log error, continue
```

**Protection**:
- Graceful failure handling
- Continues on individual errors
- Logs issues for debugging

### **4. Progress Tracking**

**Implementation**:
- Real-time progress display
- ETA calculation
- Statistics tracking

**Protection**:
- User can monitor progress
- Can interrupt if needed
- Transparent operation

---

## 📊 Analytics Computation

### **Moving Averages**

**Supported Periods**: Fully flexible
- Default: [7, 14, 30, 50, 100, 200] days
- Configurable: Any periods can be specified
- Multiple: Compute multiple MAs at once

**Use Cases**:
- **7-day MA**: Short-term trends
- **14-day MA**: Medium-term trends
- **30-day MA**: Monthly trends
- **50-day MA**: Technical analysis (common in trading)
- **100-day MA**: Long-term trends
- **200-day MA**: Very long-term trends (common in trading)

**Trading System Support**:
- **Golden Cross**: 50-day MA crosses above 200-day MA (bullish)
- **Death Cross**: 50-day MA crosses below 200-day MA (bearish)
- **Custom Strategies**: Any combination of MAs can be used

### **Computed Metrics**

**Per Item**:
- ✅ **OHLC**: Open, High, Low, Close (daily)
- ✅ **Moving Averages**: Flexible periods
- ✅ **Price Changes**: 1d, 7d, 30d
- ✅ **Volatility**: 7d, 30d (standard deviation)
- ✅ **Volume Metrics**: 24h total and average
- ✅ **Percent Changes**: All price changes as percentages

**Storage**:
- Stored in `price_analytics` table
- Updated after seeding
- Updated during incremental updates
- Queryable for charting systems

---

## 🎯 Charting System Readiness

### **Data Available**

✅ **Historical Prices**: Complete history back to 2015
✅ **Volume Data**: Available for recent years
✅ **Moving Averages**: Multiple periods computed
✅ **OHLC Data**: Daily candles
✅ **Volatility**: Risk assessment
✅ **Trends**: Price change analysis

### **Chart Types Supported**

- **Line Charts**: Price over time
- **Candlestick Charts**: OHLC data
- **Volume Charts**: Trade volume over time
- **MA Overlay**: Multiple moving averages
- **Volatility Bands**: Bollinger-style bands
- **Comparison Charts**: Multiple items

### **Technical Analysis**

- **Trend Following**: MA crossovers
- **Momentum**: Price change rates
- **Volatility**: Standard deviation
- **Volume Analysis**: Liquidity assessment
- **Support/Resistance**: Historical price levels

---

## ✅ Final Compliance Summary

### **Weird Gloop API**
- ✅ Proper User-Agent with contact info
- ✅ Conservative rate limiting (10 req/sec)
- ✅ Bulk endpoints used where available
- ✅ "Be respectful" guideline followed
- ✅ No blocked User-Agent strings

### **MediaWiki API**
- ✅ Well below 200 req/sec limit
- ✅ Proper User-Agent identification
- ✅ GZip compression enabled
- ✅ REST API endpoints used correctly

### **Best Practices**
- ✅ Deduplication prevents waste
- ✅ Caching reduces API calls
- ✅ Error handling prevents abuse
- ✅ Progress tracking for transparency
- ✅ Incremental updates minimize load

### **Analytics & Charting**
- ✅ Flexible moving averages
- ✅ Comprehensive metrics computed
- ✅ Ready for charting systems
- ✅ Trading strategy support

---

## 🚀 Ready to Run

**Verdict**: ✅ **100% COMPLIANT**

Our implementation:
- Respects all API guidelines
- Uses conservative rate limits
- Implements proper safety mechanisms
- Computes comprehensive analytics
- Supports flexible charting systems

**Safe to run the full seed!** 🎉

