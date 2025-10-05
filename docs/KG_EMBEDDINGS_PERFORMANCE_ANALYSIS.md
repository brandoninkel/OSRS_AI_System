# KG Embeddings Performance Analysis

## System Specs
- **CPU**: M4 Pro with 12 cores (24 logical)
- **RAM**: 24GB
- **GPU**: Metal (Apple Silicon integrated)
- **Model**: mxbai-embed-large (334M params, F16 quantization)
- **Model Size in VRAM**: 1.2GB

## Current Performance (64 concurrency, individual requests)
- **Speed**: ~120 entities/second
- **Total time**: ~21 minutes for 149,047 entities
- **Ollama CPU usage**: 92.6%
- **Bottleneck**: Ollama model inference, not our code

## Batch API Performance (tested)
| Batch Size | Speed (embeddings/sec) |
|------------|------------------------|
| 50         | 124.4                  |
| **100**    | **141.4** ⭐ OPTIMAL   |
| 200        | 139.7                  |
| 500        | 135.4                  |
| 1000       | 129.7                  |

## Key Findings

### 1. Ollama is the bottleneck, not our Python code
- Ollama processes embeddings sequentially through a single model instance
- Even with 64 concurrent requests, they queue up and process one at a time
- The model is already using Metal GPU acceleration (loaded in VRAM)
- CPU usage at 92.6% indicates we're maxing out the model's inference capacity

### 2. Batch API provides marginal improvement
- Peak: 141/sec vs current 120/sec = **17.5% faster**
- Reduces HTTP overhead but doesn't change model inference speed
- Main benefits:
  * Cleaner code (single request instead of managing 64 concurrent requests)
  * Less network overhead
  * Better memory efficiency
  * Simpler error handling

### 3. Why we can't go faster
- **Ollama architecture**: Runs a single model instance per model
- **Model inference**: Sequential processing (can't parallelize within a single model)
- **Metal GPU**: Already being used (model loaded in VRAM)
- **Hardware limit**: M4 Pro is already very capable

### 4. To go faster, we would need to:
- ❌ **Run multiple Ollama instances** on different ports (complex setup, diminishing returns)
- ❌ **Use a faster model** (but less accurate, defeats the purpose)
- ❌ **Use a more powerful GPU** (but M4 Pro is already excellent for this task)
- ✅ **Use batch API** (17.5% speedup, already implemented)

## Optimal Configuration

### Recommended Settings:
```python
# Use batch API with optimal batch size
use_batch_api=True
batch_size=100  # Sweet spot for mxbai-embed-large

# No need for high concurrency with batch API
max_concurrency=4  # Just enough for retry/fallback scenarios
```

### Expected Performance:
- **Speed**: ~141 embeddings/second
- **Total time**: ~18 minutes for 149,047 entities
- **Improvement**: 14% faster than current (21 min → 18 min)

## Conclusion

**The system is already well-optimized.** The 120-141 embeddings/sec we're achieving is close to the hardware limit for this model on the M4 Pro.

The batch API implementation provides:
- ✅ 17.5% speedup (141 vs 120 embeddings/sec)
- ✅ Cleaner, simpler code
- ✅ Less HTTP overhead
- ✅ Better resource utilization

**Don't expect dramatic improvements** - we're already maxing out Ollama's throughput for this model. The bottleneck is the model's inference speed, not our code or system resources.

## Implementation Status

- ✅ Batch API implemented in `api/embeddings/embedding_service.py`
- ✅ Fallback to individual requests if batch fails
- ✅ Cache compatibility maintained
- ⏳ Integration with KG embeddings system (pending)
- ⏳ Testing with full 149k entity dataset (pending)

## Next Steps

1. Update `scripts/create_osrs_embeddings.py` to use batch API by default
2. Test with full KG entity dataset
3. Integrate into watchdog system
4. Monitor for any issues with large batches

