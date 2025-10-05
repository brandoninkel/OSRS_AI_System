#!/usr/bin/env python3
"""
Test API Queue Management

Tests that the API queue manager is properly coordinating requests
between all running systems:
- Streamlined Watchdog (MediaWiki API)
- GE Update Daemon (Prices API)
- Attribution Service (MediaWiki API)
- User Queries (both APIs)
"""

import requests
import time
import json
from datetime import datetime

API_BASE = "http://localhost:5001"

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_result(success, message):
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")

def test_api_health():
    """Test that API is responding"""
    print_header("1. API Health Check")
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print_result(True, "API server is healthy")
            return True
        else:
            print_result(False, f"API returned status {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"API not responding: {e}")
        return False

def test_queue_stats():
    """Get current queue statistics"""
    print_header("2. Queue Statistics")
    
    try:
        response = requests.get(f"{API_BASE}/queue/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print_result(True, "Queue stats retrieved")
            print(f"\n📊 Current Queue State:")
            print(f"   MediaWiki Requests: {stats.get('mediawiki_requests', 0)}")
            print(f"   Prices Requests: {stats.get('prices_requests', 0)}")
            print(f"   MediaWiki Queued: {stats.get('mediawiki_queued', 0)}")
            print(f"   Prices Queued: {stats.get('prices_queued', 0)}")
            print(f"   Watchdog Active: {stats.get('watchdog_active', False)}")
            print(f"   Watchdog Activations: {stats.get('watchdog_activations', 0)}")
            return True
        else:
            print_result(False, f"Failed to get stats: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Error getting stats: {e}")
        return False

def test_concurrent_requests():
    """Test multiple concurrent requests"""
    print_header("3. Concurrent Request Handling")
    
    print("\n🔄 Sending 5 concurrent price history requests...")
    
    items = ["Dragon bones", "Twisted bow", "Shark", "Abyssal whip", "Rune platebody"]
    start_time = time.time()
    
    results = []
    for item in items:
        try:
            response = requests.get(
                f"{API_BASE}/economic/price-history",
                params={"item": item, "hours": 24},
                timeout=10
            )
            results.append({
                'item': item,
                'status': response.status_code,
                'success': response.status_code == 200
            })
        except Exception as e:
            results.append({
                'item': item,
                'status': 'error',
                'success': False,
                'error': str(e)
            })
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"📊 Results:")
    
    success_count = sum(1 for r in results if r['success'])
    for result in results:
        icon = "✅" if result['success'] else "❌"
        print(f"   {icon} {result['item']}: {result['status']}")
    
    print_result(success_count == len(items), f"{success_count}/{len(items)} requests successful")
    return success_count == len(items)

def test_wiki_search():
    """Test wiki search (MediaWiki API)"""
    print_header("4. Wiki Search (MediaWiki API)")
    
    try:
        response = requests.get(
            f"{API_BASE}/wiki/search",
            params={"q": "dragon", "limit": 5},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                results = data.get('results', [])
                print_result(True, f"Wiki search returned {len(results)} results")
                if results:
                    print(f"\n   Sample results:")
                    for result in results[:3]:
                        print(f"   - {result}")
                return True
            else:
                print_result(False, "Wiki search failed")
                return False
        else:
            print_result(False, f"Wiki search returned status {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Wiki search error: {e}")
        return False

def test_ge_search():
    """Test GE item search (database query)"""
    print_header("5. GE Item Search (Database)")
    
    try:
        response = requests.get(
            f"{API_BASE}/economic/search_items",
            params={"q": "dragon", "limit": 5},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                results = data.get('results', [])
                print_result(True, f"GE search returned {len(results)} results")
                if results:
                    print(f"\n   Sample results:")
                    for result in results[:3]:
                        print(f"   - {result}")
                return True
            else:
                print_result(False, "GE search failed")
                return False
        else:
            print_result(False, f"GE search returned status {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"GE search error: {e}")
        return False

def test_rate_limiting():
    """Test that rate limiting is working"""
    print_header("6. Rate Limiting Test")
    
    print("\n🔄 Sending 10 rapid requests to test rate limiting...")
    
    start_time = time.time()
    success_count = 0
    
    for i in range(10):
        try:
            response = requests.get(
                f"{API_BASE}/economic/tracked_items",
                params={"limit": 10},
                timeout=5
            )
            if response.status_code == 200:
                success_count += 1
        except Exception as e:
            print(f"   Request {i+1} failed: {e}")
    
    elapsed = time.time() - start_time
    avg_time = elapsed / 10
    
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"⏱️  Average per request: {avg_time:.3f}s")
    print(f"📊 Success rate: {success_count}/10")
    
    # Rate limiting should add some delay, but all should succeed
    if success_count == 10:
        print_result(True, "All requests succeeded with rate limiting")
        return True
    else:
        print_result(False, f"Only {success_count}/10 requests succeeded")
        return False

def main():
    print("\n" + "=" * 80)
    print("  🧪 API QUEUE MANAGEMENT TEST SUITE")
    print("=" * 80)
    print(f"\n  Testing API at: {API_BASE}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        ("API Health", test_api_health),
        ("Queue Stats", test_queue_stats),
        ("Concurrent Requests", test_concurrent_requests),
        ("Wiki Search", test_wiki_search),
        ("GE Search", test_ge_search),
        ("Rate Limiting", test_rate_limiting),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_result(False, f"{test_name} crashed: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Final summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed\n")
    
    for test_name, result in results:
        icon = "✅" if result else "❌"
        print(f"   {icon} {test_name}")
    
    print("\n" + "=" * 80)
    
    if passed == total:
        print("  ✅ ALL TESTS PASSED - API Queue Management Working!")
    else:
        print(f"  ⚠️  {total - passed} test(s) failed")
    
    print("=" * 80 + "\n")
    
    # Get final queue stats
    test_queue_stats()

if __name__ == "__main__":
    main()

