#!/usr/bin/env python3
"""
Populate Price History Database
Queries multiple items to build historical price data for trend analysis.
"""
import requests
import time
import json

API_BASE = "http://localhost:5001"

# Popular OSRS items to query
ITEMS_TO_QUERY = [
    "Abyssal whip",
    "Dragon scimitar",
    "Bandos chestplate",
    "Armadyl crossbow",
    "Twisted bow",
    "Scythe of vitur",
    "Dragon claws",
    "Ancestral robe top",
    "Elysian spirit shield",
    "Dragon warhammer",
    "Toxic blowpipe",
    "Trident of the swamp",
    "Amulet of fury",
    "Berserker ring",
    "Primordial boots",
    "Dragon boots",
    "Black d'hide body",
    "Rune platebody",
    "Shark",
    "Prayer potion(4)",
    "Super combat potion(4)",
    "Saradomin brew(4)",
    "Dragon bones",
    "Magic logs",
    "Runite ore"
]

def query_item_price(item_name):
    """Query an item price through the chat API"""
    print(f"📊 Querying: {item_name}...")
    
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={"query": f"How much is a {item_name}?"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', '')
            print(f"   ✅ Response: {answer[:100]}...")
            return True
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def check_price_history(item_name):
    """Check if price history exists for an item"""
    try:
        response = requests.get(
            f"{API_BASE}/economic/price-history",
            params={"item": item_name, "hours": 24},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                trend = data.get('trend', {})
                data_points = trend.get('data_points', 0)
                return data_points
        return 0
        
    except Exception as e:
        print(f"   ⚠️  Could not check history: {e}")
        return 0

def main():
    print("=" * 80)
    print("🚀 POPULATING PRICE HISTORY DATABASE")
    print("=" * 80)
    print()
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding!")
            return
        print("✅ API server is healthy")
        print()
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        return
    
    successful = 0
    failed = 0
    
    for i, item in enumerate(ITEMS_TO_QUERY, 1):
        print(f"\n[{i}/{len(ITEMS_TO_QUERY)}] {item}")
        print("-" * 80)
        
        # Check if we already have data
        existing_points = check_price_history(item)
        if existing_points > 0:
            print(f"   ℹ️  Already have {existing_points} data points, skipping...")
            continue
        
        # Query the item
        if query_item_price(item):
            successful += 1
            
            # Wait a bit between queries to avoid overwhelming the system
            if i < len(ITEMS_TO_QUERY):
                print("   ⏳ Waiting 3 seconds...")
                time.sleep(3)
        else:
            failed += 1
    
    print()
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"✅ Successful queries: {successful}")
    print(f"❌ Failed queries: {failed}")
    print(f"📈 Total items: {len(ITEMS_TO_QUERY)}")
    print()
    
    # Check database status
    print("🔍 Checking price history database...")
    items_with_data = 0
    for item in ITEMS_TO_QUERY:
        points = check_price_history(item)
        if points > 0:
            items_with_data += 1
            print(f"   ✅ {item}: {points} data points")
    
    print()
    print(f"📊 Items with price history: {items_with_data}/{len(ITEMS_TO_QUERY)}")
    print()
    print("✅ Price history population complete!")

if __name__ == "__main__":
    main()

