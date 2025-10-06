#!/usr/bin/env python3
"""
Quick script to check OSRS wiki activity rate
"""
import requests
from datetime import datetime

response = requests.get(
    'https://oldschool.runescape.wiki/api.php',
    params={
        'action': 'query',
        'list': 'recentchanges',
        'rcnamespace': '0|4|12',  # Main, OSRS, Guide
        'rctype': 'edit|new',
        'rclimit': 500,
        'format': 'json'
    },
    headers={'User-Agent': 'OSRS-AI-RAG-System/1.0 (brandoninkel@gmail.com)'}
)

data = response.json()
changes = data['query']['recentchanges']

print(f"Total changes fetched: {len(changes)}")

if len(changes) > 1:
    timestamps = [datetime.fromisoformat(c['timestamp'].replace('Z', '+00:00')) for c in changes]
    duration_minutes = (timestamps[0] - timestamps[-1]).total_seconds() / 60
    rate = len(changes) / duration_minutes
    
    print(f"Time span: {duration_minutes:.1f} minutes")
    print(f"Rate: {rate:.2f} changes/minute")
    print(f"\nIn 10 minutes: ~{rate * 10:.0f} changes")
    print(f"In 1 hour: ~{rate * 60:.0f} changes")
    
    if rate * 10 > 500:
        print(f"\n⚠️  WARNING: At this rate, 10 minutes = {rate * 10:.0f} changes (>500 limit!)")
        print("   Continuation IS necessary!")
    else:
        print(f"\n✅ OK: At this rate, 10 minutes = {rate * 10:.0f} changes (<500 limit)")
        print("   Continuation probably not necessary for normal operation")

