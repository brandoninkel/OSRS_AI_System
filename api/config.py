#!/usr/bin/env python3
"""
Configuration for OSRS AI System

Centralized configuration for API endpoints, User-Agent headers, and rate limits.
Follows Weird Gloop and MediaWiki API guidelines.
"""

# User-Agent Configuration
# Per Weird Gloop requirements: Must be descriptive with contact info
# Format: "ProjectName/Version (contact) Library/Version"
USER_AGENT = "OSRS-AI-RAG-System/1.0 (brandoninkel@gmail.com) Python/requests"

# API Endpoints
OSRS_WIKI_API = "https://oldschool.runescape.wiki/api.php"
OSRS_WIKI_REST_API = "https://oldschool.runescape.wiki/rest.php/v1"
PRICES_API_BASE = "https://prices.runescape.wiki/api/v1/osrs"

# Rate Limits (requests per second)
# Weird Gloop: No explicit rate limit, but be respectful
# MediaWiki REST API: 200 requests/second (we use conservative limits)
PRICES_API_RATE_LIMIT = 10  # requests per second (conservative)
WIKI_API_RATE_LIMIT = 5     # requests per second (conservative)
WIKI_REST_API_RATE_LIMIT = 10  # requests per second (conservative)

# Request Headers
def get_headers(additional_headers=None):
    """
    Get standard headers for API requests
    
    Args:
        additional_headers: Optional dict of additional headers
        
    Returns:
        Dict of headers with User-Agent and optional additions
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip"
    }
    
    if additional_headers:
        headers.update(additional_headers)
    
    return headers

# GE Price API Endpoints
GE_ENDPOINTS = {
    "latest": f"{PRICES_API_BASE}/latest",           # Latest prices for all items
    "mapping": f"{PRICES_API_BASE}/mapping",         # Item ID to name mapping
    "5m": f"{PRICES_API_BASE}/5m",                   # 5-minute averages
    "1h": f"{PRICES_API_BASE}/1h",                   # 1-hour averages
    "timeseries": f"{PRICES_API_BASE}/timeseries"    # Historical time-series
}

