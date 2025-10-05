#!/usr/bin/env python3
"""
OSRS AI System - Comprehensive Playwright Testing Suite

Tests the complete system including:
- GUI interaction
- API responses
- Attribution system
- Economic hypothesis mode
- Browser console logs
- Terminal output monitoring
"""

import asyncio
import json
import time
from datetime import datetime
from playwright.async_api import async_playwright, Page, Browser
from typing import Dict, List, Any

class OSRSAITester:
    """Comprehensive testing suite for OSRS AI system"""
    
    def __init__(self):
        self.gui_url = "http://localhost:3005"
        self.api_url = "http://localhost:5001"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": []
            }
        }
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 80)
        print("🧪 OSRS AI SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print()
        
        async with async_playwright() as p:
            # Launch browser with console logging
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            
            # Capture console messages
            console_logs = []
            page.on("console", lambda msg: console_logs.append({
                "type": msg.type,
                "text": msg.text,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Capture network errors
            page.on("pageerror", lambda err: console_logs.append({
                "type": "error",
                "text": str(err),
                "timestamp": datetime.now().isoformat()
            }))
            
            try:
                # Test 1: GUI loads correctly
                await self.test_gui_loads(page)
                
                # Test 2: Simple query
                await self.test_simple_query(page)
                
                # Test 3: Price query
                await self.test_price_query(page)
                
                # Test 4: Economic hypothesis query
                await self.test_economic_hypothesis(page)
                
                # Test 5: Attribution system
                await self.test_attributions(page)
                
                # Test 6: Barrows brothers query (previously failed)
                await self.test_barrows_query(page)
                
                # Test 7: Charged item price query
                await self.test_charged_item_price(page)
                
                # Analyze console logs
                self.analyze_console_logs(console_logs)
                
            finally:
                await browser.close()
        
        # Print results
        self.print_results()
        
        # Save results to file
        self.save_results()
    
    async def test_gui_loads(self, page: Page):
        """Test 1: Verify GUI loads correctly"""
        test_name = "GUI Loads"
        print(f"\n📋 Test 1: {test_name}")
        print("-" * 80)
        
        try:
            start_time = time.time()
            await page.goto(self.gui_url, wait_until="networkidle", timeout=30000)
            load_time = time.time() - start_time
            
            # Check for key elements
            title = await page.title()
            has_input = await page.locator('textbox, textarea, input').count() > 0
            has_send_button = await page.locator('button:has-text("Send")').count() > 0
            
            if has_input and has_send_button:
                self.record_test(test_name, "PASSED", {
                    "load_time": f"{load_time:.2f}s",
                    "title": title,
                    "has_input": True,
                    "has_send_button": True
                })
                print(f"✅ PASSED - GUI loaded in {load_time:.2f}s")
            else:
                self.record_test(test_name, "FAILED", {
                    "reason": "Missing required elements",
                    "has_input": has_input,
                    "has_send_button": has_send_button
                })
                print(f"❌ FAILED - Missing required elements")
                
        except Exception as e:
            self.record_test(test_name, "FAILED", {"error": str(e)})
            print(f"❌ FAILED - {str(e)}")
    
    async def test_simple_query(self, page: Page):
        """Test 2: Simple factual query"""
        test_name = "Simple Query"
        query = "What is Zulrah's combat level?"
        print(f"\n📋 Test 2: {test_name}")
        print(f"Query: {query}")
        print("-" * 80)
        
        try:
            result = await self.send_query_and_wait(page, query, timeout=60)
            
            if result["success"]:
                # Check if response contains expected info
                response_text = result["response"].lower()
                has_number = any(char.isdigit() for char in result["response"])
                has_sources = result["sources_count"] > 0
                
                if has_number and has_sources:
                    self.record_test(test_name, "PASSED", {
                        "query": query,
                        "response_length": len(result["response"]),
                        "sources_count": result["sources_count"],
                        "response_time": result["response_time"]
                    })
                    print(f"✅ PASSED - Response received in {result['response_time']:.1f}s")
                    print(f"   Response: {result['response'][:200]}...")
                    print(f"   Sources: {result['sources_count']}")
                else:
                    self.record_test(test_name, "WARNING", {
                        "reason": "Response may be incomplete",
                        "has_number": has_number,
                        "has_sources": has_sources
                    })
                    print(f"⚠️  WARNING - Response may be incomplete")
            else:
                self.record_test(test_name, "FAILED", result)
                print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.record_test(test_name, "FAILED", {"error": str(e)})
            print(f"❌ FAILED - {str(e)}")
    
    async def test_price_query(self, page: Page):
        """Test 3: Price query using live API"""
        test_name = "Price Query"
        query = "How much is an abyssal whip?"
        print(f"\n📋 Test 3: {test_name}")
        print(f"Query: {query}")
        print("-" * 80)
        
        try:
            result = await self.send_query_and_wait(page, query, timeout=60)
            
            if result["success"]:
                response_text = result["response"].lower()
                has_price = any(word in response_text for word in ["gp", "gold", "coins", "price"])
                has_number = any(char.isdigit() for char in result["response"])
                
                if has_price and has_number:
                    self.record_test(test_name, "PASSED", {
                        "query": query,
                        "response_length": len(result["response"]),
                        "response_time": result["response_time"]
                    })
                    print(f"✅ PASSED - Price data received in {result['response_time']:.1f}s")
                    print(f"   Response: {result['response'][:200]}...")
                else:
                    self.record_test(test_name, "WARNING", {
                        "reason": "Response may not contain price data",
                        "has_price": has_price,
                        "has_number": has_number
                    })
                    print(f"⚠️  WARNING - Response may not contain price data")
            else:
                self.record_test(test_name, "FAILED", result)
                print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.record_test(test_name, "FAILED", {"error": str(e)})
            print(f"❌ FAILED - {str(e)}")
    
    async def test_economic_hypothesis(self, page: Page):
        """Test 4: Economic hypothesis mode"""
        test_name = "Economic Hypothesis"
        query = "Is it profitable to kill Vorkath?"
        print(f"\n📋 Test 4: {test_name}")
        print(f"Query: {query}")
        print("-" * 80)
        
        try:
            result = await self.send_query_and_wait(page, query, timeout=120)
            
            if result["success"]:
                response_text = result["response"].lower()
                has_profit_analysis = any(word in response_text for word in ["profit", "gp/hour", "cost", "worth"])
                has_multiple_prices = result["response"].count("gp") > 1 or result["response"].count("GP") > 1
                
                if has_profit_analysis:
                    self.record_test(test_name, "PASSED", {
                        "query": query,
                        "response_length": len(result["response"]),
                        "has_multiple_prices": has_multiple_prices,
                        "response_time": result["response_time"]
                    })
                    print(f"✅ PASSED - Economic analysis received in {result['response_time']:.1f}s")
                    print(f"   Response: {result['response'][:300]}...")
                else:
                    self.record_test(test_name, "WARNING", {
                        "reason": "Response may not contain economic analysis",
                        "has_profit_analysis": has_profit_analysis
                    })
                    print(f"⚠️  WARNING - Response may not contain economic analysis")
            else:
                self.record_test(test_name, "FAILED", result)
                print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.record_test(test_name, "FAILED", {"error": str(e)})
            print(f"❌ FAILED - {str(e)}")

    async def test_attributions(self, page: Page):
        """Test 5: Attribution system"""
        test_name = "Attribution System"
        print(f"\n📋 Test 5: {test_name}")
        print("-" * 80)
        print("   (Skipping for now - will test manually)")
        self.record_test(test_name, "SKIPPED", {"reason": "Manual testing required"})

    async def test_barrows_query(self, page: Page):
        """Test 6: Barrows brothers query"""
        test_name = "Barrows Brothers Query"
        query = "What are each of the Barrows brothers' abilities?"
        print(f"\n📋 Test 6: {test_name}")
        print(f"Query: {query}")
        print("-" * 80)

        try:
            result = await self.send_query_and_wait(page, query, timeout=120)

            if result["success"] and len(result["response"]) > 500:
                self.record_test(test_name, "PASSED", {
                    "query": query,
                    "response_length": len(result["response"]),
                    "response_time": result["response_time"]
                })
                print(f"✅ PASSED - Comprehensive response received")
            else:
                self.record_test(test_name, "FAILED", result)
                print(f"❌ FAILED")
        except Exception as e:
            self.record_test(test_name, "FAILED", {"error": str(e)})
            print(f"❌ FAILED - {str(e)}")

    async def test_charged_item_price(self, page: Page):
        """Test 7: Charged item price query"""
        test_name = "Charged Item Price"
        query = "How much is a trident of the seas?"
        print(f"\n📋 Test 7: {test_name}")
        print(f"Query: {query}")
        print("-" * 80)

        try:
            result = await self.send_query_and_wait(page, query, timeout=60)

            if result["success"]:
                response_text = result["response"].lower()
                has_price = "gp" in response_text or "gold" in response_text

                if has_price:
                    self.record_test(test_name, "PASSED", {
                        "query": query,
                        "response_time": result["response_time"]
                    })
                    print(f"✅ PASSED - Price data received")
                else:
                    self.record_test(test_name, "WARNING", {"reason": "No price data"})
                    print(f"⚠️  WARNING - No price data")
            else:
                self.record_test(test_name, "FAILED", result)
                print(f"❌ FAILED")
        except Exception as e:
            self.record_test(test_name, "FAILED", {"error": str(e)})
            print(f"❌ FAILED - {str(e)}")

    async def send_query_and_wait(self, page: Page, query: str, timeout: int = 60) -> Dict[str, Any]:
        """Send a query and wait for response"""
        start_time = time.time()

        try:
            # Find input field using accessibility role
            input_field = page.get_by_role("textbox")
            await input_field.fill(query)

            # Find and click send button
            send_button = page.get_by_role("button", name="Send")
            await send_button.click()

            # Wait for response (look for new content)
            await asyncio.sleep(5)  # Initial wait

            # Poll for response with timeout
            elapsed = 0
            while elapsed < timeout:
                page_text = await page.inner_text('body')

                # Check if we have a response (look for sources or substantial text)
                if "Sources Used" in page_text or len(page_text) > 1000:
                    response_time = time.time() - start_time

                    # Extract response and sources
                    sources_count = page_text.count("http") // 2  # Rough estimate

                    return {
                        "success": True,
                        "response": page_text,
                        "sources_count": sources_count,
                        "response_time": response_time
                    }

                await asyncio.sleep(2)
                elapsed = time.time() - start_time

            return {
                "success": False,
                "error": "Timeout waiting for response"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_console_logs(self, logs: List[Dict]):
        """Analyze browser console logs"""
        print(f"\n📋 Browser Console Analysis")
        print("-" * 80)

        errors = [log for log in logs if log["type"] == "error"]
        warnings = [log for log in logs if log["type"] == "warning"]

        print(f"   Total logs: {len(logs)}")
        print(f"   Errors: {len(errors)}")
        print(f"   Warnings: {len(warnings)}")

        if errors:
            print(f"\n   ❌ Console Errors:")
            for err in errors[:5]:  # Show first 5
                print(f"      - {err['text'][:100]}")

        self.results["console_logs"] = {
            "total": len(logs),
            "errors": len(errors),
            "warnings": len(warnings)
        }

    def record_test(self, name: str, status: str, details: Dict = None):
        """Record test result"""
        self.results["tests"].append({
            "name": name,
            "status": status,
            "details": details or {}
        })

        self.results["summary"]["total"] += 1
        if status == "PASSED":
            self.results["summary"]["passed"] += 1
        elif status == "FAILED":
            self.results["summary"]["failed"] += 1
        elif status == "WARNING":
            self.results["summary"]["warnings"].append(name)

    def print_results(self):
        """Print test results summary"""
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        summary = self.results["summary"]
        print(f"\nTotal Tests: {summary['total']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️  Warnings: {len(summary['warnings'])}")

        if summary['warnings']:
            print(f"\nWarnings:")
            for warning in summary['warnings']:
                print(f"   - {warning}")

        pass_rate = (summary['passed'] / summary['total'] * 100) if summary['total'] > 0 else 0
        print(f"\nPass Rate: {pass_rate:.1f}%")

    def save_results(self):
        """Save results to JSON file"""
        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")


async def main():
    """Main entry point"""
    tester = OSRSAITester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
