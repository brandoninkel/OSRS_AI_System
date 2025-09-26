#!/usr/bin/env python3
"""
Performance testing script for mxbai KG embeddings

Tests different worker counts and monitors system resources to find optimal settings.
"""

import json
import logging
import os
import sys
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the embeddings service to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'api', 'embeddings'))
from embedding_service import EmbeddingService, EmbeddingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTester:
    def __init__(self):
        """Initialize the performance tester"""
        self.repo_root = Path(__file__).resolve().parents[2]
        self.kg_model_dir = self.repo_root / "data" / "kg_model"
        
        # Initialize mxbai embedding service
        config = EmbeddingConfig(
            model_name="mxbai-embed-large:latest",
            batch_size=1,
            timeout=60
        )
        self.embedding_service = EmbeddingService(config)
        
        # Load sample entities for testing
        self.test_entities = self.load_test_entities()
        
    def load_test_entities(self, sample_size: int = 100) -> List[str]:
        """Load a small sample of entities for performance testing"""
        entity_map_path = self.kg_model_dir / "entity_to_id.json"
        if not entity_map_path.exists():
            raise FileNotFoundError(f"KG entity mapping not found: {entity_map_path}")
            
        with open(entity_map_path, 'r', encoding='utf-8') as f:
            all_entities = json.load(f)
            
        # Take first N entities for consistent testing
        entities = list(all_entities.keys())[:sample_size]
        logger.info(f"Loaded {len(entities)} test entities")
        return entities
        
    def monitor_system_resources(self, duration: float) -> Dict[str, float]:
        """Monitor system resources during a test"""
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < duration:
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            memory_samples.append(psutil.virtual_memory().percent)
            time.sleep(0.5)
            
        return {
            'avg_cpu': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            'max_cpu': max(cpu_samples) if cpu_samples else 0,
            'avg_memory': sum(memory_samples) / len(memory_samples) if memory_samples else 0,
            'max_memory': max(memory_samples) if memory_samples else 0
        }
        
    def test_embedding_speed(self, entities: List[str], max_workers: int) -> Dict[str, Any]:
        """Test embedding speed with given worker count"""
        print(f"🧪 Testing {len(entities)} entities with {max_workers} workers...")
        
        processed_count = 0
        failed_count = 0
        lock = threading.Lock()
        
        def process_entity(entity_name: str) -> bool:
            nonlocal processed_count, failed_count
            try:
                embedding = self.embedding_service.embed_text(entity_name)
                with lock:
                    if embedding:
                        processed_count += 1
                        return True
                    else:
                        failed_count += 1
                        return False
            except Exception as e:
                with lock:
                    failed_count += 1
                return False
        
        # Start monitoring
        start_time = time.time()
        
        # Process entities in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_entity, entity) for entity in entities]
            
            # Monitor resources while processing
            monitor_thread = threading.Thread(
                target=lambda: self.monitor_system_resources(60),  # Monitor for up to 60 seconds
                daemon=True
            )
            monitor_thread.start()
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions
                
        end_time = time.time()
        duration = end_time - start_time
        
        # Get final system stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        return {
            'workers': max_workers,
            'entities': len(entities),
            'processed': processed_count,
            'failed': failed_count,
            'duration': duration,
            'entities_per_second': processed_count / duration if duration > 0 else 0,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'success_rate': processed_count / len(entities) if entities else 0
        }
        
    def run_performance_tests(self) -> List[Dict[str, Any]]:
        """Run performance tests with different worker counts"""
        print("🚀 Starting Performance Tests")
        print("=" * 60)
        
        # Test different worker counts
        worker_counts = [1, 2, 4, 8, 12, 16, 24, 32]
        results = []
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"💻 System Info:")
        print(f"   CPU cores: {cpu_count}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Test entities: {len(self.test_entities)}")
        print()
        
        for workers in worker_counts:
            if workers > cpu_count * 4:  # Skip excessive worker counts
                continue
                
            try:
                result = self.test_embedding_speed(self.test_entities, workers)
                results.append(result)
                
                print(f"✅ Workers: {workers:2d} | "
                      f"Speed: {result['entities_per_second']:.1f} entities/sec | "
                      f"CPU: {result['cpu_percent']:.1f}% | "
                      f"Memory: {result['memory_percent']:.1f}% | "
                      f"Success: {result['success_rate']*100:.1f}%")
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Test failed for {workers} workers: {e}")
                continue
                
        return results
        
    def find_optimal_workers(self, results: List[Dict[str, Any]]) -> int:
        """Find optimal worker count based on results"""
        if not results:
            return 8  # Default fallback
            
        # Filter successful results
        successful_results = [r for r in results if r['success_rate'] > 0.8]
        if not successful_results:
            return 8
            
        # Find the worker count with best entities/second that doesn't max out CPU
        best_result = None
        best_score = 0
        
        for result in successful_results:
            # Penalize if CPU usage is too high (>90%)
            cpu_penalty = 1.0 if result['cpu_percent'] < 90 else 0.5
            score = result['entities_per_second'] * cpu_penalty
            
            if score > best_score:
                best_score = score
                best_result = result
                
        return best_result['workers'] if best_result else 8
        
    def print_recommendations(self, results: List[Dict[str, Any]]):
        """Print performance recommendations"""
        if not results:
            print("❌ No results to analyze")
            return
            
        optimal_workers = self.find_optimal_workers(results)
        best_result = next((r for r in results if r['workers'] == optimal_workers), None)
        
        print("\n🎯 PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        if best_result:
            print(f"✅ Optimal worker count: {optimal_workers}")
            print(f"📊 Expected performance:")
            print(f"   Speed: {best_result['entities_per_second']:.1f} entities/second")
            print(f"   CPU usage: {best_result['cpu_percent']:.1f}%")
            print(f"   Memory usage: {best_result['memory_percent']:.1f}%")
            print(f"   Success rate: {best_result['success_rate']*100:.1f}%")
            
            # Calculate time estimates for full dataset
            entity_map_path = self.kg_model_dir / "entity_to_id.json"
            if entity_map_path.exists():
                with open(entity_map_path, 'r') as f:
                    total_entities = len(json.load(f))
                    
                estimated_hours = total_entities / best_result['entities_per_second'] / 3600
                print(f"\n⏱️  Full dataset estimate:")
                print(f"   Total entities: {total_entities:,}")
                print(f"   Estimated time: {estimated_hours:.1f} hours")
                
        print(f"\n🚀 Recommended command:")
        print(f"   python3 scripts/kg/create_mxbai_kg_embeddings.py --workers {optimal_workers}")

def main():
    """Main function"""
    try:
        tester = PerformanceTester()
        results = tester.run_performance_tests()
        tester.print_recommendations(results)
        
        # Save results for reference
        results_file = Path("data/mxbai_performance_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
