#!/usr/bin/env python3
"""
Test script to simulate embedding processes for orchestration testing
"""
import time
import sys
import argparse

def simulate_embeddings_process(process_name, duration=10, progress_mode=False):
    """Simulate an embedding process with progress reporting"""
    print(f"Starting {process_name} simulation...")
    
    if progress_mode:
        print("Progress: 0.0%", flush=True)
        print("Status: initializing", flush=True)
    
    steps = 20
    for i in range(steps + 1):
        progress = (i / steps) * 100
        
        if progress_mode:
            if i < 5:
                status = "loading data"
            elif i < 10:
                status = "processing"
            elif i < 15:
                status = "embedding"
            else:
                status = "finalizing"
                
            print(f"Progress: {progress:.1f}%", flush=True)
            print(f"Status: {status}", flush=True)
        else:
            print(f"{process_name}: {progress:.1f}% complete")
        
        time.sleep(duration / steps)
    
    if progress_mode:
        print("Progress: 100.0%", flush=True)
        print("Status: completed", flush=True)
    
    print(f"{process_name} simulation completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Test embedding process simulation")
    parser.add_argument('--process', choices=['regular', 'kg'], required=True,
                       help='Which process to simulate')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration in seconds (default: 10)')
    parser.add_argument('--progress-mode', action='store_true',
                       help='Enable progress reporting for orchestration')
    parser.add_argument('--fail', action='store_true',
                       help='Simulate process failure')
    
    args = parser.parse_args()
    
    if args.fail:
        print(f"Simulating {args.process} process failure...")
        time.sleep(2)
        sys.exit(1)
    
    process_name = "Regular Embeddings" if args.process == 'regular' else "KG Embeddings"
    simulate_embeddings_process(process_name, args.duration, args.progress_mode)

if __name__ == "__main__":
    main()
