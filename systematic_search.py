#!/usr/bin/env python3
import os
import sys
import subprocess
import time
from itertools import product

def run_experiment(config, dataset="clothing"):
    """Run a single experiment with given config"""
    cmd = [
        "python", "src/main.py", 
        "--dataset", dataset,
        "--alpha_contrast", str(config['alpha_contrast']),
        "--temp", str(config['temp']),
        "--beta", str(config['beta'])
    ]
    
    print(f"Running: {config}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        # Parse final results from output
        lines = result.stdout.split('\n')
        final_results = {}
        
        for line in lines:
            if "████████████████" in line and "BEST" in line:
                # Look for the results in next few lines
                for i in range(len(lines)):
                    if "recall@20:" in lines[i] and "test:" in lines[i].lower():
                        # Extract recall@20 and ndcg@20
                        parts = lines[i].split()
                        for j, part in enumerate(parts):
                            if part == "recall@20:":
                                final_results['recall_20'] = float(parts[j+1])
                            elif part == "ndcg@20:":
                                final_results['ndcg_20'] = float(parts[j+1])
                        break
                break
        
        return final_results
        
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out: {config}")
        return None
    except Exception as e:
        print(f"Experiment failed: {config}, Error: {e}")
        return None

def search_batch_size_impact():
    """Test if larger batch sizes significantly improve performance"""
    print("=== BATCH SIZE IMPACT SEARCH ===")
    
    # Our best base config
    base_config = {
        'alpha_contrast': 0.3,
        'temp': 0.15,
        'beta': 0.005
    }
    
    batch_sizes = [256, 512, 768, 1024]  # Current: 256
    
    results = []
    for batch_size in batch_sizes:
        # Modify main.py temporarily to use this batch size
        config = base_config.copy()
        config['batch_size'] = batch_size
        
        print(f"\nTesting batch_size={batch_size}")
        
        # Update main.py config temporarily
        update_batch_size_in_main(batch_size)
        
        result = run_experiment(config)
        if result:
            result['batch_size'] = batch_size
            results.append(result)
            print(f"Result: Recall@20={result.get('recall_20', 'N/A')}")
    
    return results

def search_architecture_variants():
    """Test architectural changes that might have big impact"""
    print("=== ARCHITECTURE VARIANT SEARCH ===")
    
    variants = [
        # (embedding_size, n_mm_layers)
        (64, 1),   # Current best
        (96, 1),   # Larger embeddings
        (128, 1),  # Even larger
        (64, 2),   # More GCN layers  
        (96, 2),   # Both larger
        (128, 2),  # Maximum
    ]
    
    base_config = {
        'alpha_contrast': 0.3,
        'temp': 0.15,
        'beta': 0.005
    }
    
    results = []
    for emb_size, n_layers in variants:
        config = base_config.copy()
        config['embedding_size'] = emb_size
        config['n_mm_layers'] = n_layers
        
        print(f"\nTesting embedding_size={emb_size}, n_mm_layers={n_layers}")
        
        # Update main.py config temporarily
        update_architecture_in_main(emb_size, n_layers)
        
        result = run_experiment(config)
        if result:
            result['embedding_size'] = emb_size
            result['n_mm_layers'] = n_layers
            results.append(result)
            print(f"Result: Recall@20={result.get('recall_20', 'N/A')}")
    
    return results

def search_learning_strategies():
    """Test different learning strategies"""
    print("=== LEARNING STRATEGY SEARCH ===")
    
    strategies = [
        # (learning_rate, epochs, stopping_step)
        (0.001, 100, 25),   # Current
        (0.002, 100, 25),   # Higher LR
        (0.0005, 150, 30),  # Lower LR, longer
        (0.001, 200, 40),   # Much longer
        (0.003, 50, 15),    # Fast and aggressive
    ]
    
    base_config = {
        'alpha_contrast': 0.3,
        'temp': 0.15,
        'beta': 0.005
    }
    
    results = []
    for lr, epochs, stop_step in strategies:
        config = base_config.copy()
        config['learning_rate'] = lr
        config['epochs'] = epochs
        config['stopping_step'] = stop_step
        
        print(f"\nTesting lr={lr}, epochs={epochs}, stopping_step={stop_step}")
        
        # Update main.py config temporarily  
        update_learning_in_main(lr, epochs, stop_step)
        
        result = run_experiment(config)
        if result:
            result['learning_rate'] = lr
            result['epochs'] = epochs
            result['stopping_step'] = stop_step
            results.append(result)
            print(f"Result: Recall@20={result.get('recall_20', 'N/A')}")
    
    return results

def update_batch_size_in_main(batch_size):
    """Temporarily update batch size in main.py"""
    # This is a simplified version - you'd need to actually modify the file
    pass

def update_architecture_in_main(embedding_size, n_mm_layers):
    """Temporarily update architecture in main.py"""
    # This is a simplified version - you'd need to actually modify the file
    pass

def update_learning_in_main(learning_rate, epochs, stopping_step):
    """Temporarily update learning parameters in main.py"""
    # This is a simplified version - you'd need to actually modify the file
    pass

def analyze_results(all_results):
    """Analyze all experimental results"""
    print("\n=== RESULTS ANALYSIS ===")
    
    if not all_results:
        print("No results to analyze")
        return
    
    # Sort by recall@20
    sorted_results = sorted(all_results, 
                          key=lambda x: x.get('recall_20', 0), 
                          reverse=True)
    
    print("Top 5 configurations:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. Recall@20: {result.get('recall_20', 'N/A'):.4f}, Config: {result}")
    
    # Look for patterns
    best_recall = max([r.get('recall_20', 0) for r in all_results])
    improvements = [r for r in all_results if r.get('recall_20', 0) > 0.01]  # > current best
    
    if improvements:
        print(f"\nConfigurations that improved over baseline (>0.010):")
        for result in improvements:
            print(f"Recall@20: {result.get('recall_20', 'N/A'):.4f}, Config: {result}")
    else:
        print("\nNo configurations significantly improved over baseline")

def main():
    """Run systematic search"""
    print("Systematic Hyperparameter Search for SEA Performance Gap")
    print("=" * 60)
    
    all_results = []
    
    # Test each category
    print("Phase 1: Batch Size Impact")
    batch_results = search_batch_size_impact()
    all_results.extend(batch_results)
    
    print("\nPhase 2: Architecture Variants")  
    arch_results = search_architecture_variants()
    all_results.extend(arch_results)
    
    print("\nPhase 3: Learning Strategies")
    learning_results = search_learning_strategies()
    all_results.extend(learning_results)
    
    # Analyze all results
    analyze_results(all_results)

if __name__ == "__main__":
    # Note: This is a template script
    # You would need to implement the main.py modification functions
    # For now, run analysis only
    print("This is a template for systematic search.")
    print("Run the analysis script first:")
    print("python analyze_performance_gap.py")
