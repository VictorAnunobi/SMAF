#!/usr/bin/env python3
"""
Analyze why SEA performance is ~10x below paper results.
Current: Recall@20 ≈ 0.009, Paper: Recall@20 ≈ 0.095

Potential issues to investigate:
1. Data preprocessing differences
2. Model architecture differences  
3. Loss function implementation
4. Evaluation differences
5. Feature normalization
"""

import numpy as np
import torch
import os
import sys

def analyze_data_statistics():
    """Analyze dataset statistics vs paper"""
    print("=== DATA ANALYSIS ===")
    
    # Check clothing dataset
    data_path = "data/clothing/"
    
    # Load interaction data
    with open(f"{data_path}clothing.inter", 'r') as f:
        lines = f.readlines()
    
    print(f"Total interactions: {len(lines)-1}")  # -1 for header
    
    # Parse interactions
    interactions = []
    for line in lines[1:]:  # Skip header
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            user_id, item_id = parts[0], parts[1]
            interactions.append((user_id, item_id))
    
    users = set([x[0] for x in interactions])
    items = set([x[1] for x in interactions])
    
    print(f"Unique users: {len(users)}")
    print(f"Unique items: {len(items)}")
    print(f"Sparsity: {100 * (1 - len(interactions) / (len(users) * len(items))):.4f}%")
    
    # Check feature files
    if os.path.exists(f"{data_path}image_feat.npy"):
        img_feat = np.load(f"{data_path}image_feat.npy")
        print(f"Image features shape: {img_feat.shape}")
        print(f"Image features mean: {img_feat.mean():.4f}, std: {img_feat.std():.4f}")
        print(f"Image features range: [{img_feat.min():.4f}, {img_feat.max():.4f}]")
    
    if os.path.exists(f"{data_path}text_feat.npy"):
        text_feat = np.load(f"{data_path}text_feat.npy")
        print(f"Text features shape: {text_feat.shape}")
        print(f"Text features mean: {text_feat.mean():.4f}, std: {text_feat.std():.4f}")
        print(f"Text features range: [{text_feat.min():.4f}, {text_feat.max():.4f}]")

def analyze_model_config():
    """Compare our config with paper specifications"""
    print("\n=== MODEL CONFIGURATION ANALYSIS ===")
    
    # Our current best config
    our_config = {
        'embedding_size': 64,
        'n_mm_layers': 1,
        'alpha_contrast': 0.3,
        'temp': 0.15,
        'beta': 0.005,
        'learning_rate': 0.001,
        'batch_size': 256,
        'dropout': 0.1
    }
    
    # Paper typical configs (from similar papers)
    paper_typical = {
        'embedding_size': 64,  # Usually 64 or 128
        'n_mm_layers': 2,      # Often 2-3 layers
        'alpha_contrast': 0.2, # Common value
        'temp': 0.2,          # Common value
        'beta': 0.01,         # Often higher
        'learning_rate': 0.001,
        'batch_size': 512,    # Often larger
        'dropout': 0.1
    }
    
    print("Configuration Comparison:")
    for key in our_config:
        print(f"{key:15}: Ours={our_config[key]:>8}, Paper={paper_typical[key]:>8}")

def analyze_loss_trends():
    """Analyze loss trends from recent logs"""
    print("\n=== LOSS TREND ANALYSIS ===")
    
    log_dir = "src/log/"
    if os.path.exists(log_dir):
        # Find latest clothing log specifically
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log') and 'clothing' in f]
        if log_files:
            # Sort by modification time to get the truly latest
            log_paths = [(f, os.path.getmtime(f"{log_dir}{f}")) for f in log_files]
            latest_log = sorted(log_paths, key=lambda x: x[1])[-1][0]
            print(f"Analyzing latest clothing log: {latest_log}")
            
            with open(f"{log_dir}{latest_log}", 'r') as f:
                lines = f.readlines()
            
            # Extract training losses
            losses = []
            epochs = []
            for line in lines:
                if "training [time:" in line and "train loss:" in line:
                    try:
                        # Extract epoch number
                        if "epoch" in line:
                            epoch_part = line.split("epoch")[1].split()[0]
                            epoch = int(epoch_part)
                            epochs.append(epoch)
                        
                        loss_str = line.split("train loss:")[1].strip().rstrip(']')
                        loss = float(loss_str)
                        losses.append(loss)
                    except:
                        continue
            
            if losses:
                print(f"Total epochs trained: {len(losses)}")
                print(f"Loss trend (first 10 epochs): {losses[:10]}")
                if len(losses) > 10:
                    print(f"Loss trend (last 10 epochs): {losses[-10:]}")
                print(f"Loss convergence: Start={losses[0]:.2f}, End={losses[-1]:.2f}")
                print(f"Loss reduction: {abs(losses[-1] - losses[0]):.2f}")
                
                # Check if loss is negative (should be for BPR)
                if losses[0] > 0:
                    print("WARNING: Positive loss values detected - this might indicate training issues!")
                else:
                    print("Loss values are negative as expected for BPR-based training")
                    
            # Also extract final performance metrics
            final_results = {}
            for line in lines:
                if "████████████████" in line and "BEST" in line:
                    # Look for results in subsequent lines
                    idx = lines.index(line)
                    for i in range(idx, min(idx + 10, len(lines))):
                        if "recall@20:" in lines[i] and "test:" in lines[i].lower():
                            parts = lines[i].split()
                            for j, part in enumerate(parts):
                                if part == "recall@20:":
                                    final_results['test_recall_20'] = float(parts[j+1])
                                elif part == "ndcg@20:":
                                    final_results['test_ndcg_20'] = float(parts[j+1])
                            break
                    break
                    
            if final_results:
                print(f"Final performance: Recall@20={final_results.get('test_recall_20', 'N/A'):.4f}, NDCG@20={final_results.get('test_ndcg_20', 'N/A'):.4f}")
        else:
            print("No clothing log files found")
    else:
        print("Log directory not found")

def suggest_improvements():
    """Suggest next steps based on analysis"""
    print("\n=== SUGGESTED IMPROVEMENTS ===")
    
    suggestions = [
        "1. Try larger batch sizes (512, 1024) - paper often uses larger batches",
        "2. Increase n_mm_layers to 2-3 - more graph convolution layers",
        "3. Try different learning rate schedules (warmup, cosine decay)",
        "4. Experiment with higher beta values (0.01, 0.02) for stronger distancing",
        "5. Check if features need different normalization (L2, batch norm)",
        "6. Try longer training with learning rate scheduling",
        "7. Experiment with different negative sampling strategies",
        "8. Validate that SVD completion is actually helping",
        "9. Check if graph construction needs adjustment",
        "10. Consider different optimizers (AdamW, SGD with momentum)"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def main():
    """Run full analysis"""
    print("SEA Performance Gap Analysis")
    print("=" * 50)
    
    try:
        analyze_data_statistics()
    except Exception as e:
        print(f"Data analysis failed: {e}")
    
    analyze_model_config()
    
    try:
        analyze_loss_trends()
    except Exception as e:
        print(f"Loss analysis failed: {e}")
    
    suggest_improvements()

if __name__ == "__main__":
    main()
