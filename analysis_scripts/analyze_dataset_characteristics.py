#!/usr/bin/env python3
"""
Dataset Characteristics Analysis for SEA Model Performance

Analyzes why Baby dataset achieves near-paper performance (2x gap) while
Clothing dataset has larger gap (7.3x). This analysis will guide dataset-specific
optimizations.

Based on breakthrough results:
- Baby: Recall@20: 0.0474 (only 2x from paper)
- Sports: Recall@20: 0.0273 (3.5x from paper) 
- Clothing: Recall@20: 0.0130 (7.3x from paper)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_interaction_data(dataset_name):
    """Analyze interaction patterns in .inter files"""
    inter_file = f"data/{dataset_name}/{dataset_name}.inter"
    
    if not os.path.exists(inter_file):
        print(f"ERROR: {inter_file} not found")
        return None
    
    print(f"\nAnalyzing {dataset_name} interaction data...")
    
    # Read interaction data
    data = pd.read_csv(inter_file, sep='\t')
    print(f"   Columns: {list(data.columns)}")
    print(f"   Shape: {data.shape}")
    
    # Basic statistics
    stats = {
        'total_interactions': len(data),
        'unique_users': data['userID'].nunique(),
        'unique_items': data['itemID'].nunique(),
        'avg_rating': data['rating'].mean() if 'rating' in data.columns else None,
        'rating_std': data['rating'].std() if 'rating' in data.columns else None,
        'sparsity': 1 - (len(data) / (data['userID'].nunique() * data['itemID'].nunique()))
    }
    
    # User activity distribution
    user_activity = data['userID'].value_counts()
    stats['avg_interactions_per_user'] = user_activity.mean()
    stats['median_interactions_per_user'] = user_activity.median()
    stats['max_interactions_per_user'] = user_activity.max()
    stats['min_interactions_per_user'] = user_activity.min()
    
    # Item popularity distribution
    item_popularity = data['itemID'].value_counts()
    stats['avg_interactions_per_item'] = item_popularity.mean()
    stats['median_interactions_per_item'] = item_popularity.median()
    stats['max_interactions_per_item'] = item_popularity.max()
    stats['min_interactions_per_item'] = item_popularity.min()
    
    return stats, user_activity, item_popularity

def analyze_feature_data(dataset_name):
    """Analyze image and text feature characteristics"""
    image_file = f"data/{dataset_name}/image_feat.npy"
    text_file = f"data/{dataset_name}/text_feat.npy"
    
    features = {}
    
    if os.path.exists(image_file):
        print(f"\nAnalyzing {dataset_name} image features...")
        img_feat = np.load(image_file)
        features['image'] = {
            'shape': img_feat.shape,
            'mean': img_feat.mean(),
            'std': img_feat.std(),
            'min': img_feat.min(),
            'max': img_feat.max(),
            'sparsity': (img_feat == 0).mean(),
            'feature_dim': img_feat.shape[1] if len(img_feat.shape) > 1 else 0
        }
        
        # Analyze feature distribution
        features['image']['per_dim_std'] = img_feat.std(axis=0).mean()
        features['image']['per_dim_mean'] = img_feat.mean(axis=0).mean()
        
    if os.path.exists(text_file):
        print(f"\nAnalyzing {dataset_name} text features...")
        text_feat = np.load(text_file)
        features['text'] = {
            'shape': text_feat.shape,
            'mean': text_feat.mean(),
            'std': text_feat.std(),
            'min': text_feat.min(),
            'max': text_feat.max(),
            'sparsity': (text_feat == 0).mean(),
            'feature_dim': text_feat.shape[1] if len(text_feat.shape) > 1 else 0
        }
        
        # Analyze feature distribution
        features['text']['per_dim_std'] = text_feat.std(axis=0).mean()
        features['text']['per_dim_mean'] = text_feat.mean(axis=0).mean()
        
    return features

def compare_datasets():
    """Compare characteristics across all three datasets"""
    datasets = ['baby', 'clothing', 'sports']
    performance = {
        'baby': {'recall20': 0.0474, 'gap_ratio': 2.0},
        'sports': {'recall20': 0.0273, 'gap_ratio': 3.5}, 
        'clothing': {'recall20': 0.0130, 'gap_ratio': 7.3}
    }
    
    print("=" * 80)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("Investigating why Baby dataset achieves near-paper performance")
    print("=" * 80)
    
    all_stats = {}
    all_features = {}
    
    for dataset in datasets:
        print(f"\n{'=' * 20} {dataset.upper()} DATASET {'=' * 20}")
        perf = performance[dataset]
        print(f"Performance: Recall@20={perf['recall20']:.4f}, Gap={perf['gap_ratio']:.1f}x")
        
        # Analyze interactions
        stats, user_activity, item_popularity = analyze_interaction_data(dataset)
        all_stats[dataset] = stats
        
        # Analyze features
        features = analyze_feature_data(dataset)
        all_features[dataset] = features
    
    # Create comparison summary
    print(f"\n{'=' * 60}")
    print("COMPARATIVE ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    
    # Performance comparison
    print("\nPERFORMANCE RANKING:")
    sorted_datasets = sorted(datasets, key=lambda d: performance[d]['recall20'], reverse=True)
    for i, dataset in enumerate(sorted_datasets, 1):
        perf = performance[dataset]
        print(f"{i}. {dataset.upper()}: Recall@20={perf['recall20']:.4f} (Gap: {perf['gap_ratio']:.1f}x)")
    
    # Key characteristics comparison
    print(f"\nINTERACTION CHARACTERISTICS:")
    print(f"{'Dataset':<10} {'Users':<8} {'Items':<8} {'Interactions':<12} {'Sparsity':<10} {'Avg/User':<10}")
    print("-" * 70)
    
    for dataset in sorted_datasets:
        stats = all_stats[dataset]
        if stats:
            print(f"{dataset:<10} {stats['unique_users']:<8} {stats['unique_items']:<8} "
                  f"{stats['total_interactions']:<12} {stats['sparsity']:<10.4f} "
                  f"{stats['avg_interactions_per_user']:<10.2f}")
    
    # Feature characteristics comparison
    print(f"\nIMAGE FEATURE CHARACTERISTICS:")
    print(f"{'Dataset':<10} {'Dim':<6} {'Mean':<10} {'Std':<10} {'Sparsity':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    
    for dataset in sorted_datasets:
        features = all_features[dataset]
        if 'image' in features:
            img = features['image']
            print(f"{dataset:<10} {img['feature_dim']:<6} {img['mean']:<10.4f} "
                  f"{img['std']:<10.4f} {img['sparsity']:<10.4f} "
                  f"{img['min']:<10.4f} {img['max']:<10.4f}")
    
    print(f"\nTEXT FEATURE CHARACTERISTICS:")
    print(f"{'Dataset':<10} {'Dim':<6} {'Mean':<10} {'Std':<10} {'Sparsity':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    
    for dataset in sorted_datasets:
        features = all_features[dataset]
        if 'text' in features:
            txt = features['text']
            print(f"{dataset:<10} {txt['feature_dim']:<6} {txt['mean']:<10.4f} "
                  f"{txt['std']:<10.4f} {txt['sparsity']:<10.4f} "
                  f"{txt['min']:<10.4f} {txt['max']:<10.4f}")
    
    # Correlation analysis
    print(f"\nKEY INSIGHTS:")
    
    # Find patterns that correlate with performance
    recall_scores = [performance[d]['recall20'] for d in sorted_datasets]
    
    if all_stats[sorted_datasets[0]]:  # Check if we have stats
        sparsities = [all_stats[d]['sparsity'] for d in sorted_datasets]
        avg_interactions = [all_stats[d]['avg_interactions_per_user'] for d in sorted_datasets]
        
        print(f"\n1. SPARSITY vs PERFORMANCE:")
        for i, dataset in enumerate(sorted_datasets):
            print(f"   {dataset}: Sparsity={sparsities[i]:.4f}, Recall@20={recall_scores[i]:.4f}")
        
        print(f"\n2. USER ACTIVITY vs PERFORMANCE:")
        for i, dataset in enumerate(sorted_datasets):
            print(f"   {dataset}: Avg interactions/user={avg_interactions[i]:.2f}, Recall@20={recall_scores[i]:.4f}")
    
    # Feature quality analysis
    print(f"\n3. FEATURE QUALITY vs PERFORMANCE:")
    for dataset in sorted_datasets:
        features = all_features[dataset]
        perf = performance[dataset]
        if 'image' in features and 'text' in features:
            img_quality = 1 - features['image']['sparsity']  # Less sparsity = better quality
            txt_quality = 1 - features['text']['sparsity']
            print(f"   {dataset}: Image quality={img_quality:.3f}, Text quality={txt_quality:.3f}, "
                  f"Recall@20={perf['recall20']:.4f}")
    
    print(f"\nRECOMMENDATIONS:")
    print(f"1. Focus on Baby dataset characteristics that lead to 2x gap performance")
    print(f"2. Investigate if Clothing/Sports can benefit from Baby-specific optimizations")
    print(f"3. Consider dataset-specific hyperparameter tuning based on these characteristics")
    print(f"4. Analyze if feature preprocessing should be dataset-specific")

if __name__ == "__main__":
    compare_datasets()
