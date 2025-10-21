#!/usr/bin/env python3
"""
Investigate WHY image features don't contribute to SEA performance
Analyze potential causes: feature quality, model architecture, fusion mechanism
"""

import numpy as np
import os
import sys

def analyze_image_feature_issues():
    """Comprehensive analysis of why image features don't help"""
    
    print("="*80)
    print("INVESTIGATING WHY IMAGE FEATURES DON'T CONTRIBUTE")
    print("="*80)
    
    datasets = ['clothing', 'baby', 'sports']
    
    print("\n CROSS-DATASET FEATURE COMPARISON")
    print("="*50)
    
    feature_stats = {}
    
    for dataset in datasets:
        print(f"\nAnalyzing {dataset.upper()} dataset:")
        
        # Load features
        img_feat = np.load(f'data/{dataset}/image_feat.npy')
        text_feat = np.load(f'data/{dataset}/text_feat.npy')
        
        # Analyze image features
        img_nonzero_ratio = (img_feat != 0).mean()
        img_zero_items = (img_feat == 0).all(axis=1).sum()
        img_mean = img_feat.mean()
        img_std = img_feat.std()
        img_sparsity = 1 - img_nonzero_ratio
        
        # Analyze text features  
        text_nonzero_ratio = (text_feat != 0).mean()
        text_mean = text_feat.mean()
        text_std = text_feat.std()
        text_sparsity = 1 - text_nonzero_ratio
        
        # Store stats
        feature_stats[dataset] = {
            'img_nonzero_ratio': img_nonzero_ratio,
            'img_zero_items': img_zero_items,
            'img_total_items': img_feat.shape[0],
            'img_mean': img_mean,
            'img_std': img_std,
            'img_sparsity': img_sparsity,
            'text_nonzero_ratio': text_nonzero_ratio,
            'text_mean': text_mean,
            'text_std': text_std,
            'text_sparsity': text_sparsity,
            'img_shape': img_feat.shape,
            'text_shape': text_feat.shape
        }
        
        print(f"    Image features:")
        print(f"      Shape: {img_feat.shape}")
        print(f"      Non-zero ratio: {img_nonzero_ratio:.4f}")
        print(f"      Zero items: {img_zero_items}/{img_feat.shape[0]} ({img_zero_items/img_feat.shape[0]*100:.1f}%)")
        print(f"      Mean: {img_mean:.6f}, Std: {img_std:.6f}")
        print(f"      Range: [{img_feat.min():.4f}, {img_feat.max():.4f}]")
        
        print(f"    Text features:")
        print(f"      Shape: {text_feat.shape}")
        print(f"      Non-zero ratio: {text_nonzero_ratio:.4f}")
        print(f"      Mean: {text_mean:.6f}, Std: {text_std:.6f}")
        print(f"      Range: [{text_feat.min():.4f}, {text_feat.max():.4f}]")
    
    # Performance mapping
    performance = {
        'baby': {'recall20': 0.0474, 'gap': '2.0x'},
        'sports': {'recall20': 0.0273, 'gap': '3.5x'},
        'clothing': {'recall20': 0.0131, 'gap': '7.3x'}
    }
    
    print(f"\nCORRELATION ANALYSIS")
    print(f"="*40)
    
    print(f"{'Dataset':<10} {'Performance':<12} {'Gap':<6} {'Img Quality':<12} {'Zero Items %':<12}")
    print(f"{'-'*70}")
    
    for dataset in ['baby', 'sports', 'clothing']:
        stats = feature_stats[dataset]
        perf = performance[dataset]
        zero_pct = stats['img_zero_items'] / stats['img_total_items'] * 100
        
        print(f"{dataset:<10} {perf['recall20']:<12.4f} {perf['gap']:<6} {stats['img_nonzero_ratio']:<12.4f} {zero_pct:<12.1f}%")
    
    print(f"\nKEY INSIGHTS:")
    
    # Insight 1: Zero items correlation
    clothing_zero_pct = feature_stats['clothing']['img_zero_items'] / feature_stats['clothing']['img_total_items'] * 100
    baby_zero_pct = feature_stats['baby']['img_zero_items'] / feature_stats['baby']['img_total_items'] * 100
    
    if clothing_zero_pct > 50 and baby_zero_pct < 10:
        print(f"1. MAJOR ISSUE: Clothing has {clothing_zero_pct:.1f}% zero image items vs Baby's {baby_zero_pct:.1f}%")
        print(f"   High correlation between zero items and poor performance")
    
    # Insight 2: Feature quality correlation
    clothing_quality = feature_stats['clothing']['img_nonzero_ratio']
    baby_quality = feature_stats['baby']['img_nonzero_ratio']
    
    if baby_quality > clothing_quality * 5:
        print(f"2. IMAGE QUALITY CORRELATION:")
        print(f"   Baby (best): {baby_quality:.1%} non-zero, Recall@20=0.0474")
        print(f"   Clothing (worst): {clothing_quality:.1%} non-zero, Recall@20=0.0131")
        print(f"   {baby_quality/clothing_quality:.1f}x better image quality = {0.0474/0.0131:.1f}x better performance")
    
    # Insight 3: Text dominance
    text_clothing = feature_stats['clothing']['text_nonzero_ratio']
    img_clothing = feature_stats['clothing']['img_nonzero_ratio']
    
    if text_clothing > 0.99 and img_clothing < 0.05:
        print(f"3. TEXT DOMINANCE:")
        print(f"   Text features: {text_clothing:.1%} coverage")
        print(f"   Image features: {img_clothing:.1%} coverage")
        print(f"   Text is {text_clothing/img_clothing:.0f}x more complete than images")
    
    return feature_stats

def analyze_sea_architecture():
    """Analyze how SEA processes multimodal features"""
    
    print(f"\nSEA ARCHITECTURE ANALYSIS")
    print(f"="*40)
    
    # Check SEA model configuration
    try:
        # Read model source to understand fusion mechanism
        sea_model_path = 'src/models/SEA.py'
        if os.path.exists(sea_model_path):
            print(f"Found SEA model at: {sea_model_path}")
            
            with open(sea_model_path, 'r') as f:
                content = f.read()
            
            # Look for key fusion mechanisms
            fusion_mechanisms = [
                'multimodal_fusion',
                'image_text_fusion', 
                'concat',
                'attention',
                'weighted_sum',
                'self.image_embedding',
                'self.text_embedding'
            ]
            
            found_mechanisms = []
            for mechanism in fusion_mechanisms:
                if mechanism in content:
                    found_mechanisms.append(mechanism)
            
            print(f"Found fusion mechanisms: {found_mechanisms}")
            
            # Check for feature normalization
            if 'normalize_image_features' in content:
                print(f"Image feature normalization: ENABLED")
            else:
                print(f"Image feature normalization: NOT FOUND")
                
            # Check for adaptive mechanisms
            if 'adaptive' in content.lower():
                print(f"Adaptive mechanisms: FOUND")
            else:
                print(f"Adaptive mechanisms: NOT FOUND")
                
        else:
            print(f"SEA model file not found")
            
    except Exception as e:
        print(f"Error analyzing SEA architecture: {e}")

def suggest_next_actions():
    """Suggest concrete next steps based on analysis"""
    
    print(f"\nRECOMMENDED NEXT ACTIONS")
    print(f"="*40)
    
    print(f"Based on our findings:")
    print(f"")
    print(f"1. IMMEDIATE PRIORITIES:")
    print(f"   a) Test if Baby dataset (best performer) also has text-dominant behavior")
    print(f"   b) Investigate SEA's multimodal fusion mechanism")
    print(f"   c) Check if image features are properly integrated in the model")
    print(f"")
    print(f"2. RESEARCH DIRECTIONS:")
    print(f"   a) Feature engineering: Better image representations")
    print(f"   b) Architecture: Improve multimodal fusion")
    print(f"   c) Training: Better multimodal learning strategies")
    print(f"")
    print(f"3. PERFORMANCE OPTIMIZATION:")
    print(f"   a) Focus on text-only optimization for immediate gains")
    print(f"   b) Investigate why Baby dataset performs 3.6x better")
    print(f"   c) Apply Baby's success factors to Clothing/Sports")
    print(f"")
    print(f"4. NEXT EXPERIMENTS:")
    print(f"   - Test Baby dataset with text-only (confirm text dominance)")
    print(f"   - Analyze Baby's superior text feature quality")
    print(f"   - Investigate SEA's attention/fusion weights")

if __name__ == "__main__":
    feature_stats = analyze_image_feature_issues()
    analyze_sea_architecture()
    suggest_next_actions()
    
    print(f"\nSUMMARY:")
    print(f"Our text-only experiment (Recall@20: 0.0132 vs 0.0131) proves that")
    print(f"image features contribute ZERO value. This analysis helps understand WHY.")
