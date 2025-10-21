#!/usr/bin/env python3
"""
Fix Clothing Dataset Image Feature Corruption

The analysis revealed that 91% of clothing items have completely zero image features,
which explains the poor performance (7.3x gap vs 2x for baby).

This script implements several strategies to fix the missing image features.
"""

import numpy as np
import os
import shutil
from datetime import datetime

def analyze_feature_corruption(dataset_name):
    """Analyze the extent of feature corruption"""
    img_file = f"data/{dataset_name}/image_feat.npy"
    
    if not os.path.exists(img_file):
        print(f"❌ {img_file} not found")
        return None
    
    img_feat = np.load(img_file)
    
    # Analyze corruption
    zero_rows = (img_feat == 0).all(axis=1)
    zero_count = zero_rows.sum()
    total_items = img_feat.shape[0]
    
    print(f"\n📊 {dataset_name.upper()} IMAGE FEATURE ANALYSIS:")
    print(f"   Total items: {total_items}")
    print(f"   Items with all-zero features: {zero_count} ({zero_count/total_items*100:.1f}%)")
    print(f"   Items with valid features: {total_items - zero_count} ({(total_items-zero_count)/total_items*100:.1f}%)")
    
    if zero_count > 0:
        # Analyze the valid features
        valid_features = img_feat[~zero_rows]
        print(f"   Valid features stats:")
        print(f"     Mean: {valid_features.mean():.6f}")
        print(f"     Std: {valid_features.std():.6f}")
        print(f"     Min/Max: [{valid_features.min():.4f}, {valid_features.max():.4f}]")
        print(f"     Non-zero ratio: {(valid_features != 0).mean():.4f}")
    
    return {
        'features': img_feat,
        'zero_mask': zero_rows,
        'zero_count': zero_count,
        'total_items': total_items,
        'valid_features': img_feat[~zero_rows] if zero_count < total_items else None
    }

def strategy_1_mean_imputation(features_info, dataset_name):
    """Strategy 1: Replace zero features with mean of valid features"""
    if features_info['valid_features'] is None:
        print("❌ No valid features found for mean imputation")
        return None
    
    print(f"\n🔧 STRATEGY 1: Mean Imputation for {dataset_name}")
    
    # Create backup
    backup_name = f"data/{dataset_name}/image_feat_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    shutil.copy(f"data/{dataset_name}/image_feat.npy", backup_name)
    print(f"   ✅ Backup created: {backup_name}")
    
    # Calculate mean of valid features
    mean_feature = features_info['valid_features'].mean(axis=0)
    
    # Replace zero rows with mean
    fixed_features = features_info['features'].copy()
    fixed_features[features_info['zero_mask']] = mean_feature
    
    # Save fixed features
    output_file = f"data/{dataset_name}/image_feat_mean_imputed.npy"
    np.save(output_file, fixed_features)
    
    print(f"   ✅ Fixed features saved: {output_file}")
    print(f"   📊 Replaced {features_info['zero_count']} zero rows with mean feature")
    
    # Verify fix
    verify_features = np.load(output_file)
    zero_rows_after = (verify_features == 0).all(axis=1).sum()
    print(f"   ✅ Verification: {zero_rows_after} zero rows remaining (should be 0)")
    
    return output_file

def strategy_2_random_sampling(features_info, dataset_name):
    """Strategy 2: Replace zero features with random sampling from valid features"""
    if features_info['valid_features'] is None:
        print("❌ No valid features found for random sampling")
        return None
    
    print(f"\n🔧 STRATEGY 2: Random Sampling for {dataset_name}")
    
    # Create backup if not already done
    backup_name = f"data/{dataset_name}/image_feat_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    if not os.path.exists(backup_name):
        shutil.copy(f"data/{dataset_name}/image_feat.npy", backup_name)
        print(f"   ✅ Backup created: {backup_name}")
    
    # Replace zero rows with random valid features
    fixed_features = features_info['features'].copy()
    num_zeros = features_info['zero_count']
    
    if num_zeros > 0:
        # Randomly sample from valid features
        valid_indices = np.random.choice(
            features_info['valid_features'].shape[0], 
            size=num_zeros, 
            replace=True
        )
        fixed_features[features_info['zero_mask']] = features_info['valid_features'][valid_indices]
    
    # Save fixed features
    output_file = f"data/{dataset_name}/image_feat_random_sampled.npy"
    np.save(output_file, fixed_features)
    
    print(f"   ✅ Fixed features saved: {output_file}")
    print(f"   📊 Replaced {features_info['zero_count']} zero rows with random samples")
    
    # Verify fix
    verify_features = np.load(output_file)
    zero_rows_after = (verify_features == 0).all(axis=1).sum()
    print(f"   ✅ Verification: {zero_rows_after} zero rows remaining (should be 0)")
    
    return output_file

def strategy_3_learned_embeddings(features_info, dataset_name):
    """Strategy 3: Use small random embeddings (will be learned during training)"""
    print(f"\n🔧 STRATEGY 3: Learned Embeddings for {dataset_name}")
    
    # Create backup if not already done
    backup_name = f"data/{dataset_name}/image_feat_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    if not os.path.exists(backup_name):
        shutil.copy(f"data/{dataset_name}/image_feat.npy", backup_name)
        print(f"   ✅ Backup created: {backup_name}")
    
    # Replace zero rows with small random values (will be learned)
    fixed_features = features_info['features'].copy()
    num_zeros = features_info['zero_count']
    feature_dim = features_info['features'].shape[1]
    
    if num_zeros > 0:
        # Small random initialization (similar to Xavier/He initialization)
        np.random.seed(42)  # For reproducibility
        small_random = np.random.normal(0, 0.01, size=(num_zeros, feature_dim))
        fixed_features[features_info['zero_mask']] = small_random
    
    # Save fixed features
    output_file = f"data/{dataset_name}/image_feat_learned_init.npy"
    np.save(output_file, fixed_features)
    
    print(f"   ✅ Fixed features saved: {output_file}")
    print(f"   📊 Replaced {features_info['zero_count']} zero rows with learnable embeddings")
    
    # Verify fix
    verify_features = np.load(output_file)
    zero_rows_after = (verify_features == 0).all(axis=1).sum()
    print(f"   ✅ Verification: {zero_rows_after} zero rows remaining (should be 0)")
    
    return output_file

def main():
    """Main function to fix clothing dataset image features"""
    print("=" * 80)
    print("🔧 CLOTHING DATASET IMAGE FEATURE REPAIR")
    print("Fixing 91% zero image features that cause poor performance")
    print("=" * 80)
    
    # Analyze the corruption
    clothing_info = analyze_feature_corruption('clothing')
    
    if clothing_info is None:
        print("❌ Could not analyze clothing features")
        return
    
    if clothing_info['zero_count'] == 0:
        print("✅ No zero features found - nothing to fix!")
        return
    
    # Apply all three strategies
    strategies = []
    
    # Strategy 1: Mean imputation
    mean_file = strategy_1_mean_imputation(clothing_info, 'clothing')
    if mean_file:
        strategies.append(('mean_imputed', mean_file))
    
    # Strategy 2: Random sampling
    random_file = strategy_2_random_sampling(clothing_info, 'clothing')
    if random_file:
        strategies.append(('random_sampled', random_file))
    
    # Strategy 3: Learned embeddings
    learned_file = strategy_3_learned_embeddings(clothing_info, 'clothing')
    if learned_file:
        strategies.append(('learned_init', learned_file))
    
    print(f"\n{'=' * 60}")
    print("🎯 REPAIR COMPLETE - NEXT STEPS")
    print(f"{'=' * 60}")
    
    print(f"\n✅ Created {len(strategies)} fixed versions:")
    for name, filepath in strategies:
        print(f"   - {name}: {filepath}")
    
    print(f"\n🚀 RECOMMENDED TESTING ORDER:")
    print(f"1. Test learned_init version first (most principled)")
    print(f"2. Test mean_imputed version if learned_init doesn't work")
    print(f"3. Test random_sampled version as fallback")
    
    print(f"\n📝 TO TEST A VERSION:")
    print(f"   # Backup current version")
    print(f"   cp data/clothing/image_feat.npy data/clothing/image_feat_broken.npy")
    print(f"   ")
    print(f"   # Use learned_init version")
    print(f"   cp data/clothing/image_feat_learned_init.npy data/clothing/image_feat.npy")
    print(f"   ")
    print(f"   # Run experiment")
    print(f"   cd src && python main.py --dataset clothing")
    print(f"   ")
    print(f"   # Restore if needed")
    print(f"   cp data/clothing/image_feat_broken.npy data/clothing/image_feat.npy")

if __name__ == "__main__":
    main()
