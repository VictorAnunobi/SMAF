#!/usr/bin/env python3
"""
Test Baby dataset with TEXT ONLY features (zero out image features)
Baby has the best performance (Recall@20: 0.0474) and excellent image quality (20.8% non-zero)
If even Baby is text-dominated, it confirms architectural issues with SEA
"""

import numpy as np
import shutil
import os

def main():
    print("🔬 TESTING BABY DATASET: TEXT-ONLY PERFORMANCE")
    print("="*60)
    print("Baby dataset has:")
    print("  • Best performance: Recall@20=0.0474 (only 2x gap to paper)")
    print("  • Excellent image quality: 20.8% non-zero features") 
    print("  • Zero broken image items: 0/7050 (0%)")
    print("  • If even Baby is text-dominated → SEA architectural issue")
    print("="*60)
    
    # Backup original image features
    print("📦 Backing up original Baby image features...")
    shutil.copy('data/baby/image_feat.npy', 'data/baby/image_feat_original_backup.npy')
    
    # Zero out image features  
    print("🚫 Zeroing out Baby image features...")
    img_feat = np.load('data/baby/image_feat.npy')
    print(f"   Original shape: {img_feat.shape}")
    print(f"   Original non-zero ratio: {(img_feat != 0).mean():.4f}")
    print(f"   Original zero items: {(img_feat == 0).all(axis=1).sum()}/{img_feat.shape[0]}")
    
    # Create zero features
    zero_img_feat = np.zeros_like(img_feat)
    np.save('data/baby/image_feat.npy', zero_img_feat)
    
    print(f"   New non-zero ratio: {(zero_img_feat != 0).mean():.4f}")
    print("✅ Baby image features zeroed out")
    
    print("\n🚀 NOW RUN: cd src && python main.py --dataset baby")
    print("📊 Expected if text-dominated: Recall@20 ≈ 0.0474 (same as multimodal)")
    print("📊 Expected if image helps: Recall@20 << 0.0474 (significant drop)")
    print("🔄 Then run this script again with --restore to restore original features")

def restore():
    print("🔄 RESTORING ORIGINAL BABY IMAGE FEATURES")
    print("="*45)
    
    if os.path.exists('data/baby/image_feat_original_backup.npy'):
        shutil.copy('data/baby/image_feat_original_backup.npy', 'data/baby/image_feat.npy')
        print("✅ Original Baby image features restored")
        
        # Verify restoration
        img_feat = np.load('data/baby/image_feat.npy')
        print(f"   Restored non-zero ratio: {(img_feat != 0).mean():.4f}")
        print(f"   Restored zero items: {(img_feat == 0).all(axis=1).sum()}/{img_feat.shape[0]}")
    else:
        print("❌ Backup file not found!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--restore':
        restore()
    else:
        main()
