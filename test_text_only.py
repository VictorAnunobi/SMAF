#!/usr/bin/env python3
"""
Test SEA performance with TEXT ONLY features (zero out image features)
This will help us understand if image features contribute meaningfully
"""

import numpy as np
import shutil
import os

def main():
    print("🔧 TESTING TEXT-ONLY PERFORMANCE")
    print("="*50)
    
    # Backup original image features
    print("📦 Backing up original image features...")
    shutil.copy('data/clothing/image_feat.npy', 'data/clothing/image_feat_original_backup.npy')
    
    # Zero out image features  
    print("🚫 Zeroing out image features...")
    img_feat = np.load('data/clothing/image_feat.npy')
    print(f"   Original shape: {img_feat.shape}")
    print(f"   Original non-zero ratio: {(img_feat != 0).mean():.4f}")
    
    # Create zero features
    zero_img_feat = np.zeros_like(img_feat)
    np.save('data/clothing/image_feat.npy', zero_img_feat)
    
    print(f"   New non-zero ratio: {(zero_img_feat != 0).mean():.4f}")
    print("✅ Image features zeroed out")
    
    print("\n🚀 NOW RUN: cd src && python main.py --dataset clothing")
    print("📝 Record the Recall@20 and NDCG@20 results")
    print("🔄 Then run this script again with --restore to restore original features")

def restore():
    print("🔄 RESTORING ORIGINAL IMAGE FEATURES")
    print("="*40)
    
    if os.path.exists('data/clothing/image_feat_original_backup.npy'):
        shutil.copy('data/clothing/image_feat_original_backup.npy', 'data/clothing/image_feat.npy')
        print("✅ Original image features restored")
        
        # Verify restoration
        img_feat = np.load('data/clothing/image_feat.npy')
        print(f"   Restored non-zero ratio: {(img_feat != 0).mean():.4f}")
    else:
        print("❌ Backup file not found!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--restore':
        restore()
    else:
        main()
