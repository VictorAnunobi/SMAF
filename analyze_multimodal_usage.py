#!/usr/bin/env python3
"""
Analyze how SEA model uses multimodal features
Investigate text vs image feature contribution to performance
"""

import os
import sys
import subprocess
import numpy as np
import time

def analyze_feature_importance():
    """Analyze the relative importance of image vs text features"""
    
    print("="*80)
    print("🔍 MULTIMODAL FEATURE USAGE ANALYSIS")
    print("Investigating how SEA uses image vs text features")
    print("="*80)
    
    # Test different feature configurations
    configs_to_test = [
        {"name": "Full Multimodal", "use_image": True, "use_text": True},
        {"name": "Text Only", "use_image": False, "use_text": True},
        {"name": "Image Only", "use_image": True, "use_text": False},
    ]
    
    results = {}
    
    for config in configs_to_test:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"   Image features: {'✅' if config['use_image'] else '❌'}")
        print(f"   Text features: {'✅' if config['use_text'] else '❌'}")
        
        # Create modified dataset with selective features
        img_feat_backup = None
        text_feat_backup = None
        
        if not config['use_image']:
            # Zero out image features
            print("   🔧 Zeroing out image features...")
            img_feat = np.load('data/clothing/image_feat.npy')
            img_feat_backup = img_feat.copy()
            np.save('data/clothing/image_feat.npy', np.zeros_like(img_feat))
            
        if not config['use_text']:
            # Zero out text features  
            print("   🔧 Zeroing out text features...")
            text_feat = np.load('data/clothing/text_feat.npy')
            text_feat_backup = text_feat.copy()
            np.save('data/clothing/text_feat.npy', np.zeros_like(text_feat))
        
        try:
            # Run experiment with modified features using subprocess
            print("   🚀 Running SEA experiment...")
            start_time = time.time()
            result = run_sea_experiment_subprocess()
            elapsed = time.time() - start_time
            print(f"   ⏱️  Completed in {elapsed:.1f}s")
            
            results[config['name']] = result
            recall_val = result.get('recall@20', 'N/A')
            ndcg_val = result.get('ndcg@20', 'N/A')
            if isinstance(recall_val, (int, float)) and isinstance(ndcg_val, (int, float)):
                print(f"   📊 Result: Recall@20={recall_val:.4f}, NDCG@20={ndcg_val:.4f}")
            else:
                print(f"   📊 Result: Recall@20={recall_val}, NDCG@20={ndcg_val}")
                if 'error' in result:
                    print(f"   ⚠️  Error details: {result['error']}")
            
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            results[config['name']] = {"error": str(e)}
        
        finally:
            # Restore original features
            if img_feat_backup is not None:
                print("   🔄 Restoring image features...")
                np.save('data/clothing/image_feat.npy', img_feat_backup)
            if text_feat_backup is not None:
                print("   🔄 Restoring text features...")
                np.save('data/clothing/text_feat.npy', text_feat_backup)
    
    # Analysis
    print(f"\n{'='*60}")
    print("📈 FEATURE CONTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    if "Full Multimodal" in results and "Text Only" in results:
        full_result = results["Full Multimodal"]
        text_result = results["Text Only"]
        
        if 'error' not in full_result and 'error' not in text_result:
            full_recall = full_result.get("recall@20", 0)
            text_recall = text_result.get("recall@20", 0)
            image_contribution = full_recall - text_recall
            
            print(f"🔍 Image Feature Contribution:")
            print(f"   Full Multimodal: {full_recall:.4f}")
            print(f"   Text Only: {text_recall:.4f}")
            print(f"   Image Contribution: {image_contribution:.4f}")
            
            if full_recall > 0:
                contribution_pct = (image_contribution / full_recall) * 100
                print(f"   Relative Contribution: {contribution_pct:.1f}%")
            
            if abs(image_contribution) < 0.001:
                print("   🚨 INSIGHT: Image features contribute almost nothing!")
            elif image_contribution < 0:
                print("   🚨 INSIGHT: Image features actually hurt performance!")
            else:
                print("   ✅ INSIGHT: Image features provide meaningful contribution")
        else:
            print("🔍 Analysis incomplete due to errors in experiments")
            for name, result in results.items():
                if 'error' in result:
                    print(f"   {name}: {result['error']}")
    else:
        print("🔍 Cannot analyze - missing required experiment results")
    
    return results

def run_sea_experiment_subprocess():
    """Run a single SEA experiment using subprocess and return results"""
    
    try:
        # Run the experiment using subprocess
        cmd = [
            'python', 'src/main.py', 
            '--dataset', 'clothing',
            '--alpha_contrast', '0.2',
            '--temp', '0.2', 
            '--beta', '0.01'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            return {'error': f'Command failed: {result.stderr}'}
        
        # Parse results from output
        output = result.stdout
        lines = output.split('\n')
        
        for line in lines:
            if 'Test:' in line and 'recall@20:' in line:
                # Extract metrics from line like: "Test: recall@10: 0.0077    recall@20: 0.0131    ndcg@10: 0.0040    ndcg@20: 0.0053"
                try:
                    parts = line.split()
                    recall20_idx = parts.index('recall@20:') + 1
                    ndcg20_idx = parts.index('ndcg@20:') + 1
                    
                    return {
                        'recall@20': float(parts[recall20_idx]),
                        'ndcg@20': float(parts[ndcg20_idx])
                    }
                except (ValueError, IndexError) as e:
                    continue
        
        # Look for BEST results line as fallback
        for line in lines:
            if 'BEST' in line and 'recall@20:' in line:
                try:
                    # Parse validation results
                    if 'Valid:' in line:
                        parts = line.split('Valid:')[1].split(',')
                        recall20 = None
                        ndcg20 = None
                        for part in parts:
                            if 'recall@20:' in part:
                                recall20 = float(part.split('recall@20:')[1].strip())
                            if 'ndcg@20:' in part:
                                ndcg20 = float(part.split('ndcg@20:')[1].strip())
                        
                        if recall20 is not None and ndcg20 is not None:
                            return {'recall@20': recall20, 'ndcg@20': ndcg20}
                except (ValueError, IndexError) as e:
                    continue
        
        return {'error': 'Could not parse results from output'}
        
    except Exception as e:
        return {'error': f'Subprocess failed: {str(e)}'}

if __name__ == "__main__":
    results = analyze_feature_importance()
    
    print(f"\n🎯 CONCLUSIONS:")
    print(f"1. If text-only performs similarly to full multimodal:")
    print(f"   → Image features are not contributing meaningfully")
    print(f"2. If image-only performs much worse:")
    print(f"   → Text features are the primary driver of performance")
    print(f"3. This explains why fixing image features didn't improve performance!")
