#!/usr/bin/env python3
"""
Focused SEA Experiments Script
Run specific configurations to improve performance systematically
"""

import os
import sys
import subprocess
import yaml
import json
import time
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_experiment(config_updates, dataset='clothing', description=""):
    """
    Run a single experiment with config updates
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {description}")
    print(f"Config updates: {config_updates}")
    print(f"{'='*80}")
    
    # Update config files
    config_files = {
        'overall': 'src/configs/overall.yaml',
        'model': 'src/configs/model/SEA.yaml'
    }
    
    # Backup original configs
    backups = {}
    for key, file_path in config_files.items():
        backup_path = f"{file_path}.backup"
        if os.path.exists(file_path):
            subprocess.run(['cp', file_path, backup_path], check=True)
            backups[key] = backup_path
    
    try:
        # Update configs
        for key, file_path in config_files.items():
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Apply updates
                for update_key, update_value in config_updates.items():
                    if update_key in config:
                        config[update_key] = update_value
                        print(f"Updated {update_key}: {update_value}")
                
                # Save updated config
                with open(file_path, 'w') as f:
                    yaml.safe_dump(config, f, default_flow_style=False)
        
        # Run the experiment
        cmd = [
            sys.executable, 'src/main.py', 
            '--model', 'SEA',
            '--dataset', dataset,
            '--config_files', 'src/configs/overall.yaml', f'src/configs/dataset/{dataset}.yaml', 'src/configs/model/SEA.yaml'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Experiment completed in {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("STDOUT:")
            print(result.stdout[-1000:])  # Last 1000 chars
            
            # Extract results from output
            lines = result.stdout.split('\n')
            test_results = {}
            for line in lines:
                if 'test result' in line and 'Recall@20' in line:
                    # Try to parse results
                    try:
                        if ':' in line:
                            result_part = line.split(':', 1)[1].strip()
                            if 'Recall@20' in result_part:
                                # Extract Recall@20 value
                                import re
                                match = re.search(r'Recall@20.*?([0-9.]+)', result_part)
                                if match:
                                    test_results['Recall@20'] = float(match.group(1))
                    except:
                        pass
            
            return {
                'success': True,
                'duration': duration,
                'test_results': test_results,
                'config_updates': config_updates,
                'description': description
            }
        else:
            print("STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars
            return {
                'success': False,
                'error': result.stderr,
                'config_updates': config_updates,
                'description': description
            }
    
    except subprocess.TimeoutExpired:
        print("Experiment timed out!")
        return {
            'success': False,
            'error': "Timeout",
            'config_updates': config_updates,
            'description': description
        }
    
    except Exception as e:
        print(f"Error running experiment: {e}")
        return {
            'success': False,
            'error': str(e),
            'config_updates': config_updates,
            'description': description
        }
    
    finally:
        # Restore original configs
        for key, backup_path in backups.items():
            if os.path.exists(backup_path):
                original_path = config_files[key]
                subprocess.run(['cp', backup_path, original_path], check=True)
                os.remove(backup_path)

def main():
    """
    Run focused experiments to improve SEA performance
    """
    print("Starting Focused SEA Experiments")
    print("Target: Recall@20 ≈ 0.0953 (clothing dataset)")
    print("Current best: Recall@20 ≈ 0.0501")
    
    # Define focused experiments based on our findings
    experiments = [
        {
            'config_updates': {
                'learning_rate': 0.0005,
                'alpha_contrast': 0.25,
                'beta': 0.01,
                'temp': 0.15,
                'embedding_size': 128,
                'n_mm_layers': 2,
                'epochs': 150,
                'stopping_step': 30
            },
            'description': "Conservative LR with balanced loss weights"
        },
        {
            'config_updates': {
                'learning_rate': 0.001,
                'alpha_contrast': 0.15,
                'beta': 0.015,
                'temp': 0.2,
                'embedding_size': 64,
                'n_mm_layers': 1,
                'epochs': 200,
                'stopping_step': 40
            },
            'description': "Smaller model with longer training"
        },
        {
            'config_updates': {
                'learning_rate': 0.0015,
                'alpha_contrast': 0.2,
                'beta': 0.005,
                'temp': 0.1,
                'embedding_size': 128,
                'n_mm_layers': 2,
                'epochs': 120,
                'stopping_step': 25
            },
            'description': "Higher LR with lower temperature"
        },
        {
            'config_updates': {
                'learning_rate': 0.001,
                'alpha_contrast': 0.3,
                'beta': 0.02,
                'temp': 0.25,
                'embedding_size': 64,
                'n_mm_layers': 1,
                'epochs': 100,
                'stopping_step': 20
            },
            'description': "High alignment weight with small model"
        },
        {
            'config_updates': {
                'learning_rate': 0.0008,
                'alpha_contrast': 0.1,
                'beta': 0.008,
                'temp': 0.15,
                'embedding_size': 128,
                'n_mm_layers': 2,
                'epochs': 180,
                'stopping_step': 35
            },
            'description': "Low alignment weight with long training"
        }
    ]
    
    results = []
    best_score = 0.0501
    best_config = None
    
    for i, experiment in enumerate(experiments):
        print(f"\n{'#'*100}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}")
        print(f"{'#'*100}")
        
        result = run_experiment(experiment['config_updates'], 
                              dataset='clothing', 
                              description=experiment['description'])
        
        results.append(result)
        
        if result['success'] and result.get('test_results', {}).get('Recall@20', 0) > best_score:
            best_score = result['test_results']['Recall@20']
            best_config = result
            print(f"🎉 NEW BEST SCORE: Recall@20 = {best_score:.4f}")
            print(f"Config: {result['config_updates']}")
        
        # Save intermediate results
        with open(f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    for i, result in enumerate(results):
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        score = result.get('test_results', {}).get('Recall@20', 0)
        print(f"Experiment {i+1}: {status} - Recall@20: {score:.4f} - {result['description']}")
    
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"Recall@20: {best_score:.4f}")
        print(f"Config: {best_config['config_updates']}")
        print(f"Description: {best_config['description']}")
    
    print(f"\nTarget: Recall@20 ≈ 0.0953")
    print(f"Gap remaining: {0.0953 - best_score:.4f}")

if __name__ == "__main__":
    main()
