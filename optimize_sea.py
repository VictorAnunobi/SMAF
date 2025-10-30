#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from itertools import product

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.quick_start import quick_start


def run_experiment(config_overrides, dataset_name='clothing', max_epochs=50):
    """
    Run a single experiment with given configuration
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config_overrides}")
    print(f"{'='*60}")
    
    # Create args object with overrides
    class Args:
        def __init__(self):
            self.model = 'SEA'
            self.dataset = dataset_name
            self.config_files = ['src/configs/overall.yaml', f'src/configs/dataset/{dataset_name}.yaml', 'src/configs/model/SEA.yaml']
            self.saved = True
            self.epochs = max_epochs
            self.valid_metric = 'Recall@20'
            
            # Apply config overrides as attributes
            for key, value in config_overrides.items():
                setattr(self, key, value)
    
    args = Args()
    
    try:
        # Run the experiment
        run_recbole(model=args.model, dataset=args.dataset, config_file_list=args.config_files, config_dict=None, saved=args.saved)
        return True
    except Exception as e:
        print(f"Experiment failed: {e}")
        return False


def run_recbole(model, dataset, config_file_list, config_dict=None, saved=True):
    """Simplified version of recbole run"""
    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.trainer import Trainer
        from recbole.utils import init_logger, get_model, init_seed, set_color
        from logging import getLogger
    except ImportError:
        print("RecBole not available in this environment - this is a planning script")
        return None
    
    # Get configurations
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # Logger initialization
    init_logger(config)
    logger = getLogger()
    
    logger.info(f"SEA Optimization for {dataset}")
    logger.info(f"Config: {config}")
    
    # Dataset loading and preparation
    dataset = create_dataset(config)
    logger.info(dataset)
    
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # Trainer loading and initialization
    trainer = Trainer(config, model)
    
    # Training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    
    # Testing
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    
    # Log results
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def main():
    """
    Main optimization loop
    """
    print("Starting SEA Hyperparameter Optimization")
    print("Target: Recall@20 ≈ 0.0953 (clothing)")
    
    # Define hyperparameter grid for systematic search
    # Focus on stable, moderate model sizes that have shown good results
    param_grid = {
        'learning_rate': [0.0005, 0.001, 0.0015],  # Conservative learning rates
        'alpha_contrast': [0.1, 0.15, 0.2, 0.25, 0.3], # Alignment loss weight
        'beta': [0.005, 0.01, 0.015, 0.02],  # Distancing loss weight
        'temp': [0.1, 0.15, 0.2, 0.25, 0.3],  # Temperature scaling
        'dropout': [0.0, 0.1, 0.2],  # Regularization
        'embedding_size': [64, 128],  # Moderate model capacity only
        'n_mm_layers': [1, 2],  # Avoid deep item graphs
    }
    
    # Start with best known configuration
    best_config = {
        'learning_rate': 0.001,
        'alpha_contrast': 0.2,
        'beta': 0.005,
        'temp': 0.2,
        'dropout': 0.1,
        'embedding_size': 128,  # Moderate size
        'n_mm_layers': 2,  # Shallow item graph
        'seed': 999,
        'reg_weight': 0.0,
        'stopping_step': 20,
        'epochs': 100
    }
    
    best_score = 0.0501  # Current best Recall@20 (clothing dataset)
    
    # Phase 1: Optimize key hyperparameters one by one
    print(f"\n{'='*80}")
    print("PHASE 1: Individual Hyperparameter Optimization")
    print(f"{'='*80}")
    
    for param_name in ['learning_rate', 'alpha_contrast', 'beta', 'temp']:
        print(f"\nOptimizing {param_name}...")
        
        for value in param_grid[param_name]:
            config = best_config.copy()
            config[param_name] = value
            
            print(f"Testing {param_name}={value}")
            
            # For this simplified version, we'll just print what would be tested
            # In practice, you'd call run_experiment(config) here
            print(f"Config: {config}")
            # result = run_experiment(config)
            # if result and result.get('test_result', {}).get('Recall@20', 0) > best_score:
            #     best_score = result['test_result']['Recall@20']
            #     best_config[param_name] = value
            #     print(f"New best {param_name}: {value} (Recall@20: {best_score:.4f})")
    
    # Phase 2: Loss balance optimization with moderate architectures
    print(f"\n{'='*80}")
    print("PHASE 2: Loss Balance and Architecture Optimization")
    print(f"{'='*80}")
    
    # Focus on stable, moderate architectures with better loss balance
    architecture_configs = [
        # Test smaller model first (more stable)
        {'embedding_size': 64, 'n_mm_layers': 1, 'learning_rate': 0.001},
        {'embedding_size': 64, 'n_mm_layers': 2, 'learning_rate': 0.0015},
        
        # Current best architecture with different loss balances
        {'embedding_size': 128, 'n_mm_layers': 1, 'alpha_contrast': 0.25, 'beta': 0.01},
        {'embedding_size': 128, 'n_mm_layers': 2, 'alpha_contrast': 0.15, 'beta': 0.015},
        
        # Focus on better alignment/distancing balance
        {'alpha_contrast': 0.3, 'beta': 0.02, 'temp': 0.15},
        {'alpha_contrast': 0.15, 'beta': 0.008, 'temp': 0.25},
        
        # Conservative regularization
        {'dropout': 0.0, 'reg_weight': 0.0001},
        {'dropout': 0.1, 'reg_weight': 0.0},
    ]
    
    for arch_config in architecture_configs:
        config = best_config.copy()
        config.update(arch_config)
        print(f"Testing architecture: {arch_config}")
        print(f"Full config: {config}")
        # result = run_experiment(config)
    
    # Phase 3: Final fine-tuning with best found parameters
    print(f"\n{'='*80}")
    print("PHASE 3: Final Fine-tuning")
    print(f"{'='*80}")
    
    # Test a few final combinations based on stable, moderate architectures
    final_configs = [
        # Conservative configuration with longer training
        {
            'learning_rate': 0.001,
            'alpha_contrast': 0.1,  # Lower alignment weight
            'beta': 0.01,           # Higher distancing weight
            'temp': 0.2,
            'dropout': 0.1,
            'embedding_size': 128,
            'n_mm_layers': 2,
            'epochs': 150,          # Longer training
            'stopping_step': 30,
        },
        
        # Smaller model with aggressive loss balance
        {
            'learning_rate': 0.0015,
            'alpha_contrast': 0.15,
            'beta': 0.005,
            'temp': 0.1,           # Lower temperature
            'dropout': 0.0,        # No dropout
            'embedding_size': 64,   # Smaller, more stable
            'n_mm_layers': 1,       # Simpler architecture
            'epochs': 200,
            'stopping_step': 40,
        },
        
        # Best balance approach
        {
            'learning_rate': 0.0012,
            'alpha_contrast': 0.2,
            'beta': 0.008,
            'temp': 0.15,
            'dropout': 0.05,
            'embedding_size': 128,  # Moderate size
            'n_mm_layers': 2,       # Moderate depth
            'epochs': 120,
            'stopping_step': 25,
        }
    ]
    
    for i, config in enumerate(final_configs, 1):
        print(f"\nFinal Configuration {i}:")
        final_config = best_config.copy()
        final_config.update(config)
        print(f"Config: {final_config}")
        # result = run_experiment(final_config, max_epochs=config.get('epochs', 100))
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print("To actually run these experiments, uncomment the run_experiment() calls")
    print("and ensure your environment is properly set up.")


if __name__ == "__main__":
    # RESCUE CONFIGURATION: Address critical issues from July 9th run
    # Issues found: SVD not working, model too large (256/3 layers), loss explosion
    rescue_config = {
        'learning_rate': 0.001,      # Conservative LR to prevent explosion
        'alpha_contrast': 0.1,       # Lower contrast weight  
        'beta': 0.01,                # Higher distancing weight
        'temp': 0.2,                 # Higher temperature for stability
        'dropout': 0.1,              # Some regularization
        'embedding_size': 64,        # MUCH smaller (was 256)
        'n_mm_layers': 1,            # MUCH simpler (was 3)
        'epochs': 100,
        'stopping_step': 25,         # More patience
        'seed': 999,
        'reg_weight': 0.0001,        # Small L2 regularization
    }
    
    # More conservative backup config
    ultra_conservative_config = {
        'learning_rate': 0.0005,     # Very conservative
        'alpha_contrast': 0.05,      # Very low contrast
        'beta': 0.02,                # Higher distancing  
        'temp': 0.3,                 # High temperature
        'dropout': 0.2,              # More dropout
        'embedding_size': 32,        # Ultra small
        'n_mm_layers': 1,            # Simplest possible
        'epochs': 150,
        'stopping_step': 40,
        'seed': 999,
        'reg_weight': 0.001,         # More regularization
    }
    
    # For now, just show the optimization plan
    main()
    
    # Display critical fixes needed
    print(f"\n{'='*80}")
    print("CRITICAL FIXES NEEDED based on July 9th log analysis:")
    print(f"{'='*80}")
    print("1. SVD COMPLETION NOT WORKING - Need to debug this first!")
    print("2. Model too large (256 emb, 3 layers) → Use 64 emb, 1 layer")
    print("3. Loss explosion (256→816) → Lower learning rate")
    print("4. Loss plateau → Better regularization")
    print()
    print("Next steps:")
    print("a) Test SVD completion fix")
    print("b) Run rescue_config (64 emb, 1 layer)")
    print("c) If still poor, try ultra_conservative_config (32 emb)")
    
    print(f"\n{'='*60}")
    print("RESCUE CONFIGURATION")
    print(f"{'='*60}")
    for key, value in rescue_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")
    print("ULTRA CONSERVATIVE BACKUP")
    print(f"{'='*60}")
    for key, value in ultra_conservative_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")
    print("DEBUGGING COMMANDS FOR SVD COMPLETION")
    print(f"{'='*60}")
    print("# Check if SVD is being called in the model:")
    print("grep -r 'svd_matrix_completion\\|SVD\\|matrix_completion' src/models/")
    print()
    print("# Check if adaptive_bpr is working:")
    print("grep -r 'adaptive_bpr\\|AdaptiveBPR' src/models/")
    print()
    print("# Check if fusion_similarity is working:")
    print("grep -r 'fusion_similarity\\|FusionSimilarity' src/models/")
    
    print("\n" + "="*60)
    print("MANUAL EXPERIMENT EXAMPLE")
    print("="*60)
    
    print("RESCUE CONFIG to fix July 9th issues:")
    print("- embedding_size: 256 → 64 (4x smaller)")
    print("- n_mm_layers: 3 → 1 (3x simpler)")  
    print("- learning_rate: 0.0015 → 0.001 (more conservative)")
    print("- Added L2 regularization to prevent overfitting")
    print("- Need to debug SVD completion separately")
    
    print("\nTo test rescue configuration, update your main.py with:")
    for key, value in rescue_config.items():
        print(f"  args.{key} = {value}")
