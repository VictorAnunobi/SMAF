import os,sys
os.chdir(sys.path[0])
import yaml
import argparse
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SEA', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')
    parser.add_argument('--mg', action="store_true", default=False, help='whether to use Mirror Gradient, default is False')
    parser.add_argument('--alpha_contrast', type=float, default=0.2)  # Back to working value
    parser.add_argument('--temp', type=float, default=0.2)  # Back to paper value
    parser.add_argument('--hidden_dim', type=int, default=64)  # Not used anymore - removed ProjectHeads
    parser.add_argument('--out_dim', type=int, default=128)  # Not used anymore - removed ProjectHeads  
    parser.add_argument('--reg_weight', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.005)  # Will be set based on dataset below
    parser.add_argument('--gpu', type=int, default=0)
    args, _ = parser.parse_known_args()

    # REVERT TO BREAKTHROUGH CONFIGURATION: Recall@20: 0.0130
    # The optimization attempt failed - stick with what works
    if args.dataset.lower() == 'clothing':
        # Exact configuration that achieved breakthrough results
        args.beta = 0.01            # Paper value that worked
        args.alpha_contrast = 0.2   # Paper value that worked
        args.temp = 0.2             # Paper value that worked
        learning_rate = 0.001       # Original working rate
        stopping_step = 25          # Original working patience
        epochs = 100                # Original working epochs
        embedding_size = 64         # Keep original successful size
        n_mm_layers = 2             # Keep successful 2 layers
        dropout = 0.1               # Original working regularization 
    elif args.dataset.lower() == 'baby':
        args.beta = 0.01  # Higher distancing for baby
        args.alpha_contrast = 0.2
        args.temp = 0.2
        learning_rate = 0.001
        stopping_step = 40
        epochs = 1000
        embedding_size = 64
        n_mm_layers = 2
        dropout = 0.1
    elif args.dataset.lower() == 'sports':
        args.beta = 0.005
        args.alpha_contrast = 0.2  
        args.temp = 0.2
        learning_rate = 0.001
        stopping_step = 30
        epochs = 1000
        embedding_size = 64
        n_mm_layers = 2
        dropout = 0.1
    else:
        # Default configuration
        learning_rate = 0.001
        stopping_step = 30
        epochs = 1000
        embedding_size = 128
        n_mm_layers = 2
        dropout = 0.1

    config_dict = {
        'gpu_id': args.gpu,
        'use_gpu': True,
        'device': 'cuda' if args.gpu >= 0 else 'cpu',
        'stopping_step': stopping_step,
        'epochs': epochs,
        'learning_rate': [learning_rate],
        'learning_rate_scheduler': None,  # Disable learning rate scheduling
        'embedding_size': embedding_size,  # Optimize model capacity
        'n_mm_layers': n_mm_layers,  # Optimize graph layers
        'dropout': [dropout],  # Optimize regularization
        # CRITICAL FIXES: Address identified issues
        'train_batch_size': 512,    # Paper value (vs our 256) - MAJOR FIX
        'eval_batch_size': 1024,    # Proportional increase
        'max_split_size_mb': 256,   # More memory for larger batches
        # Add args parameters to config as lists for hyperparameter processing
        'alpha_contrast': [args.alpha_contrast],
        'temp': [args.temp],
        'beta': [args.beta],
        'reg_weight': [args.reg_weight],
        # SUCCESSFUL CONFIGURATION: Keep what achieved breakthrough
        'use_svd_completion': False,    # CRITICAL: Keep disabled
        'svd_k': 50,                
        'use_fusion_similarity': True,  # Keep working
        'fusion_weight': 0.5,           # Revert to working value
        'use_adaptive_bpr': True,       # Keep working
        # FEATURE NORMALIZATION: Critical fix for image features [0,23.4] vs text [-0.26,0.25]
        'normalize_image_features': True,  # Add L2 normalization for image features
        'normalize_text_features': False   # Text features already well-normalized
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, args=args,  save_model=True, mg=args.mg)


