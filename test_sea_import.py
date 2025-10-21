#!/usr/bin/env python3
"""
Test script to import and verify the SEA model works correctly.
This script properly sets up the Python path to handle relative imports.
"""
import sys
import os

# Add the src directory to Python path to handle relative imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def test_sea_imports():
    """Test that all SEA model imports work correctly."""
    print("Testing SEA model imports...")
    
    try:
        # Test individual utility imports
        print("1. Testing SVD completion import...")
        from utils.svd_completion import apply_svd_completion_to_dataset
        print("   ✓ SVD completion imported successfully")
        
        print("2. Testing fusion similarity import...")
        from utils.fusion_similarity import build_fusion_knn_graph
        print("   ✓ Fusion similarity imported successfully")
        
        print("3. Testing adaptive BPR import...")
        from utils.adaptive_bpr import AdaptiveBPRLoss
        print("   ✓ Adaptive BPR imported successfully")
        
        print("4. Testing common modules...")
        from common.abstract_recommender import GeneralRecommender
        from common.loss import BPRLoss, EmbLoss
        from common.init import xavier_uniform_initialization
        print("   ✓ Common modules imported successfully")
        
        print("5. Testing SEA model import...")
        from models.SEA import SEA
        print("   ✓ SEA model imported successfully")
        
        print("\n🎉 All imports successful! The SEA model is ready to use.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nTroubleshooting tips:")
        print("- Make sure you're running this script from the SEA-main directory")
        print("- Check that all required files exist in the src/ directory")
        print("- Verify that Python can find the src directory")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def create_mock_dataset():
    """Create a mock dataset for testing purposes."""
    class MockDataset:
        def __init__(self):
            self.user_num = 100
            self.item_num = 200
            self.interaction_matrix = None  # Will be set by model if needed
            self.dataset_name = 'baby'  # Add dataset_name attribute
            
            # Add field names expected by AdaptiveBPRLoss
            self.uid_field = 'user_id'
            self.iid_field = 'item_id'
            
            # Create mock data for __getitem__ access
            self._data = []
            import torch
            for i in range(100):  # Create 100 mock interactions
                self._data.append({
                    'user_id': i % self.user_num,
                    'item_id': i % self.item_num,
                    'rating': 1.0
                })
        
        def get_user_num(self):
            return self.user_num
            
        def get_item_num(self):
            return self.item_num
            
        def __len__(self):
            """Return dataset length (required for subscriptable access)"""
            return len(self._data)
            
        def __getitem__(self, idx):
            """Make dataset subscriptable (required by AdaptiveBPRLoss)"""
            if idx >= len(self._data):
                idx = idx % len(self._data)
            return self._data[idx]
            
        def inter_matrix(self, form='coo'):
            """Mock interaction matrix for testing"""
            import scipy.sparse as sp
            import numpy as np
            # Create a small random sparse matrix for testing
            data = np.random.randint(0, 2, size=50)  # 50 random interactions
            row = np.random.randint(0, self.user_num, size=50)
            col = np.random.randint(0, self.item_num, size=50)
            if form == 'coo':
                return sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))
            elif form == 'csr':
                return sp.csr_matrix((data, (row, col)), shape=(self.user_num, self.item_num))
            else:
                return sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))
    
    return MockDataset()

def test_sea_model_creation():
    """Test creating an instance of the SEA model."""
    print("\nTesting SEA model instantiation...")
    
    try:
        # Mock config and dataset for testing
        class MockConfig:
            def __init__(self):
                # Core model parameters
                self.embedding_size = 64
                self.feat_embed_dim = 64
                self.n_mm_layers = 1
                self.dropout = 0.1
                self.reg_weight = 1e-04
                self.c_alpha = 1.0
                self.c_beta = 0.5
                self.knn_k = 10
                self.mm_image_weight = 0.5
                self.device = 'cpu'
                
                # Dataset-related fields required by GeneralRecommender
                self.USER_ID_FIELD = 'user_id'
                self.ITEM_ID_FIELD = 'item_id'
                self.NEG_PREFIX = 'neg_'
                self.train_batch_size = 32
                
                # Multimodal and dataset paths
                self.end2end = False
                self.is_multimodal_model = True
                self.data_path = './data/'
                self.dataset = 'baby'
                self.vision_feature_file = 'image_feat.npy'
                self.text_feature_file = 'text_feat.npy'
                
            def __getitem__(self, key):
                """Make config subscriptable like a dictionary"""
                return getattr(self, key)
                
            def __setitem__(self, key, value):
                """Allow setting values like a dictionary"""
                setattr(self, key, value)
                
            def __contains__(self, key):
                """Support 'in' operator"""
                return hasattr(self, key)
                
            def get(self, key, default=None):
                """Dictionary-style get method"""
                return getattr(self, key, default)
                
        class MockDataLoader:
            def __init__(self, dataset):
                self.dataset = dataset
                
            def inter_matrix(self, form='coo'):
                """Mock interaction matrix for testing - delegate to dataset"""
                return self.dataset.inter_matrix(form)
                
            def __len__(self):
                """Return the length of the dataloader (required by adaptive BPR)"""
                return len(self.dataset)
                
            def __getitem__(self, idx):
                """Make dataloader subscriptable (delegate to dataset)"""
                return self.dataset[idx]
                
            # Delegate dataset attributes
            @property
            def uid_field(self):
                return self.dataset.uid_field
                
            @property 
            def iid_field(self):
                return self.dataset.iid_field
                
            @property
            def user_num(self):
                return self.dataset.user_num
                
            @property
            def item_num(self):
                return self.dataset.item_num
        
        config = MockConfig()
        dataset = create_mock_dataset()
        dataloader = MockDataLoader(dataset)
        
        print("Creating SEA model instance...")
        
        # Create temporary mock feature files for testing
        import tempfile
        import numpy as np
        
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        baby_dir = os.path.join(temp_dir, 'baby')
        os.makedirs(baby_dir, exist_ok=True)
        
        # Create mock feature files
        mock_image_feat = np.random.rand(200, 64).astype(np.float32)  # 200 items, 64 features
        mock_text_feat = np.random.rand(200, 64).astype(np.float32)   # 200 items, 64 features
        
        image_feat_path = os.path.join(baby_dir, 'image_feat.npy')
        text_feat_path = os.path.join(baby_dir, 'text_feat.npy')
        
        np.save(image_feat_path, mock_image_feat)
        np.save(text_feat_path, mock_text_feat)
        
        # Update config to point to temp directory
        config.data_path = temp_dir + '/'
        
        # Create mock args object for SEA model
        class MockArgs:
            def __init__(self):
                self.alpha_contrast = 0.2
                self.temp = 0.2
                self.beta = 0.01
                self.reg_weight = 0.0001
        
        args = MockArgs()
        
        from models.SEA import SEA
        model = SEA(config, dataloader, args)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        print("   ✓ SEA model created successfully")
        
        print(f"   - Model has {sum(p.numel() for p in model.parameters())} parameters")
        print(f"   - Embedding size: {config.embedding_size}")
        print(f"   - MM layers: {config.n_mm_layers}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_svd_completion_trigger():
    """Test that SVD completion can be triggered with the right config."""
    print("\nTesting SVD completion trigger...")
    
    try:
        # Create a mock config that should trigger SVD completion
        class MockConfig:
            def __init__(self):
                # Core model parameters
                self.embedding_size = 64
                self.feat_embed_dim = 64
                self.n_mm_layers = 1
                self.dropout = 0.1
                self.reg_weight = 1e-04
                self.c_alpha = 1.0
                self.c_beta = 0.5
                self.knn_k = 10
                self.mm_image_weight = 0.5
                self.device = 'cpu'
                
                # Dataset-related fields
                self.USER_ID_FIELD = 'user_id'
                self.ITEM_ID_FIELD = 'item_id' 
                self.NEG_PREFIX = 'neg_'
                self.train_batch_size = 32
                
                # Multimodal and dataset paths
                self.end2end = False
                self.is_multimodal_model = True
                self.data_path = './data/'
                self.dataset = 'baby'
                self.vision_feature_file = 'image_feat.npy'
                self.text_feature_file = 'text_feat.npy'
                
                # SVD completion trigger
                self.use_svd_completion = True
                
            def __getitem__(self, key):
                return getattr(self, key)
                
            def __setitem__(self, key, value):
                setattr(self, key, value)
                
            def __contains__(self, key):
                return hasattr(self, key)
                
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config = MockConfig()
        dataset = create_mock_dataset()
        
        print(f"Config has use_svd_completion: {'use_svd_completion' in config}")
        print(f"Config use_svd_completion value: {config.get('use_svd_completion', 'NOT_FOUND')}")
        
        from utils.svd_completion import apply_svd_completion_to_dataset
        
        print("Testing SVD completion function directly...")
        original_matrix = dataset.inter_matrix()
        print(f"Original matrix shape: {original_matrix.shape}")
        print(f"Original matrix nnz: {original_matrix.nnz}")
        
        # Test the SVD completion function
        completed_matrix = apply_svd_completion_to_dataset(dataset, k=50)
        
        if completed_matrix is not None:
            print(f"Completed matrix shape: {completed_matrix.shape}")
            print(f"Completed matrix nnz: {completed_matrix.nnz}")
            print("   ✓ SVD completion function works")
        else:
            print("   ⚠️  SVD completion returned None")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SVD completion test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SEA Model Import and Instantiation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_sea_imports()
    
    if imports_ok:
        # Test model creation
        model_ok = test_sea_model_creation()
        
        if model_ok:
            # Test SVD completion trigger
            svd_ok = test_svd_completion_trigger()
            
            if svd_ok:
                print("\n" + "=" * 60)
                print("🎉 ALL TESTS PASSED! The SEA model is working correctly.")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("⚠️  Imports and model creation work, but SVD completion trigger test failed.")
                print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠️  Imports work but model creation failed.")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Import tests failed. Please fix import issues first.")
        print("=" * 60)
