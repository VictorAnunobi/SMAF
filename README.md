# SMAF

A Systematic Framework for Testing Multimodal Feature Contribution in Recommendation Systems

Paper Authors: Jifang Wang and Anunobi Victor Chibueze

Conference: The 3rd International Conference on Networks, Communications and Intelligent Computing, NCIC 2025

Date: 17-19th October 2025, Jiaozuo, China.

# Background/Objective:

Multimodal recommendation systems have gained widespread adoption with the promise of improved performance by leveraging diverse data modalities (text, images, audio). However, despite extensive research, the actual contribution of individual modalities to system performance remains largely untested, creating a critical evaluation gap in the field.

Methods: We propose SMAF (Systematic Multimodal Ablation Framework), a rigorous methodology for quantifying true multimodal contribution in recommendation systems. Our framework employs controlled ablation studies with statistical significance testing across multiple datasets, systematically isolating each modality's contribution through feature masking while maintaining architectural integrity.

Results: Through comprehensive evaluation of SEA (It is Never Too Late to Mend: Separate Learning for Multimedia Recommendation), a state-of-the-art multimodal recommender, across three Amazon datasets, we demonstrate that this "multimodal" system achieves 98.5% of its performance using only text features. Image features contribute less than 2% to overall performance (Recall@20: 0.7%-1.5% improvement), with 95% confidence intervals often including zero, classifying SEA as pseudo-multimodal according to our framework criteria.

Conclusion: Our findings challenge fundamental assumptions about multimodal effectiveness in recommendation systems, revealing that sophisticated architectures may be functionally unimodal despite multimodal design. SMAF provides essential evaluation tools for the research community to systematically assess and improve multimodal systems, promoting more honest evaluation practices and efficient resource allocation.

Keywords: Multimodal Recommendation, Ablation Studies, Feature Fusion, Evaluation Methodology, Systematic Evaluation

# Supplementary Materials

## Overview

This supplementary package contains all materials necessary to reproduce the experimental results reported in our paper. We provide the complete source code, datasets, experimental scripts, and detailed documentation for applying the Systematic Multimodal Ablation Framework (SMAF) to multimodal recommendation systems.

### Contents of Supplementary Materials

-   **Source Code**: Complete SEA model implementation with ablation experiment scripts
-   **Datasets**: Amazon review datasets (Baby, Clothing, Sports & Outdoors) with extracted features
-   **Experimental Logs**: Complete training logs from all experiments reported in the paper
-   **Analysis Scripts**: Tools for dataset analysis, performance evaluation, and statistical testing
-   **Documentation**: Methodology paper draft and detailed experimental reports

## Table of Contents

1.  [System Requirements](#1-system-requirements)
2.  [Installation Instructions](#2-installation-instructions)
3.  [Dataset Preparation](#3-dataset-preparation)
4.  [Reproducing Paper Results](#4-reproducing-paper-results)
5.  [Understanding the SMAF Methodology](#5-understanding-the-smaf-methodology)
6.  [File Structure](#6-file-structure)
7.  [Experimental Logs and Results](#7-experimental-logs-and-results)
8.  [Analysis Tools](#8-analysis-tools)
9.  [Configuration and Customization](#9-configuration-and-customization)
10.  [Troubleshooting](#10-troubleshooting)
11.  [Extending to Other Models](#11-extending-to-other-models)
12.  [Citation](#12-citation)
13.  [Contact Information](#13-contact-information)
14.  [Acknowledgments](#14-acknowledgments)
15.  [License](#15-license)
16.  [FAQ](#16-faq)

## 1. System Requirements

### Hardware Requirements

-   **CPU**: Intel Core i7 or equivalent (multi-core recommended)
-   **RAM**: Minimum 16GB (32GB recommended for all three datasets)
-   **GPU**: NVIDIA GPU with CUDA support (strongly recommended)
    -   **Minimum**: GTX 1080 Ti (11GB VRAM)
    -   **Recommended**: RTX 3090 or A100 (24GB+ VRAM)
-   **Storage**: At least 20GB free disk space for datasets and logs

### Software Requirements

-   **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10/11
-   **Python**: Version 3.7.x (as specified in requirements.txt)
-   **CUDA**: Version 10.2 or 11.x (if using GPU)
-   **Package Manager**: pip or conda

### Computational Time Estimates

Based on NVIDIA RTX 3090:

-   **Full baseline training** (single dataset): 2-4 hours
-   **Text-only ablation** (single dataset): 2-4 hours
-   **Complete ablation study** (all datasets, 5 seeds): 48-72 hours
-   **Dataset analysis scripts**: 10-30 minutes

## 2. Installation Instructions

### Step 1: Extract the Supplementary Materials

```bash
# Extract the archive
unzip SMAF-main.zip
cd SMAF-main
```

### Step 2: Create Python Environment (Recommended)

```bash
# Using conda (recommended)
conda create -n smaf python=3.7.11
conda activate smaf

# Or using venv
python3.7 -m venv smaf_env
source smaf_env/bin/activate

# On Linux/macOS#
smaf_envScriptsactivate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch;
print('PyTorch version:', torch.__version__)
python -c "import torch;
print('CUDA available:', torch.cuda.is_available())"
```

#### Note: The requirements.txt specifies:

-   numpy==1.21.5
-   pandas==1.3.5
-   scipy==1.7.3
-   torch==1.11.0
-   pyyaml==6.0

# 3. Dataset Preparation

## Dataset Overview

We use three Amazon review datasets with pre-extracted multimodal features for our experiments.

### Dataset Statistics

| Dataset  | Users  | Items  | Interactions | Text Features  | Image Features |
|----------|--------|--------|--------------|----------------|----------------|
| Baby     | 19,445 | 7,050  | 160,792      | Sentence-BERT  | ResNet-50      |
| Clothing | 39,387 | 23,033 | 278,677      | Sentence-BERT  | ResNet-50      |
| Sports   | 35,598 | 18,357 | 296,337      | Sentence-BERT  | ResNet-50      |

## Feature Details

### Text Features

-   **Model**: Sentence-BERT embeddings (384 dimensions)
-   **Pre-training**: Semantic textual similarity tasks
-   **Source**: Product descriptions and reviews
-   **Capabilities**: Captures semantic meaning of text content

### Image Features

-   **Model**: ResNet-50 visual features (4096 dimensions)
-   **Extraction**: From product images using pre-trained ResNet-50
-   **Layer**: Final layer features before classification
-   **Representation**: Visual appearance and attributes

## Dataset Structure

Each dataset directory follows this structure:
```
data/{dataset}/
├── {dataset}.inter          # User-item interactions (train/val/test splits)
├── text_feat.npy            # Text features (Sentence-BERT embeddings, 384d)
├── image_feat.npy           # Image features (ResNet-50 features, 4096d)
├── u_id_mapping.csv         # User ID to index mapping
└── i_id_mapping.csv         # Item ID to index mapping
```

## Download Datasets

### Option 1: Google Drive (Recommended)

[📎 Download from Google Drive](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG)

After downloading, extract to the data directory:
```bash
unzip baby.zip -d data/baby/
unzip clothing.zip -d data/clothing/
unzip sports.zip -d data/sports/
```

## Dataset Quality Analysis

Run our analysis script to examine dataset characteristics:

```bash
python analyze_dataset_characteristics.py
```

This analysis provides:

-   **Interaction Statistics**: Sparsity levels, user/item activity distributions
-   **Feature Quality Metrics**: Coverage, distribution, missing values
-   **Missing Feature Analysis**: Percentage of zero/missing features per modality
-   **Rating Distributions**: User rating patterns (if available)

### Key Findings from Analysis:

-   **Clothing Dataset**: 91% missing/zero image features (critical quality issue)
-   **Baby Dataset**: 23% missing image features
-   **Sports Dataset**: 15% missing image features

These feature quality issues significantly impact the multimodal learning capability of the model.

## 4. Reproducing Paper Results

### Overview of Experiments

Our paper reports results from three main types of experiments:

-   **Baseline Multimodal Performance** (Table 2 in paper)
-   **Text-Only Ablation** (Table 3 in paper)
-   **Image-Only Ablation** (Table 3 in paper)
-   **Statistical Significance Analysis** (Section 4 in paper)
-   **Cross-Dataset Validation** (Section 4 in paper)

### Experiment 1: Baseline Multimodal Performance (Table 1 in Paper)

Train the full multimodal SEA model on all datasets:

```bash
# Baby dataset
cd src
python main.py --model SEA --dataset baby --gpu 0

# Clothing dataset
python main.py --model SEA --dataset clothing --gpu 0

# Sports dataset
python main.py --model SEA --dataset sports --gpu 0
```

### Experiment 2: Text-Only Ablation (Critical Finding)

This experiment demonstrates that SEA achieves 98.5%+ of its performance using only text features.

```bash
# Run text-only experiment for Clothing dataset
python test_text_only.py

# For Baby dataset (dedicated script)
python test_baby_text_only.py
```

**Expected Results:**

-   Text-only performance retains 98.5%+ of baseline
-   Image contribution: <2% across all datasets
-   See `MULTIMODAL_ANALYSIS_REPORT.md` for detailed findings

### Experiment 3: Systematic Multi-Seed Evaluation

Run multiple seeds for statistical rigor (as reported in paper):

```bash
# Run 5-seed evaluation for each dataset
python run_focused_experiments.py --dataset baby --num-seeds 5
python run_focused_experiments.py --dataset clothing --num-seeds 5
python run_focused_experiments.py --dataset sports --num-seeds 5
```

### Experiment 4: Statistical Analysis

Analyze performance gaps and statistical significance:

```bash
# Compute statistical tests, confidence intervals, and effect sizes
python analyze_performance_gap.py

# Analyze multimodal usage patterns across datasets
python analyze_multimodal_usage.py
```

This generates:

-   Paired t-tests for baseline vs. text-only
-   Cohen's d effect sizes
-   95% confidence intervals
-   Cross-dataset consistency analysis

### Experiment 5: Feature Quality Investigation

Investigate image feature quality issues (important finding in paper):

```bash
# Analyze image feature quality across all datasets
python investigate_image_issues.py
```

This reveals:
- Clothing: 91% missing/zero image features
- Baby: 23% missing image features  
- Sports: 15% missing image features

### Complete Reproduction Pipeline

For automated reproduction of all experiments:

```bash
# Run comprehensive experimental pipeline.
/run_all_experiments.sh
```

**Runtime Estimate:** 48-72 hours on RTX 3090

## 5. Understanding the SMAF Methodology

### Conceptual Overview

The Systematic Multimodal Ablation Framework (SMAF) is implemented through the experimental scripts in this repository. The methodology consists of three core components:

#### 1. Baseline Establishment

-   Train full multimodal model with all features
-   Use multiple random seeds (5+) for statistical validity
-   Record comprehensive metrics (Recall@K, NDCG@K)

#### 2. Systematic Ablation

-   **Zero-Fill Masking**: Replace target modality features with zeros
-   **Architecture Preservation**: Keep model structure identical
-   **Controlled Isolation**: Change only the feature matrix, not the code

#### 3. Statistical Analysis

-   **Significance Testing**: Paired t-tests between baseline and ablated
-   **Effect Size**: Cohen's d for practical significance
-   **Confidence Intervals**: 95% CI for contribution estimates
-   **Cross-Dataset Validation**: Verify findings across multiple datasets

### Implementation Approach

Our implementation uses feature masking rather than architectural changes:

```python
# Example: Text-only ablation (from test_text_only.py)# Zero out image features while keeping text features intactimage_features = np.zeros_like(original_image_features)
```

**Why Feature Masking?**This approach ensures:

-   No code modifications needed
-   Identical model architecture across experiments
-   Clean isolation of modality contributions
-   Easy reproducibility

## Classification Criteria (from paper)

Based on performance retention when using only a single modality:

| Retention | Classification | Description |
|-----------|----------------|-------------|
| >98% | **Pseudo-Multimodal** | Functionally unimodal despite multimodal design |
| 90-98% | **Partially Multimodal** | Some multimodal benefit but imbalanced |
| <90% | **True Multimodal** | Genuine multimodal learning with synergy |

### Understanding the Classifications

#### Pseudo-Multimodal (>98% retention)

-   Model achieves nearly identical performance with only one modality
-   Indicates one modality dominates completely
-   Other modalities contribute <2% to performance
-   **Example**: SEA on all three datasets (text-only retains 98.5%+ performance)

#### Partially Multimodal (90-98% retention)

-   Model shows some multimodal benefit but highly imbalanced
-   One modality is dominant but not overwhelming
-   Room for improved multimodal fusion strategies

#### True Multimodal (<90% retention)

-   Significant performance drop when any modality is removed
-   Both modalities contribute substantially
-   Evidence of genuine multimodal synergy and complementary learning

### Our Key Finding

**SEA is Pseudo-Multimodal**: Text-only ablation retains 98.5%+ of full multimodal performance across all datasets, indicating the model is effectively text-only despite its multimodal architecture.

| Dataset | Full Multimodal (Recall@20) | Text-Only (Recall@20) | Image Contribution (Recall@20) | Classification |
|---------|-----------------------------|----------------------|-------------------------------|----------------|
| Baby | 0.0474 | 0.0467 | +1.5% | Pseudo-Multimodal |
| Clothing | 0.0441 | 0.0438 | -0.8% | Pseudo-Multimodal |
| Sports | 0.0695 | 0.0689 | (+0.7%) | Pseudo-Multimodal |

### Implications:

-   Despite having a sophisticated multimodal architecture, SEA functions as a text-only model
-   Image features provide minimal benefit (<2% across all datasets)
-   This challenges assumptions about multimodal learning in recommendation systems
-   Suggests need for better multimodal fusion strategies or higher-quality image features

## 6. File Structure

### Root Directory

```
SMAF_PROJECT/
├── README.md                          # Original SEA project README
├── requirements.txt                   # Python dependencies and versions
├── MULTIMODAL_ANALYSIS_REPORT.md     # Key findings and analysis
```

### Source Code (src/)

```
src/
├── main.py                            # Main training entry point
│
├── models/                            # Model implementations
│   ├── __init__.py
│   ├── SEA.py                         # SEA model (full version)
│   └── SEA_simple.py                  # Simplified SEA variant
│
├── common/                            # Core components
│   ├── __init__.py
│   ├── abstract_recommender.py        # Base recommender class
│   ├── encoders.py                    # Feature encoders (text/image)
│   ├── trainer.py                     # Training loop implementation
│   └── loss.py                        # Loss functions (BPR, contrastive)
│
├── utils/                             # Utilities and helpers
│   ├── __init__.py
│   ├── dataloader.py                  # Data loading and batching
│   ├── dataset.py                     # Dataset class
│   ├── metrics.py                     # Evaluation metrics (Recall, NDCG)
│   ├── topk_evaluator.py              # Top-K evaluation
│   ├── quick_start.py                 # Training orchestration
│   ├── logger.py                      # Logging utilities
│   ├── configurator.py                # Configuration management
│   ├── data_utils.py                  # Data processing utilities
│   ├── fusion_similarity.py           # Feature fusion methods
│   ├── adaptive_bpr.py                # Adaptive BPR loss
│   ├── svd_completion.py              # SVD-based feature completion
│   └── utils.py                       # General utilities
│
├── configs/                           # Configuration files
│   ├── overall.yaml                   # Global training settings
│   ├── mg.yaml                        # Mirror Gradient configuration
│   │
│   ├── dataset/                       # Dataset-specific configs
│   │   ├── baby.yaml                  # Baby dataset settings
│   │   ├── clothing.yaml              # Clothing dataset settings
│   │   ├── sports.yaml                # Sports dataset settings
│   │   └── elec.yaml                  # Electronics dataset settings
│   │
│   └── model/                         # Model-specific configs
│       └── SEA.yaml                   # SEA hyperparameters
│
└── log/                               # Training logs (auto-generated)
```

### Datasets (data/)

```
data/
├── README.md                          # Dataset documentation and links
│
├── baby/                              # Baby Products Dataset
│   ├── baby.inter                     # User-item interactions
│   ├── text_feat.npy                  # Sentence-BERT embeddings (384d)
│   ├── image_feat.npy                 # ResNet-50 features (4096d)
│   ├── u_id_mapping.csv               # User ID mappings
│   └── i_id_mapping.csv               # Item ID mappings
│
├── clothing/                          # Clothing Dataset
│   ├── clothing.inter
│   ├── text_feat.npy
│   ├── image_feat.npy
│   ├── u_id_mapping.csv
│   └── i_id_mapping.csv
│
└── sports/                            # Sports & Outdoors Dataset
    ├── sports.inter
    ├── text_feat.npy
    ├── image_feat.npy
    ├── u_id_mapping.csv
    └── i_id_mapping.csv
```

### Analysis Scripts (analysis_scripts/)

```
analysis_scripts/
├── analyze_dataset_characteristics.py  # Dataset statistics & quality
├── analyze_multimodal_usage.py        # Cross-dataset modality analysis
├── analyze_performance_gap.py         # Statistical significance testing
└── investigate_image_issues.py        # Image feature quality analysis
```

### Experimental & Test Scripts (Root Level)

Root level utility scripts:

```
├── test_text_only.py                  # Text-only ablation (generic)
├── test_baby_text_only.py             # Baby-specific text-only ablation
├── test_sea_import.py                 # Import verification script
│
├── run_focused_experiments.py         # Automated multi-seed experiments
├── optimize_sea.py                    # Hyperparameter optimization
├── systematic_search.py               # Grid search utility
│
├── fix_clothing_features.py           # Feature repair attempts
├── generate_user_graph.py             # User interaction graph generation
│
├── analyze_dataset_characteristics.py # (Root copy of analysis script)
├── analyze_multimodal_usage.py        # (Root copy of analysis script)
├── analyze_performance_gap.py         # (Root copy of analysis script)
└── investigate_image_issues.py        # (Root copy of analysis script)
```

### Key File Descriptions

| File/Directory | Purpose |
|---------------|---------|
| `src/main.py` | Main entry point for training SEA model |
| `src/models/SEA.py` | Core SEA implementation with multimodal learning |
| `src/common/trainer.py` | Training loop, validation, and checkpointing |
| `src/utils/dataloader.py` | Handles data loading and batch preparation |
| `src/configs/*.yaml` | All configuration files for reproducibility |
| `data/*/text_feat.npy` | Sentence-BERT embeddings (768d → 384d) |
| `data/*/image_feat.npy` | ResNet-50 visual features (4096d) |
| `test_text_only.py` | Critical ablation script for text-only experiments |
| `run_focused_experiments.py` | Automated multi-seed experimental pipeline |
| `analyze_performance_gap.py` | Statistical analysis with t-tests and effect sizes |
| `MULTIMODAL_ANALYSIS_REPORT.md` | Primary findings document - pseudo-multimodal discovery |
| `main_experiment_log/` | Complete training logs from all paper experiments |

## 7. Experimental Logs and Results

### Overview

The `main_experiment_log/` directory contains complete training logs from all experiments reported in the paper, providing full transparency and reproducibility.

### Main Experimental Logs

#### Paper Results (Table 2)

- `baby_log.log`: Baby dataset full training log
- `clothing_log.log`: Clothing dataset full training log
- `Sports_log.log`: Sports dataset full training log

Each log contains:
- Epoch-by-epoch training progress
- Validation performance metrics
- Final test set results
- Complete hyperparameter settings
- Training time and resource usage

### Key Findings Documentation

#### MULTIMODAL_ANALYSIS_REPORT.md

**Critical Discovery**: Comprehensive analysis proving SEA is effectively text-only

**Document Structure:**

-   Executive summary of breakthrough finding
-   Experimental evidence (text-only vs. full multimodal)
-   Architectural analysis
-   Root cause investigation
-   Implications and next steps

**Main Finding:**Our ablation experiments revealed that SEA achieves virtually identical performance using only text features:

| Dataset | Full Multimodal (Recall@20) | Text-Only (Recall@20) | Image Contribution | Performance Loss |
|---------|-----------------------------|----------------------|-------------------|------------------|
| Clothing | 0.0131 | 0.0132 | -0.0001 | -0.8% |
| Baby | 0.0474 | 0.0467 | 0.0007 | 1.5% |
| Sports | 0.0273 | ~0.0271 | 0.0002 | 0.7% |

**Key Insights:**

-   **Text-only performance matches or exceeds multimodal**: In Clothing dataset, text-only slightly outperforms full multimodal
-   **Image features are ineffective**: Contributing <2% across all datasets
-   **Consistent pattern across datasets**: All three datasets show the same pseudo-multimodal behavior
-   **Root cause identified**: 91% missing image features in Clothing, 15-23% in Baby/Sports

**Statistical Significance:**

-   Paired t-tests confirm differences are statistically insignificant (p > 0.05)
-   Cohen's d effect sizes near zero, indicating no practical significance
-   95% confidence intervals for image contribution

**Architectural Analysis:**The report includes detailed analysis of:

-   Feature encoder architecture
-   Fusion mechanism design
-   Loss function components
-   Gradient flow during training

**Implications for Research:**

-   **Questioning multimodal claims**: Many "multimodal" models may be pseudo-multimodal
-   **Need for rigorous testing**: SMAF methodology essential for validation
-   **Feature quality matters**: Missing/poor-quality features prevent multimodal learning
-   **Architecture improvements needed**: Better fusion strategies required

## 8. Analysis Tools

### Dataset Characteristics Analysis

```bash
python analyze_dataset_characteristics.py
```

**Outputs:**

-   Interaction statistics (user/item counts, sparsity)
-   User/item activity distributions
-   Rating distributions (if available)
-   Feature coverage analysis

**Use Case**: Understanding dataset properties that may affect model performance

### Multimodal Usage Analysis

```bash
python analyze_multimodal_usage.py
```

**Outputs:**

-   Cross-dataset consistency of findings
-   Modality contribution quantification
-   Statistical significance tests
-   Visualization of multimodal effectiveness

**Use Case**: Analyzing how well different modalities are utilized

### Performance Gap Analysis

```bash
python analyze_performance_gap.py
```

**Outputs:**

-   Baseline vs. ablated performance comparison
-   Statistical significance tests (paired t-tests)
-   Effect sizes (Cohen's d)
-   95% confidence intervals
-   Contribution percentage calculations

**Use Case**: Quantifying exact modality contributions with statistical rigor

### Image Feature Quality Investigation

```bash
python investigate_image_issues.py
```

**Outputs:**

-   Missing feature statistics (% zero features per item)
-   Feature distribution analysis
-   Quality metrics
-   Dataset-specific issues

**Key Finding**: Clothing dataset has 91% missing/zero image features

## 9. Configuration and Customization

### Model Configuration

Edit `src/configs/model/SEA.yaml` to modify model hyperparameters:

```yaml
# Embedding dimensions
embedding_size: 64

# Number of GNN layers
n_mm_layers: 2

# Loss function weights
reg_weight: 0.0001       # L2 regularization
alpha_contrast: 0.2      # Contrastive loss weight
beta: 0.01               # Modal distancing weight
temp: 0.2                # Temperature for contrastive learning

# Training
dropout: 0.1
```

### Dataset Configuration

Edit `src/configs/dataset/{dataset}.yaml`:

```yaml
# Data paths
data_path: data/clothing/

# Features
use_text: True
use_image: True
text_dim: 384
image_dim: 4096
```

### Training Configuration

Edit `src/configs/overall.yaml`:

```yaml
# Training parameters
learning_rate: 0.001
epochs: 1000
batch_size: 2048
stopping_step: 25        # Early stopping patience

# Evaluation
topk: [5, 10, 20]
metrics: ['Recall', 'NDCG']
valid_metric: Recall@20
```

## 10. Troubleshooting

### Common Issues

**Issue 1: Dataset Not Found**

```
Error: FileNotFoundError: data/baby/baby.inter not found
```

**Solution:**

```bash
# Verify datasets are in correct location
ls data/baby/
ls data/clothing/
ls data/sports/

# Download from Google Drive if missing
# https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG
```

**Issue 2: CUDA Out of Memory**

```
Error: RuntimeError: CUDA out of memory
```

**Solution:**

```bash
# Reduce batch size in src/main.py or configs/overall.yaml
# Default: 2048 → Try: 1024 or 512

# Or use CPU (slower)
python main.py --gpu -1  # Use CPU
```

**Issue 3: Import Errors**

```
Error: ModuleNotFoundError: No module named 'utils'
```

**Solution:**

```bash
# Make sure you're in the src directory when running main.py
cd src
python main.py --dataset baby

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/SMAF_PROJECT/src"
```

**Issue 4: Different Results****Possible causes:**

-   Different random seeds → Results vary slightly across runs
-   PyTorch version → Use exact versions from requirements.txt
-   CUDA determinism → Some operations are non-deterministic on GPU
-   Dataset version → Ensure you downloaded from the correct source

**Verification:**

```bash
# Run with multiple seeds and check average
python run_focused_experiments.py --num-seeds 5

# Results should be within ±2% of reported values
```

**Issue 5: Feature Files Missing (.icloud)**

```
Error: Files like .image_feat.npy.icloud instead of image_feat.npy
```

**Solution (macOS iCloud):**

```bash
# Download files from iCloud
cd data/baby

# Click on each .icloud file to download from iCloud
# Or disable iCloud for this directory
```

## 11. Extending to Other Models

### Applying SMAF to Your Model

The ablation methodology can be applied to any multimodal recommendation model:

**Step 1: Prepare Your Model**Ensure your model loads features from files:

```python
# Your model should load features like:
text_features = np.load('data/your_dataset/text_feat.npy')
image_features = np.load('data/your_dataset/image_feat.npy')
```

**Step 2: Create Ablation Script**

```python
# my_model_ablation.py
import numpy as np
import shutil

# Backup original features
shutil.copy('data/your_dataset/image_feat.npy',
            'data/your_dataset/image_feat_backup.npy')

# Zero out image features for text-only test
img_feat = np.load('data/your_dataset/image_feat.npy')
np.save('data/your_dataset/image_feat.npy', np.zeros_like(img_feat))

# Train your model
# ... your training code ...

# Restore features
shutil.copy('data/your_dataset/image_feat_backup.npy',
            'data/your_dataset/image_feat.npy')
```

**Step 3: Run Statistical Analysis**Use our analysis scripts:

```bash
# Adapt analyze_performance_gap.py for your results
python analyze_performance_gap.py --your-model-results
```

## 12. Citation

If you use this code, framework, or findings in your research, please cite:

```bibtex
@inproceedings{
  title={A Systematic Framework for Testing Multimodal Feature Contribution in Recommendation Systems},
  author={Wang Jifang and Anunobi Victor Chibueze},
  booktitle={The 3rd International Conference on Networks, Communications and Intelligent Computing (NCIC 2025)},
  year={2025},
  month={October},
  address={Jiaozuo, China}
}
```

**Original SEA Model:**

```bibtex
@inproceedings{
  title={It is Never Too Late to Mend: Separate Learning for Multimedia Recommendation},
  author={Wei, Yongxu and others},
  booktitle={Proceedings of ACM Conference},
  year={2023}
}
```

## 13. Contact Information

For questions, issues, or collaboration:

**Primary Authors:**

-   Jifang Wang ([Wangjianfang@hpu.edu.cn](mailto:Wangjianfang@hpu.edu.cn))
-   Anunobi Victor Chibueze ([522309010004@home.hpu.edu.cn](mailto:522309010004@home.hpu.edu.cn))

**Conference:** NCIC 2025, Jiaozuo, China  
**Project Repository:**[📎[https://github.com/VictorAnunobi/SMAF.git]](https://github.com/VictorAnunobi/SMAF.git%5D)

## 14. Acknowledgments

We thank:

-   The NCIC 2025 reviewers for their valuable feedback
-   Julian McAuley (UCSD) for providing the Amazon review datasets
-   The original SEA authors for their open-source codebase
-   MMRec project for providing the foundation

## 15. License

This code is released for academic and research purposes. For commercial use, please contact the authors.

## 16. FAQ

**Q: What is the main finding of this work?**  
A: SEA, a state-of-the-art “multimodal” recommender, achieves 98.5 %+ of its performance using only text features. Image features contribute <2 %, making it effectively text-only despite multimodal design.

**Q: How long does it take to reproduce all experiments?**  
A: Approximately 48–72 hours on a modern GPU (RTX 3090 or equivalent) for all datasets with 5 seeds each.

**Q: Do I need the exact same hardware?**  
A: No, but GPU is strongly recommended. CPU-only would take weeks. Results may vary slightly across hardware but findings remain consistent.

**Q: Can I use this framework with other models?**  
A: Yes! The SMAF methodology (feature masking + statistical testing) works with any multimodal model. See Section 11.

**Q: Why are my results slightly different?**  
A: Small variations (±2 %) are normal due to random-seed differences and hardware variations. Use multiple seeds and statistical testing.

**Q: What if I don't have access to the datasets?**  
A: Download from the provided Google Drive link. The datasets are publicly available Amazon-review data with pre-extracted features.

**Q: How do I know if my model is pseudo-multimodal?**  
A: Run the ablation experiments. If single-modality performance retains >98 % of baseline, your model is pseudo-multimodal.
