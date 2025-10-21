# SMAF
A Systematic Framework for Testing Multimodal Feature Contribution in Recommendation Systems

Paper Authors: Jifang Wang and Anunobi Victor Chibueze

Conference: The 3rd International Conference on Networks, Communications and Intelligent Computing, NCIC 2025

Date: 17-19th October 2025, Jiaozuo, China.

# Background/Objective:
Multimodal recommendation systems have gained widespread adoption with the promise of improved performance by leveraging diverse data modalities (text, images, audio). However, despite extensive research, the actual contribution of individual modalities to system performance remains largely untested, creating a critical evaluation gap in the field.

Methods: We propose SMAF (Systematic Multimodal Ablation Framework), a rigorous methodology for quantifying true multimodal contribution in recommendation systems. Our framework employs controlled ablation studies with statistical significance testing across multiple datasets, systematically isolating each modality's contribution through feature masking while maintaining architectural integrity.

Results: Through comprehensive evaluation of SEA (Self-supervised Multi-modal Alignment), a state-of-the-art multimodal recommender, across three Amazon datasets, we demonstrate that this "multimodal" system achieves 98.5% of its performance using only text features. Image features contribute less than 2% to overall performance (Recall@20: 0.7%-1.5% improvement), with 95% confidence intervals often including zero, classifying SEA as pseudo-multimodal according to our framework criteria.

Conclusion: Our findings challenge fundamental assumptions about multimodal effectiveness in recommendation systems, revealing that sophisticated architectures may be functionally unimodal despite multimodal design. SMAF provides essential evaluation tools for the research community to systematically assess and improve multimodal systems, promoting more honest evaluation practices and efficient resource allocation.

Keywords: Multimodal Recommendation, Ablation Studies, Feature Fusion, Evaluation Methodology, Systematic Evaluation

# Overview
This supplementary package contains all materials necessary to reproduce the experimental results reported in our paper. We provide the complete source code, datasets, experimental scripts, and detailed documentation for applying the Systematic Multimodal Ablation Framework (SMAF) to multimodal recommendation systems.

# Contents of Supplementary Materials
Source Code: Complete SEA model implementation with ablation experiment scripts

Datasets: Amazon review datasets (Baby, Clothing, Sports & Outdoors) with extracted features

Experimental Logs: Complete training logs from all experiments reported in the paper

Analysis Scripts: Tools for dataset analysis, performance evaluation, and statistical testing

# Table of Contents
1. System Requirements
2. Installation Instructions
3. Dataset Preparation
4. Reproducing Paper Results
5. Understanding the SMAF Methodology
6. File Structure
7. Experimental Logs and Results
8. Analysis Tools
9. Configuration and Customization
10. Troubleshooting
11. Extending to Other Models
12. Citation
13. Contact Information
14. Acknowledgments
15. License
16. FAQ

# 1. System Requirements
# Hardware Requirements
CPU: Intel Core i7 or equivalent (multi-core recommended)

RAM: Minimum 16GB (32GB recommended for all three datasets)

GPU: NVIDIA GPU with CUDA support (strongly recommended)

Minimum: GTX 1080 Ti (11GB VRAM)

Recommended: RTX 3090 or A100 (24GB+ VRAM)

Storage: At least 20GB free disk space for datasets and logs

# Software Requirements
Operating System: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10/11

Python: Version 3.7.x (as specified in requirements.txt)

CUDA: Version 10.2 or 11.x (if using GPU)

Package Manager: pip or conda

# Computational Time Estimates
Based on NVIDIA RTX 3090:

Full baseline training (single dataset): 2-4 hours

Text-only ablation (single dataset): 2-4 hours

Complete ablation study (all datasets, 5 seeds): 48-72 hours

Dataset analysis scripts: 10-30 minutes

# 2. Installation Instructions
Step 1: Extract the Supplementary Materials

Extract the archive

unzip SMAF_PROJECT-supplementary.zip

cd SMAF_PROJECT

Step 2: Create Python Environment (Recommended)

Using conda (recommended)

conda create -n smaf python=3.7.11

conda activate smaf

Or using venv

python3.7 -m venv smaf_env

source smaf_env/bin/activate 

On Linux/macOS

smaf_env

Scriptsactivate  

On Windows

Step 3: Install Dependencies

Install all required packages

pip install -r requirements.txt

# Verify PyTorch installation

python -c "import torch; 

print('PyTorch version:', torch.__version__)"

python -c "import torch; 

print('CUDA available:', torch.cuda.is_available())"

Note: The requirements.txt specifies:

1.numpy==1.21.5

2. pandas==1.3.5

3. scipy==1.7.3

4. torch==1.11.0

5. pyyaml==6.0

# 3. Dataset Preparation
# Dataset Overview

We use three Amazon review datasets with pre-extracted multimodal features for our experiments.

Dataset Statistics
