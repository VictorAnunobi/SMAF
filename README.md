# SMAF
A Systematic Framework for Testing Multimodal Feature Contribution in Recommendation Systems
Paper Authors: Jifang Wang and Anunobi Victor Chibueze
Conference: The 3rd International Conference on Networks, Communications and Intelligent Computing, NCIC 2025
Date: 17-19th October 2025, Jiaozuo, China.

Background/Objective: Multimodal recommendation systems have gained widespread adoption with the promise of improved performance by leveraging diverse data modalities (text, images, audio). However, despite extensive research, the actual contribution of individual modalities to system performance remains largely untested, creating a critical evaluation gap in the field.

Methods: We propose SMAF (Systematic Multimodal Ablation Framework), a rigorous methodology for quantifying true multimodal contribution in recommendation systems. Our framework employs controlled ablation studies with statistical significance testing across multiple datasets, systematically isolating each modality's contribution through feature masking while maintaining architectural integrity.

Results: Through comprehensive evaluation of SEA (Self-supervised Multi-modal Alignment), a state-of-the-art multimodal recommender, across three Amazon datasets, we demonstrate that this "multimodal" system achieves 98.5% of its performance using only text features. Image features contribute less than 2% to overall performance (Recall@20: 0.7%-1.5% improvement), with 95% confidence intervals often including zero, classifying SEA as pseudo-multimodal according to our framework criteria.

Conclusion: Our findings challenge fundamental assumptions about multimodal effectiveness in recommendation systems, revealing that sophisticated architectures may be functionally unimodal despite multimodal design. SMAF provides essential evaluation tools for the research community to systematically assess and improve multimodal systems, promoting more honest evaluation practices and efficient resource allocation.

Keywords: Multimodal Recommendation, Ablation Studies, Feature Fusion, Evaluation Methodology, Systematic Evaluation
