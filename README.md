# DCGAT-DTI: Dynamic Cross-Graph Attention for Drug–Target Interaction Prediction

**DCGAT-DTI** is a novel deep learning framework for **Drug–Target Interaction (DTI) prediction**, designed to enhance drug discovery by dynamically modeling interactions between chemical compounds and proteins. Unlike traditional methods that process drug and protein similarity graphs independently, **DCGAT-DTI** leverages a **Dynamic Cross-Graph Attention (DCGAT)** module to capture intra- and cross-graph dependencies.

## Key Features and Novelty
- **DCGAT Module**: Enables **cross-modal message passing** between drugs and proteins, allowing embeddings to dynamically **incorporate information across both modalities** through intra- and cross-graph attention mechanisms.
- **CNS Network (Cross Neighborhood Selection)**: A **GCN-based selection mechanism** that uses **Gumbel-Softmax Estimator** to  **dynamically selects cross-modal neighbors**, ensuring that each drug and protein node interacts with the most relevant counterparts.

