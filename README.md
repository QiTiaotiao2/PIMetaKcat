# PIMetaKcat
PIMetaKcat is a physics-informed meta-learning framework for accurate, few-shot prediction of temperature-dependent enzyme kcat values.
Description:
This repository contains the official implementation of PIMetaKcat, a physics-informed meta-learning framework for predicting the catalytic turnover number (kcat) of enzymes, with explicit modeling of temperature dependence. The model combines advanced molecular embeddings (ESM-2, ProtT5, ChemBERTa) with a task-adaptive meta-learning strategy and a physics-constrained loss based on the Arrhenius equation to achieve accurate and robust prediction under few-shot and temperature-varying scenarios.

Key Features:
🔬 Few-Shot Meta-Learning: Enables rapid adaptation to unseen enzyme families with limited kinetic data.
🌡️ Physics-Informed Modeling: Integrates the Arrhenius law into the loss function to ensure physically consistent temperature-dependent predictions.
🔗 Multi-Modal Input: Supports enzyme sequence, substrate SMILES, and reaction temperature as joint inputs.
📈 Benchmark Evaluation: Includes sequence-similarity controlled evaluation and case studies on mutant prediction and thermal adaptation.

Paper: PIMetaKcat: Few-Shot Prediction of Temperature-Dependent Enzyme kcat using Physics-Informed Meta-Learning
