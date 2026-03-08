# Robust automated identification of diatom microfossils under severe data scarcity via deep ensemble learning

This repository provides a simplified implementation of the method proposed in the paper "Robust automated identification of diatom microfossils under severe data scarcity via deep ensemble learning."

The core idea of this work is a deep ensemble framework designed to improve diatom classification under severe data scarcity. Instead of relying on a single model, the proposed approach combines multiple pretrained convolutional neural networks and integrates their predictions through our ensemble strategies.

## Repository Structure

- **crawling.py**  
  Collects and prepares diatom image data used for training.

- **individual_train.py**  
  Trains individual CNN backbone models using transfer learning.

- **base_ensemble.py**  
  Evaluates baseline ensemble strategies (soft voting and hard voting).

- **proposed_ensemble.py**  
  Implements the proposed deep ensemble framework for diatom classification.
