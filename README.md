# Generative Virtual Screening Toolkit

This repository introduces a **proof-of-concept toolkit for generative virtual screening**, combining seed-based molecule generation, QSAR predictions, retrosynthesis, and reaction yield estimation powered by **Hugging Face transformer models**.

The repository contains the following Python modules and notebooks:

## Repository Structure

- **`ChemBERT_module.py`**  
  An OOP-based module for molecule generation using the **ChemBERTaLM** model  
  (https://huggingface.co/gokceuludogan/ChemBERTaLM).

- **`QSAR.ipynb`**  
  A Jupyter notebook demonstrating basic data preprocessing and **AutoML-based QSAR model construction**.

- **`ReactionT5Retrosynthesis_module.py`**  
  An OOP-based module for retrosynthesis prediction using the  
  **ReactionT5v2-retrosynthesis-USPTO_50k** model  
  (https://huggingface.co/sagawa/ReactionT5v2-retrosynthesis-USPTO_50k).

- **`ReactionT5Yield.py`**  
  An OOP-based module for reaction yield prediction using the  
  **ReactionT5v2-yield** model  
  (https://huggingface.co/sagawa/ReactionT5v2-yield).

- **`VS_tool.ipynb`**  
  The main notebook implementing the **generative virtual screening workflow**, integrating all functionalities provided in this repository.

- **`output_visualization.py`**  
  A lightweight and elegant script for visualizing the generated results.

---

If you find this repository interesting or useful, feel free to ⭐ star it!
