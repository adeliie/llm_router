# LL-Router Codebase Overview

This repository contains multiple scripts and utilities for training and evaluating different router models.  
The goal is to determine whether a query requires a strong LLM (e.g., GPT-4) or if a weaker/smaller LLM is sufficient, in order to optimize efficiency and performance.

## Main Scripts

- **`train_BERT.py`**  
  Trains a BERT-based classifier on the Arena 55k dataset to determine whether a query should be routed to a strong or weak LLM.  
  The dataset itself does not explicitly label models as "strong" or "weak," so two predefined lists at the beginning of the script map specific models into these categories.  
  The resulting model is stored in `models/router_arena`.

- **`svm.py`, `knn.py`, `knn_arena.py`**  
  Train alternative routing models (SVM or KNN) to predict when a large model is required.  
  The `_arena` suffix indicates that training was performed on the Arena dataset.  
  All results are stored in `models`.

## Caching and Results

- **`small_llm_cache_{dataset_name}`, `small_cache_{dataset_name}`, `routers_results_{dataset_name}`**  
  These files store cached results to avoid repeating computations.

  - `small_llm_cache_*`: results for small models  
  - `routers_results_*`: consolidated router outputs for evaluation

## Evaluation & Visualization

- **`evaluation.py`**  
  Computes all evaluation metrics and prepares data for visualization.  
  The dataset can be easily changed by modifying the `selected_dataset` variable.

- **`router_comparaison_{dataset_name}.png`**  
  Image showing a comparison of router performances.  
  The graph plots the average score based on the proportion of queries routed to strong LLMs.  
  If the curve is above the random baseline (a straight line), the router is considered effective.

- **`plot_results.py`**  
  Generates and saves graphs for different datasets.  
  To visualize results for a specific dataset, replace occurrences of the dataset name (search via Ctrl+F) with the desired dataset.
