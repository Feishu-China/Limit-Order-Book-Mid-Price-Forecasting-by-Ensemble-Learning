# High Frequency Trading - Codebase

---

## Problem Description

This project focuses on **High Frequency Trading (HFT)** strategies using **Limit Order Book (LOB)** data. The primary goal is to predict short-term future returns based on the microstructure of the market.

The challenge involves processing massive amounts of tick-level data, extracting meaningful features from the order book state (prices, volumes, order flows), and building robust predictive models that can generalize well to unseen data.

---

### Task Definition

The core task is to predict a future financial metric (e.g., return or price movement direction) given a sequence of historical LOB states.

- **Input**: A time series of LOB updates (Ask Prices, Bid Prices, Ask Sizes, Bid Sizes, per level).
- **Output**: A continuous value representing the predicted future return (`y`).

---

### Key Constraints

1.  **Latency**: Feature generation and inference must be highly optimized for real-time applications.
2.  **Robustness**: Models must be resilient to market noise and regime changes.
3.  **Interpretability**: Understanding feature importance is crucial for strategy refinement.

---

### Dependencies

The implementation relies on a standard Python scientific computing stack, with specific optimizations for PyTorch.

#### Core Scientific Computing
- **numpy**: Numerical array operations.
- **pandas**: Time-series manipulation and feature aggregation.

#### Visualization
- **matplotlib**: Plotting performance metrics and feature analysis.
- **rich**: Enhanced console output and progress bars.

#### Machine Learning & Deep Learning
- **torch**: PyTorch for optimized Lasso and GRU models (supports GPU/AMP).
- **scikit-learn**: Metrics (R2) and preprocessing (RobustScaler).
- **scipy**: Statistical functions (Winsorization) and optimization (SLSQP for ensembling).

---

## Code Structure

The codebase has been refactored into a consolidated `code/` directory for better organization and maintainability.

### 1. `function.py`
This is the central library containing all core functionality. The module is structurally divided into distinct domain sections:

-   **UTILS**:
    -   `winsorize_series`: Robust outlier handling.
    -   `_rolling_sum`: Efficient rolling sum utility.
    -   `generate_features_vectorized`: High-performance feature engineering (Imbalance, Momentum, Order Flow, etc.).
    -   `SlidingDataset` & `construct_dataloader`: PyTorch dataset and configurable dataloader.

-   **MODELS**:
    -   `GRUModel`: A Gated Recurrent Unit neural network for sequence modeling.
    -   `OptimizedLassoRegression`: Fast Lasso implementation optimized for GPU and mixed precision.
    -   `l1_regularization`: L1 penalty computation module for Lasso optimization.

-   **TRAINING LOGIC**:
    -   `train_lasso_optimized`: Efficient PyTorch Lasso training loop leveraging AMP and early stopping.

-   **PREDICTION LOGIC**:
    -   `run_prediction_nn`: Scaled end-to-end inference pipeline for neural networks.

-   **ANALYSIS & VISUALIZATION**:
    -   `analyze_feature_correlations_final`: Calculates Pearson correlations between LOB features and targets.
    -   `plot_tornado`: Visualizes feature importance using Tornado Charts.
    -   `plot_performance`: Generates hexbin and regression plots comparing predicted vs. actual performance.
    -   `optimize_weights`: Finds optimal ensemble weights using SLSQP.
    -   `load_prediction`: Utility to read and standardize score predictions.

-   **ADDITIONAL MODELS & TRAINING**:
    -   `MixerBlock` & `MLPMixer`: Multi-Layer Perceptron Mixer architecture.
    -   `train_nn_model`: Generic neural network training loop.
    -   `train_xgb_pytorch`: XGBoost model training adapted for evaluation on partitioned data.
    -   `SimpleLassoRegression`: Standard Lasso basic implementation.
    -   `train_lasso_simple`: Loop to train the standardized PyTorch lasso variant.
    -   `model_predict_xgb`: Inference utility for trained XGBoost models.
    -   `construct_predictions`: Aligns predictions with valid input timestamps efficiently.
    -   `model_pred_nn`: Flexible NN inference allowing dynamic model architecture selection.
    -   `model_predict_lasso_pytorch`: PyTorch-based Lasso inference handling.
    -   `ensemble_predictions`: Weighted or unweighted aggregation of multiple predictive results.

-   **ADVANCED MODELS & UTILS**:
    -   `GRUModelMagic`: Advanced GRU targeting specific feature combinations for performance gains.
    -   `MLPMixerMagic`: Tailored MLPMixer utilizing learnable weights on specialized linear feature combinations.
    -   `get_robust_scaler`: Automated initialization or retrieval of robust feature scalers.

### 2. `visualization.ipynb`
A Jupyter Notebook that demonstrates the usage of `function.py`. It is used for:
-   Loading pre-computed predictions.
-   Optimizing ensemble weights.
-   Visualizing model performance.
-   Analyzing feature importance (Tornado Charts).

---

## Usage

### Setup
Ensure all dependencies are installed and the data files are located in the parent directory (`../`).

### Importing Functions
You can import all necessary tools directly from `function.py`:

```python
import sys
import os
sys.path.append(os.getcwd())

from function import *

# Example: Loading a model
model = GRUModel(input_dim=48, hidden_dim=64)
```

### Running the Visualization Notebook
Launch Jupyter Lab or Notebook and open `code/visualization.ipynb`. Run the cells to generate performance reports and plots.

```bash
jupyter lab code/visualization.ipynb
```

---

## Model Interpretability

We use **Tornado Charts** to visualize the correlation of individual features with the target variable.

-   **Positive Correlation (Red)**: Features that tend to move in the same direction as the target.
-   **Negative Correlation (Blue)**: Features that tend to move in the opposite direction.

Analysis functions in `function.py` (e.g., `analyze_feature_correlations_final`) automatically identify feature groups (Ask/Bid Rate, Size, etc.) and compute these metrics.

---

## Acknowledgment

This codebase refactoring consolidates efforts to improve the maintainability and performance of the HFT project.

