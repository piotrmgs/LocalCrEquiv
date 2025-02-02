# Local $C^r$-Equivalence of Loss Landscapes: Code Repository

This repository contains the implementation of the experiments presented in the paper "Local $C^r$-Equivalence of Loss Landscapes: A Geometric Approach to Deep Learning Optimization." It includes data preprocessing, model training for both regression (MSE) and classification (Cross-Entropy) tasks, and code for verifying local $C^r$-equivalence.

---

# 1. Dataset Preparation

File: `dataset_preprocessing.py`

This script handles data loading and preprocessing for both experiments:
- California Housing Dataset (Regression with MSE Loss)
- CIFAR-10 Dataset (Classification with Cross-Entropy Loss)

Functions:

- load_california_housing_data(test_size=0.2, batch_size=64)
  Description:
    - Loads the California Housing dataset from sklearn.datasets.
    - Splits data into training and testing sets (default: 80/20 split).
    - Standardizes features (zero mean, unit variance).
    - Converts data to PyTorch tensors and returns DataLoaders.

- load_cifar10_data(batch_size=64)
  Description:
    - Downloads the CIFAR-10 dataset using torchvision.datasets.
    - Applies basic data augmentation: random horizontal flip, random crop, normalization.
    - Returns training and testing DataLoaders.

Usage Example:

Run the script:
`python dataset_preprocessing.py`

Sample output:
California Housing: Train batches: 206 Test batches: 52
CIFAR-10: Train batches: 782 Test batches: 157

---

# 2. California Housing Regression (MLP with MSE Loss)

File: `train_california_mlp.py`

This script defines and trains a simple Multilayer Perceptron (MLP) for the California Housing dataset.

Model Architecture:
- Input Layer: 8 features
- Hidden Layers: Two fully connected layers with 64 ReLU units each
- Output Layer: Single neuron for regression output

Key Components:
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate = 0.001
- Epochs: 30 (modifiable)

Saving the Model:
At the end of training, the model’s parameters are saved as:
model_mlp_california.pt

Note: Replace [...] in the script with the appropriate path where the model should be saved:
folder_path = "[...]"  # Replace with your desired directory path

Run the Script:
`python train_california_mlp.py`

Sample output:
Epoch [1/30] - Train Loss: 0.7234
...
Final Test MSE: 0.2868
Saved MLP model parameters to model_mlp_california.pt

---

# 3. CIFAR-10 Classification (CNN with Cross-Entropy Loss)

File: `train_cifar_cnn.py`

This script defines and trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

Model Architecture:
- Convolutional Layers:
  - Conv1: 32 filters, kernel size 3
  - Conv2: 64 filters, kernel size 3
  - MaxPooling: 2x2
- Fully Connected Layers:
  - FC1: 256 units (ReLU activation)
  - FC2: 10 output classes (for CIFAR-10)

Key Components:
- Loss Function: Cross-Entropy Loss
- Optimizer: Adam with learning rate = 0.001
- Epochs: 20 (modifiable)

Saving the Model:
At the end of training, the model’s parameters are saved as:
model_cnn_cifar10.pt

Note: Replace [...] in the script with the appropriate path where the model should be saved:
folder_path = "[...]"  # Replace with your desired directory path

Run the Script:
`python train_cifar_cnn.py`

Sample output:
Epoch [1/20] - Train Loss: 1.8621, Train Acc: 32.45%
...
Final Test Loss: 0.7591, Test Accuracy: 73.43%
Saved CNN model parameters to model_cnn_cifar10.pt

---

# 4. Output Files

Script: `train_california_mlp.py`
Generated File: `model_mlp_california.pt`
Description: Trained MLP model for California Housing

Script: `train_cifar_cnn.py`
Generated File: `model_cnn_cifar10.pt`
Description: Trained CNN model for CIFAR-10

---

# 5. Verification of Local $C^r$-Equivalence (MSE Experiment)

File: `check_cr_equiv_param_space.py`

This script verifies the local $C^r$-equivalence condition for the MSE loss landscape using the trained MLP model on the California Housing dataset.

How to Run:
1. Make sure the trained model file `model_mlp_california.pt` exists.
2. Specify the correct paths:
   - Replace `[...]` in `model_path` with the path to `model_mlp_california.pt`.
   - Replace `[...]` in `output_path` with the directory where you want to save the results.
3. Run the script: `python check_cr_equiv_param_space.py`.

Example output:
- A text file `ratios_results.txt` containing:
  - General information about the experiment
  - Statistical analysis results (mean ratios, standard deviations, normality tests, ANOVA, Kruskal-Wallis test)
  - Summary of findings

- Visualization files saved in the specified output directory:
  - `confidence_intervals.png`: Confidence intervals for inequality coefficients
  - `kruskal_results.png`: Kruskal-Wallis test results visualization
  - `anova_and_normality.png`: ANOVA and normality tests visualization
  - `ratios_vs_scale_with_std.png`: Dependency of inequality coefficient on perturbation scale
  - `ratios_boxplot.png`: Distribution of inequality coefficients across perturbation scales
  - `iteration_times.png`: Execution time for each iteration

- The results include information about execution time, GPU usage, and detailed statistical outputs for analysis.

---

# 6. Verification of Local $C^r$-Equivalence (Cross-Entropy Experiment)

File: `cifar10_crossentropy_verification.py`

This script verifies the local $C^r$-equivalence condition for the Cross-Entropy loss landscape using the trained CNN model on the CIFAR-10 dataset.

How to Run:
1. Ensure the trained model file `model_cnn_cifar10.pt` exists.
2. Specify the correct paths:
   - Replace `[...]` in `model_path` with the path to `model_cnn_cifar10.pt`.
   - Replace `[...]` in `output_path` with the directory where you want to save the results.
3. Run the script: `cifar10_crossentropy_verification.py`.

Example output:
- A text file `ratios_results.txt` containing:
  - General information about the experiment
  - Statistical analysis results (mean ratios, standard deviations, normality tests, ANOVA, Kruskal-Wallis test)
  - Summary of findings

- Visualization files saved in the specified output directory:
  - `confidence_intervals.png`: Confidence intervals for inequality coefficients
  - `kruskal_results.png`: Kruskal-Wallis test results visualization
  - `anova_and_normality.png`: ANOVA and normality tests visualization
  - `ratios_vs_scale_with_std.png`: Dependency of inequality coefficient on perturbation scale
  - `ratios_boxplot.png`: Distribution of inequality coefficients across perturbation scales
  - `iteration_times.png`: Execution time for each iteration

- The results include information about execution time, GPU usage, and detailed statistical outputs for analysis.

---
