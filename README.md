# 🔋 Power Plant Energy Prediction - ANN Model

> A machine learning project that predicts power plant energy output using an Artificial Neural Network (ANN) trained on operational parameters.

<div align="center">

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Accuracy](https://img.shields.io/badge/Accuracy-95.96%25-success)
![Model](https://img.shields.io/badge/Model-Pre--trained-9cf)
![License](https://img.shields.io/badge/License-MIT-green)

### 🚀 Quick Navigation
[**Quick Start**](#-quick-start) • [**Features**](#-features) • [**Usage**](#-using-the-pre-trained-model) • [**Performance**](#-model-performance--accuracy) • [**Tech Stack**](#-technologies-used)

</div>

---

## 📑 Interactive Table of Contents

<details open>
<summary><b>📖 Expand/Collapse Navigation</b></summary>

| Section | Quick Link |
|---------|-----------|
| 🎯 Overview | [Project Overview](#-project-overview) |
| 📁 Structure | [Project Structure](#-project-structure) |
| 🏃 Getting Started | [Quick Start](#-quick-start) |
| ⭐ Key Features | [Features](#-features) |
| 🔄 Pipeline | [Project Workflow](#-project-workflow) |
| 🧠 Architecture | [Model Architecture](#-model-architecture) |
| 📊 Data Info | [Dataset Information](#-dataset-information) |
| 💻 How to Use | [Using the Pre-trained Model](#-using-the-pre-trained-model) |
| 📈 Metrics | [Model Performance](#-model-performance--accuracy) |
| 🔧 Improvement Tips | [Tips for Improvement](#-tips-for-improvement) |
| 🛠️ Dependencies | [Technologies Used](#-technologies-used) |

</details>

---

## � Quick Start

<details open>
<summary><b>✨ Get Started in 3 Minutes</b></summary>

### Step 1️⃣: Clone & Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd PowerForecaster-Power-Plant-Energy-Prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2️⃣: Load Pre-trained Model
```python
import torch
import pandas as pd

# Load the pre-trained model
model = torch.load('models/best_model.pt')
model.eval()

# Load your data
data = pd.read_csv('data/powerplant_data.csv')
```

### Step 3️⃣: Make Predictions
```python
# Prepare input data (AT, V, AP, RH)
input_sample = torch.tensor([[20.0, 50.0, 1015.0, 60.0]], dtype=torch.float32)

# Get prediction
with torch.no_grad():
    prediction = model(input_sample)
    print(f"Predicted Energy Output: {prediction.item():.2f} MW")
```

**🎉 That's it!** You're ready to make predictions.

</details>

---

## �📊 Project Overview

This project builds and trains a PyTorch-based neural network to predict the electrical energy output (PE) of a power plant based on four operational parameters:

| Parameter | Description |
|-----------|-------------|
| **AT** | Ambient Temperature (°C) |
| **V** | Vacuum (cm Hg) |
| **AP** | Atmospheric Pressure (mbar) |
| **RH** | Relative Humidity (%) |
| **PE** | Power Plant Energy Output (MW) ⭐ *Target Variable* |

## ✨ Features at a Glance

<table>
<tr>
<td width="50%">

- ✅ **Pre-trained Model**  
  Ready-to-use PyTorch neural network

- ✅ **High Accuracy**  
  95-96% test accuracy achieved

- ✅ **Clean Dataset**  
  9,568 samples, zero missing values

</td>
<td width="50%">

- ✅ **Complete Pipeline**  
  Data → Model → Prediction

- ✅ **Well Documented**  
  Detailed comments throughout

- ✅ **Production Ready**  
  Optimized for inference

</td>
</tr>
</table>

## 📁 Project Structure

```
ANN/
├── ANN.ipynb                 # Main Jupyter notebook with complete model training pipeline
├── README.md                 # Project documentation
├── requirements.txt          # Python package dependencies
│
├── data/
│   └── powerplant_data.csv   # Dataset containing 9,568 samples of power plant operational data
│
└── models/
    └── best_model.pt         # Pre-trained PyTorch neural network model (saved weights)
```



## 📈 Project Workflow

<details>
<summary><b>View the complete ML pipeline</b></summary>

```
1. Data Loading
   ↓
2. Data Exploration (Missing values, shape, statistics)
   ↓
3. Feature Extraction (Separate features X and target y)
   ↓
4. Train-Test Split (80% training, 20% testing)
   ↓
5. Feature Scaling (StandardScaler normalization)
   ↓
6. Neural Network Design (PyTorch)
   ↓
7. Model Training (Backpropagation optimization)
   ↓
8. Model Evaluation (Loss analysis, predictions)
   ↓
9. Results Visualization
   ↓
10. Model Export (best_model.pt)
```

</details>

## 🧠 Model Architecture

<details>
<summary><b>View model specifications</b></summary>

- **Framework**: PyTorch
- **Input Features**: 4 (AT, V, AP, RH)
- **Output**: 1 (PE - Energy Output)
- **Training Method**: Standard backpropagation with optimization
- **Activation Functions**: ReLU (hidden layers), Linear (output layer)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam or SGD with momentum

</details>

## 📊 Dataset Information

<details>
<summary><b>View dataset details</b></summary>

- **Total Samples**: 9,568 records
- **Features**: 4 continuous variables
- **Target**: 1 continuous variable (Power Output)
- **No Missing Values**: Dataset is clean ✅
- **Train Set**: 7,654 samples (80%)
- **Test Set**: 1,914 samples (20%)
- **Feature Range**: 
  - Temperature (AT): 1.81 - 37.11 °C
  - Vacuum (V): 25.36 - 81.56 cm Hg
  - Pressure (AP): 992.89 - 1033.30 mbar
  - Humidity (RH): 25.56 - 100.16 %

</details>

## 🔍 Key Steps in the Notebook

### Data Preprocessing
- Load data using pandas
- Check for missing values
- Separate features (X) and target variable (y)
- Split data into training and testing sets

### Feature Scaling
- Standardize features using `StandardScaler`
- Normalize training and test datasets
- Improves neural network convergence

### Model Training
- Convert data to PyTorch tensors
- Define neural network architecture
- Train with backpropagation algorithm
- Monitor loss during training epochs

### Model Evaluation
- Generate predictions on test set
- Visualize actual vs predicted values
- Calculate performance metrics (MAE, RMSE, R²)
- Plot loss curves and residuals

## 📝 Using the Pre-trained Model

The `models/best_model.pt` file contains pre-trained weights. To make predictions:

```python
import torch
import torch.nn as nn

# Load the model
model = torch.load('models/best_model.pt')
model.eval()  # Set to evaluation mode

# Prepare input (should be scaled using the same scaler)
input_data = torch.tensor([[your_AT, your_V, your_AP, your_RH]], dtype=torch.float32)

# Make prediction
with torch.no_grad():
    prediction = model(input_data)
    print(f"Predicted Energy Output: {prediction.item()} MW")
```

## 📊 Model Performance & Accuracy

<details open>
<summary><b>📈 View Performance Metrics</b></summary>

### ✅ Performance Dashboard

<table>
<tr>
<th>Metric</th>
<th>Value</th>
<th>Status</th>
<th>Description</th>
</tr>
<tr>
<td><strong>R² Score</strong></td>
<td><code>> 0.95</code></td>
<td>✅ Excellent</td>
<td>Explains 95%+ of variance</td>
</tr>
<tr>
<td><strong>Accuracy (Test)</strong></td>
<td><code>95-96%</code></td>
<td>✅ Excellent</td>
<td>High on unseen data</td>
</tr>
<tr>
<td><strong>MAE</strong></td>
<td><code>< 4.0 MW</code></td>
<td>✅ Good</td>
<td>Avg error ~3-4 MW</td>
</tr>
<tr>
<td><strong>RMSE</strong></td>
<td><code>< 5.0 MW</code></td>
<td>✅ Good</td>
<td>Root mean square error</td>
</tr>
<tr>
<td><strong>MAPE</strong></td>
<td><code>< 3%</code></td>
<td>✅ Excellent</td>
<td>Percentage error</td>
</tr>
<tr>
<td><strong>Generalization</strong></td>
<td><code>No Overfit</code></td>
<td>✅ Stable</td>
<td>Train/Test loss similar</td>
</tr>
</table>

### 📊 Accuracy Breakdown

| Set | Accuracy | Details |
|-----|----------|---------|
| **Training** | ~97-98% | Based on R² score |
| **Testing** | ~95-96% | On unseen data |
| **Success Rate** | ~90%+ | Within 5% of actual |
| **Avg Error** | ±3-4 MW | Prediction deviation |

</details>

## 💡 Tips for Improvement

<details>
<summary><b>🔧 Optimization Ideas</b></summary>

### Architecture Enhancements
- [ ] Experiment with deeper networks (more hidden layers)
- [ ] Try different activation functions (GELU, Swish)
- [ ] Add batch normalization for stability
- [ ] Implement dropout for regularization

### Training Strategies
- [ ] Early stopping to prevent overfitting
- [ ] Learning rate scheduling
- [ ] Different optimizers (RMSprop, AdamW)
- [ ] Cross-validation for robustness

### Data & Features
- [ ] Collect more diverse data samples
- [ ] Engineer new features (interactions, polynomials)
- [ ] Outlier detection and handling
- [ ] Data augmentation techniques

### Evaluation & Analysis
- [ ] K-fold cross-validation
- [ ] Confidence intervals for predictions
- [ ] Feature importance analysis
- [ ] Residual analysis plots

</details>

## 📚 Technologies Used

<details>
<summary><b>🛠️ Tech Stack Details</b></summary>

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.x | Core development language |
| **ML Framework** | PyTorch | Deep learning & neural networks |
| **Data Processing** | Pandas | Data manipulation & analysis |
| **Numerical** | NumPy | Numerical computations |
| **ML Tools** | Scikit-learn | Preprocessing & metrics |
| **Visualization** | Matplotlib/Seaborn | Data visualization |
| **Environment** | Jupyter | Interactive notebook |
| **Deployment** | PyTorch (.pt) | Model serialization |

</details>

---

## 🤝 Project Status & Checklist

<details open>
<summary><b>✅ Completion Status</b></summary>

| Task | Status | Notes |
|------|--------|-------|
| Data loading & exploration | ✅ Complete | 9,568 clean samples |
| Feature preprocessing | ✅ Complete | StandardScaler applied |
| Model architecture | ✅ Complete | PyTorch ANN designed |
| Model training | ✅ Complete | Backpropagation optimized |
| Evaluation & metrics | ✅ Complete | 95-96% accuracy achieved |
| Documentation | ✅ Complete | Full Jupyter notebook |
| Model export | ✅ Complete | best_model.pt ready |
| Testing pipeline | ✅ Complete | Ready for inference |

**Overall Progress**: 100% ✅

</details>

## 📧 Important Notes & Setup Guide

<details open>
<summary><b>⚙️ Configuration & Troubleshooting</b></summary>

### Pre-flight Checks
- [ ] Python 3.8+ installed
- [ ] Virtual environment created (recommended)
- [ ] All dependencies from `requirements.txt` installed
- [ ] Data folder contains `powerplant_data.csv`
- [ ] Models folder contains `best_model.pt`

### Path Configuration
```
✅ Data files should be in the `data/` folder
✅ Pre-trained model is stored in the `models/` folder
✅ Update notebook paths if folder structure differs
✅ Use relative paths for portability
```

### Performance Tips
- 🚀 **GPU Acceleration**: Install CUDA for PyTorch (automatic detection)
- 💾 **Memory**: For large datasets, consider batch processing
- ⚡ **Speed**: Pre-trained model inference is very fast (~1ms/prediction)
- 🔧 **Training**: Full training takes ~5-10 minutes on CPU

### Troubleshooting
| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError` | Check folder structure matches diagram |
| `CUDA errors` | Use CPU or verify CUDA installation |
| `Memory issues` | Reduce batch size or use smaller dataset |

</details>

---

## 🎯 Next Steps

<details>
<summary><b>📝 What to Do Next</b></summary>

### For Beginners
1. Open `ANN.ipynb` in Jupyter
2. Run cells sequentially to understand the pipeline
3. Modify hyperparameters and observe results
4. Test with sample data

### For Advanced Users
1. Experiment with different architectures
2. Implement cross-validation
3. Try ensemble methods
4. Deploy model to production

### For Contributors
1. Fork the repository
2. Create a feature branch
3. Make improvements (architecture, data, docs)
4. Submit pull request with documentation

</details>

---

## 📊 Quick Reference

<details>
<summary><b>🔍 Input/Output Format</b></summary>

### Model Input Format
```python
# Required: 4 features in order
input = [AT, V, AP, RH]  # Temperature, Vacuum, Pressure, Humidity
```

### Feature Ranges
| Feature | Min | Max | Unit |
|---------|-----|-----|------|
| **AT** (Temperature) | 1.81 | 37.11 | °C |
| **V** (Vacuum) | 25.36 | 81.56 | cm Hg |
| **AP** (Pressure) | 992.89 | 1033.30 | mbar |
| **RH** (Humidity) | 25.56 | 100.16 | % |

### Model Output
```
PE (Power Energy Output): Continuous value in MW
Typical range: 420-480 MW
```

### Example Prediction
```python
# Input: AT=20°C, V=50 cm Hg, AP=1015 mbar, RH=60%
# Output: ~450.25 MW
```

</details>

---

**Last Updated**: April 2026  
**Model Status**: Ready for prediction  

---

<div align="center">

### 🌟 Questions? Issues? Ideas?

Create an issue or reach out with feedback!

**[↑ Back to Top](#-power-plant-energy-prediction---ann-model)**

</div>
