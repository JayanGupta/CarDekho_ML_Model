<p align="center">
  <h1 align="center">🚗 Car Price Prediction — End-to-End Machine Learning Project</h1>
  <p align="center">
    A complete, beginner-friendly <strong>teaching notebook</strong> that walks you through every step of the Machine Learning lifecycle — from raw CSV to trained model — using the <strong>CarDekho Used Cars</strong> dataset.
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"></a>
  <a href="https://colab.research.google.com/"><img src="https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logoColor=white" alt="Seaborn">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Lasso-Regression-blue?style=flat-square" alt="Lasso">
  <img src="https://img.shields.io/badge/Ridge-Regression-blue?style=flat-square" alt="Ridge">
  <img src="https://img.shields.io/badge/Pipeline-ColumnTransformer-orange?style=flat-square" alt="Pipeline">
  <img src="https://img.shields.io/badge/Feature-Engineering-brightgreen?style=flat-square" alt="Feature Engineering">
</p>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Skills & Concepts Covered](#-skills--concepts-covered)
- [Dataset](#-dataset)
- [Notebook Structure](#-notebook-structure)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Custom Predictions](#-custom-predictions)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🎯 About the Project

This is a **teaching-first** Jupyter Notebook designed to introduce absolute beginners to the world of Machine Learning through a real-world business problem: **predicting the resale price of used cars**.

Every section of the notebook follows a structured lesson format:

| Step | Purpose |
|:-----|:--------|
| **What is it?** | Explains the concept in simple language |
| **Why do we need it?** | Provides real-world business justification |
| **Code** | Simplest possible implementation with comments |
| **What did we observe?** | Explains the output |
| **What did we learn?** | Summarizes the takeaway |

> **Note**: This is NOT a Kaggle competition notebook or a production ML system. This is a classroom-ready teaching resource.

---

## 🧠 Skills & Concepts Covered

<table>
<tr>
<td>

### 📊 Data Science
- Exploratory Data Analysis (EDA)
- Histograms, Scatter Plots, Heatmaps
- Correlation Analysis
- Outlier Detection & Discussion

</td>
<td>

### 🧹 Data Cleaning
- Duplicate Detection
- String Parsing (`str.extract`)
- Missing Value Investigation
- MCAR / MAR / MNAR Analysis
- Structural Missingness (Electric Vehicles)

</td>
</tr>
<tr>
<td>

### 🛠️ Feature Engineering
- `Car_Age` (Depreciation signal)
- `Power_per_CC` (Engine efficiency ratio)
- `Km_per_Year` (Annual usage intensity)
- Feature Selection (Curse of Dimensionality)

</td>
<td>

### 🤖 Machine Learning
- Lasso Regression (L1 Regularization)
- Ridge Regression (L2 Regularization)
- Train/Test Split (Data Leakage Prevention)
- Model Evaluation (MAE, RMSE, R²)
- Model Interpretation (Coefficients)

</td>
</tr>
<tr>
<td colspan="2">

### ⚙️ Scikit-Learn Pipeline Components
`Pipeline` · `ColumnTransformer` · `SimpleImputer` · `StandardScaler` · `OneHotEncoder` · `TransformedTargetRegressor` · `PredictionErrorDisplay` · `sklearn.base.clone`

</td>
</tr>
</table>

---

## 📁 Dataset

| Property | Value |
|:---------|:------|
| **Source** | CarDekho (Used Car Listings) |
| **File** | `CarDekho.csv` |
| **Rows** | 2,059 |
| **Columns** | 20 |
| **Target Variable** | `Price` (INR) |

### Feature Glossary

| Feature | Business Meaning | Type |
|:--------|:-----------------|:-----|
| `Make` | Car manufacturer (Honda, Hyundai, etc.) | Categorical |
| `Model` | Specific model name | Categorical (dropped — high cardinality) |
| `Price` | 🎯 Resale price in INR | Numerical (Target) |
| `Year` | Manufacturing year | Numerical → engineered to `Car_Age` |
| `Kilometer` | Total kilometers driven | Numerical |
| `Fuel Type` | Petrol, Diesel, CNG, Electric, etc. | Categorical |
| `Transmission` | Manual or Automatic | Categorical |
| `Location` | City of listing | Categorical (dropped — high cardinality) |
| `Color` | Exterior color | Categorical (dropped) |
| `Owner` | First, Second, Third, etc. | Categorical |
| `Seller Type` | Individual, Corporate, Commercial | Categorical |
| `Engine` | Engine displacement (e.g., "1198 cc") | Text → parsed to `Engine_cc` |
| `Max Power` | Peak power (e.g., "87 bhp @ 6000 rpm") | Text → parsed to `Max_Power_bhp` |
| `Max Torque` | Peak torque | Text (dropped for simplicity) |
| `Drivetrain` | FWD, RWD, AWD | Categorical |
| `Length / Width / Height` | Vehicle dimensions (mm) | Numerical |
| `Seating Capacity` | Number of seats | Numerical |
| `Fuel Tank Capacity` | Tank size in liters | Numerical |

### Engineered Features

| Feature | Formula | Business Intuition |
|:--------|:--------|:-------------------|
| `Car_Age` | `2026 - Year` | Captures depreciation — older = cheaper |
| `Power_per_CC` | `Max_Power_bhp / Engine_cc` | Engine efficiency — turbo vs. naturally aspirated |
| `Km_per_Year` | `Kilometer / Car_Age` | Usage intensity — highway cruiser vs. city crawler |

---

## 📓 Notebook Structure

The notebook follows the complete Machine Learning lifecycle across **18 structured sections**:

```
 1. 🌟 Project Introduction
 2. 🎯 Problem Statement
 3. 📦 Import Libraries
 4. 📥 Load Dataset
 5. 🔍 Understand Dataset
 6. 🧹 Data Cleaning
       ├── Duplicate Check
       ├── String Parsing (Engine, Max Power)
       └── Missing Value Investigation (Structural vs Random)
 7. 📊 Exploratory Data Analysis
       ├── Price Distribution (Histogram)
       ├── Kilometer vs Price (Scatter + Outlier Discussion)
       ├── Fuel Type Counts (Countplot)
       └── Correlation Matrix (Heatmap)
 8. 🛠️ Feature Engineering
       ├── Car_Age (from Year)
       ├── Power_per_CC (engine efficiency ratio)
       └── Km_per_Year (annual usage intensity)
 9. ⚙️ Data Preprocessing (Encoding & Scaling Concepts)
10. 🎯 Feature Selection
11. 🔀 Split Data
       ├── Separate X and y
       ├── Train/Test Split (80/20)
       └── Impute Structural Missingness (EVs → 0)
12. 🏗️ Model Building
       ├── ColumnTransformer + Pipeline
       ├── Lasso Pipeline (TransformedTargetRegressor)
       └── Ridge Pipeline (clone + TransformedTargetRegressor)
13. 🏋️ Model Training
14. 🔮 Predictions
15. 📈 Model Evaluation
       ├── Metrics (MAE, RMSE, R²)
       └── PredictionErrorDisplay (Actual vs Predicted, Residuals)
16. 🧠 Feature Importance / Model Interpretation
       ├── Extract Coefficients
       ├── Top 10 Coefficients Plot
       └── Positive & Negative Drivers
17. 🏁 Conclusion
       ├── Champion Model Summary
       └── 🔧 Custom Car Prediction (Interactive Placeholder)
18. 🎓 Key Learnings (ML Vocabulary Dictionary)
```

---

## 📈 Results

### Model Comparison

| Model | MAE (INR) | RMSE (INR) | R² Score |
|:------|----------:|-----------:|---------:|
| **Lasso Regression** | 376,957 | 857,226 | **0.8948** |
| Ridge Regression | **327,841** | 1,294,528 | 0.7601 |

> **Champion Model**: Lasso Regression — explains **~89.5%** of used car price variation.

### Impact of Feature Engineering

| Version | R² Score | Change |
|:--------|:--------:|:------:|
| Baseline (no engineered features) | 0.8848 | — |
| + `Power_per_CC` + `Km_per_Year` | **0.8948** | **+1.0 pp** ⬆️ |

### Key Price Drivers (Lasso Coefficients)

| Rank | Feature | Direction | Business Interpretation |
|:----:|:--------|:---------:|:------------------------|
| 1 | `Car_Age` | ⬇️ Negative | Older cars lose value (depreciation) |
| 2 | `Transmission_Manual` | ⬇️ Negative | Manual cars sell for less than automatic |
| 3 | `Max_Power_bhp` | ⬆️ Positive | More powerful engines = higher price |
| 4 | `Make_Land Rover` | ⬆️ Positive | Luxury brand commands premium |
| 5 | `Fuel Type_Petrol` | ⬇️ Negative | Petrol cars are priced lower than diesel on average |

### Custom Prediction Example

```
Input:  Honda | 45,000 km | Petrol | Manual | First Owner | FWD | 1198 cc | 87 bhp | 5 years old
        Power_per_CC = 0.073 | Km_per_Year = 9,000
Output: ₹ 7,87,608 INR (predicted resale value)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/CarDekho-Price-Prediction.git
   cd CarDekho-Price-Prediction
   ```

2. Open the notebook:
   ```bash
   jupyter notebook CarPrice_Prediction_Project.ipynb
   ```

3. Run all cells sequentially (Kernel → Restart & Run All).

### Run on Google Colab

1. Upload `CarPrice_Prediction_Project.ipynb` and `CarDekho.csv` to Colab.
2. Run all cells — no additional setup needed!

---

## 🔧 Custom Predictions

Navigate to **Section 17.2** in the notebook. You will find a ready-to-use placeholder:

```python
custom_car = pd.DataFrame([{
    'Make': 'Honda',
    'Kilometer': 45000,
    'Fuel Type': 'Petrol',
    'Transmission': 'Manual',
    'Owner': 'First',
    'Seller Type': 'Individual',
    'Drivetrain': 'FWD',
    'Engine_cc': 1198.0,
    'Max_Power_bhp': 87.0,
    'Length': 3990.0,
    'Width': 1680.0,
    'Height': 1505.0,
    'Seating Capacity': 5.0,
    'Fuel Tank Capacity': 35.0,
    'Car_Age': 5.0,
    'Power_per_CC': 87.0 / 1198.0,  # Derived: bhp / cc
    'Km_per_Year': 45000 / 5.0      # Derived: km / age
}])

predicted_val = model_lasso.predict(custom_car)[0]
print(f'Predicted Resale Value: {predicted_val:,.2f} INR')
```

Simply modify the values in the dictionary and re-run the cell!

---

## 📂 Project Structure

```
CarDekho/
├── CarDekho.csv                        # Raw dataset (2,059 listings × 20 columns)
├── CarPrice_Prediction_Project.ipynb   # Complete teaching notebook
└── README.md                           # This file
```

---

## 🛡️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logoColor=white" alt="Seaborn">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab">
</p>

---

## 📜 License

This project is licensed under the MIT License. Feel free to use it for teaching, learning, and portfolio purposes.

---

<p align="center">
  <strong>⭐ If this notebook helped you learn Machine Learning, give it a star!</strong>
</p>
