<div align="center">

<!-- Animated Typing Header -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=35&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&width=900&height=100&lines=%F0%9F%9A%97+Car+Price+Prediction;End-to-End+Machine+Learning+Project" alt="Typing SVG" /></a>

<br/>

<!-- Animated Wave Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=200&section=header&text=Car%20Price%20Prediction&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=A%20beginner-friendly%20teaching%20notebook%20%7C%20From%20Raw%20CSV%20to%20Trained%20Model&descSize=18&descAlignY=55&descColor=58a6ff" width="100%"/>

<br/>

<!-- Status Badges -->
![GitHub last commit](https://img.shields.io/github/last-commit/JayanGupta/Car_Price_Predicting_Model?style=for-the-badge&color=00d9ff&labelColor=0d1117)
![GitHub repo size](https://img.shields.io/github/repo-size/JayanGupta/Car_Price_Predicting_Model?style=for-the-badge&color=7c3aed&labelColor=0d1117)
![GitHub stars](https://img.shields.io/github/stars/JayanGupta/Car_Price_Predicting_Model?style=for-the-badge&color=fbbf24&labelColor=0d1117)
![GitHub forks](https://img.shields.io/github/forks/JayanGupta/Car_Price_Predicting_Model?style=for-the-badge&color=34d399&labelColor=0d1117)
![Profile Views](https://komarev.com/ghpvc/?username=JayanGupta-CarPrice&label=Repository+Views&color=f472b6&style=for-the-badge)

</div>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🧠 About This Project

> *A complete, end-to-end Machine Learning pipeline predicting used car resale prices — designed as a classroom-ready resource for beginners.*

This project demonstrates the entire Data Science lifecycle using the **CarDekho Used Cars** dataset. It serves as a comprehensive teaching tool, emphasizing clean code, robust data pipelines (preventing data leakage), and clear pedagogical explanations for every step of the process.

<div align="center">

| 🎯 Goal | 📊 Dataset | 🏆 Champion Model | 📈 Accuracy (R²) |
|:--------|:-----------|:------------------|:-----------------|
| Predict Resale Price | 2,059 listings × 20 features | Lasso Regression (L1) | **89.5%** |

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🗺️ Machine Learning Pipeline

The project follows a structured, production-aligned approach to model development. Each phase is detailed in the accompanying Jupyter Notebook with "What, Why, and How" explanations.

```mermaid
graph LR
    A["📥 Load Data"] --> B["🧹 Clean & Parse"]
    B --> C["📊 EDA"]
    C --> D["🛠️ Engineer Features"]
    D --> E["🔀 Train/Test Split"]
    E --> F["🏗️ Build Pipeline"]
    F --> G["🏋️ Train Models"]
    G --> H["📈 Evaluate"]
    H --> I["🏆 Champion Model"]

    style A fill:#0d1117,stroke:#00d9ff,color:#ffffff
    style B fill:#0d1117,stroke:#7c3aed,color:#ffffff
    style C fill:#0d1117,stroke:#34d399,color:#ffffff
    style D fill:#0d1117,stroke:#fbbf24,color:#ffffff
    style E fill:#0d1117,stroke:#f472b6,color:#ffffff
    style F fill:#0d1117,stroke:#00d9ff,color:#ffffff
    style G fill:#0d1117,stroke:#7c3aed,color:#ffffff
    style H fill:#0d1117,stroke:#34d399,color:#ffffff
    style I fill:#0d1117,stroke:#fbbf24,color:#ffffff
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ❓ Q&A Notebook Outline

The notebook is structured around key questions guiding the learning process:

1. **Q1.** How do we load our dataset and get our first look at the data?
2. **Q2.** What are the data types and where are the missing values?
3. **Q3.** Feature Engineering: How do we clean string features into usable numbers?
4. **Q4.** Logical Imputation: How do we handle missing values for Electric Vehicles?
5. **Q5.** EDA: What does the distribution of our target variable (Price) look like?
6. **Q6.** EDA: How does the car price vary based on the fuel type?
7. **Q7.** EDA: Does Transmission Type Affect Price?
8. **Q8.** EDA: Who are you buying from?
9. **Q9.** EDA: The Depreciation Curve (Year vs Price)
10. **Q10.** EDA: Which numerical features are highly correlated with Price?
11. **Q11.** Splitting the Data
12. **Q12.** Constructing the Preprocessing Pipeline
13. **Q13.** Training the Models within the Pipeline
14. **Q14.** Evaluating the Models
15. **Q15.** Saving the Pipeline for Future Data

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🛠️ Feature Engineering Impact

Raw data often lacks the nuanced signals required for high-performing models. By applying domain knowledge, three critical features were engineered, resulting in a measurable performance boost.

<div align="center">

| Engineered Feature | Logic | Business Intuition |
|:-------------------|:------|:-------------------|
| `Car_Age` | `2026 - Year` | Directly captures depreciation |
| `Power_per_CC` | `Max_Power_bhp / Engine_cc` | Engine efficiency (identifies turbocharged engines) |
| `Km_per_Year` | `Kilometer / Car_Age` | Usage intensity (highway cruising vs city driving) |

<br>

**Performance Gain Breakdown**

| Model Configuration | R² Score | MAE (INR) |
|:--------------------|:--------:|:---------:|
| Baseline (Raw Features) | 0.8848 | ₹4,00,064 |
| **+ Engineered Features** | **0.8948** | **₹3,76,957** |

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 📈 Results & Evaluation

The predictive models were integrated within strict Scikit-Learn `Pipeline` architectures to prevent data leakage and handle complex transformations (like log-transforming the highly skewed target variable).

### 🏆 Model Comparison

| Model | Regularization | MAE (INR) | RMSE (INR) | R² Score | Verdict |
|:------|:--------------:|----------:|-----------:|---------:|:-------:|
| **Lasso Regression** | L1 | ₹3,76,957 | ₹8,57,226 | **0.8948** | 🏆 Champion |
| Ridge Regression | L2 | ₹3,27,841 | ₹1,294,528| 0.7601 | Runner-up |

### 🔑 Top Predictive Drivers

Using Lasso's coefficient penalization, the model automatically selected the most impactful features:

1. **`Car_Age` (📉 Negative)**: Depreciation is the strongest factor reducing value.
2. **`Transmission_Manual` (📉 Negative)**: Manual vehicles consistently price lower than automatics.
3. **`Max_Power_bhp` (📈 Positive)**: Higher performance directly correlates with higher resale prices.
4. **`Make_Land Rover` (📈 Positive)**: Premium luxury brands maintain a significant base premium.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 💻 Tech Stack & Tools

<div align="center">

<p>
<img src="https://skillicons.dev/icons?i=python,vscode,jupyter,git,github&theme=dark" height="50"/>
</p>

<p>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" />
</p>

<p>
<img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white" />
<img src="https://img.shields.io/badge/Lasso_(L1)-00d9ff?style=for-the-badge&labelColor=0d1117" />
<img src="https://img.shields.io/badge/Ridge_(L2)-7c3aed?style=for-the-badge&labelColor=0d1117" />
<img src="https://img.shields.io/badge/Sklearn_Pipeline-34d399?style=for-the-badge&labelColor=0d1117" />
</p>

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/JayanGupta/Car_Price_Predicting_Model.git
cd Car_Price_Predicting_Model

# 2. Open the notebook
jupyter notebook CarPrice_Prediction_Project.ipynb

# 3. Run all cells sequentially (Kernel → Restart & Run All)
```

### ☁️ Run on Google Colab
1. Upload `CarPrice_Prediction_Project.ipynb` and `CarDekho.csv` to Google Colab.
2. Run all cells — no additional setup is required.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 📊 Dataset Overview

The dataset consists of **2,059** used car listings from CarDekho. 

**Key Features Include:**
- **Categorical**: `Make`, `Fuel Type`, `Transmission`, `Owner`, `Seller Type`, `Drivetrain`
- **Numerical**: `Kilometer`, `Engine_cc`, `Max_Power_bhp`, `Length`, `Width`, `Height`, `Seating Capacity`, `Fuel Tank Capacity`

*(Note: High cardinality features like `Model` and `Location` were dropped during the feature selection phase to mitigate the curse of dimensionality and prevent overfitting.)*

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🤝 Contributing & Support

Contributions, issues, and feature requests are welcome! 

1. 🍴 Fork the Project
2. 🌿 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your Changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

**If you found this project helpful for learning Machine Learning, please give it a ⭐!**

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=JayanGupta/Car_Price_Predicting_Model&type=Date&theme=dark)](https://star-history.com/#JayanGupta/Car_Price_Predicting_Model&Date)

</div>

---

<div align="center">

<!-- Animated Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=120&section=footer" width="100%"/>

<br/>

<b>Made with ❤️ by <a href="https://github.com/JayanGupta">Jayan Gupta</a></b>

<br/><br/>

<img src="https://forthebadge.com/images/badges/built-with-love.svg" />
<img src="https://forthebadge.com/images/badges/made-with-python.svg" />
<img src="https://forthebadge.com/images/badges/open-source.svg" />

</div>
