# 🛠️ Predictive Maintenance for Turbofan Engines (CMAPSS)

This project implements a predictive maintenance model for aircraft turbofan engines using the NASA C-MAPSS dataset. The goal is to estimate the **Remaining Useful Life (RUL)** of engines based on multivariate time-series sensor data.

> ✅ Built entirely using Python, Pandas, Matplotlib, Seaborn, and Scikit-Learn  
> ✅ Fully operable via the command line (no Jupyter Notebook required)

---

## 🚀 Project Overview

### 🔍 What it does:
- Preprocesses raw sensor data
- Calculates RUL for each engine over time
- Selects meaningful sensor features
- Trains a machine learning model (Linear Regression / Gradient Boosting)
- Evaluates with metrics and residual plots
- Saves prediction results for visualization and analysis

---

## 📂 Folder Structure

RUL_estimaton/
├── data/
│ ├── train_FD001.txt ← Training data (until failure)
│ ├── test_FD001.txt ← Test data (truncated)
│ └── RUL_FD001.txt ← Actual RUL for test engines
├── src/
│ └── main.py ← Main model pipeline
├── output/
│ ├── rul_predictions.csv ← Prediction vs actual
│ └── rul_residuals.png ← Error histogram
├── run.sh ← Shell script to automate run
├── requirements.txt
└── README.md


---

## 🧠 Dataset Info: NASA CMAPSS (FD001)

- 📊 100 engines
- 🕐 Each row = engine at a single time step
- 3 operating settings + 21 sensors
- Engine fails at end of each time series in training data

📥 Download from Kaggle or NASA:
https://data.nasa.gov/d/xaut-bemq  
Kaggle mirror: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

---

## 🧪 Example Features Used

- Sensor readings (e.g., temperature, pressure)
- Derived features like engine speed, acceleration
- Removed constant/noisy features based on domain knowledge

---

## 📦 Installation & Setup

```bash
# Clone the project
git clone https://github.com/BlackVenus2003/Turbofan_RUL_Estimation.git
cd Turbofan_RUL_Estimation

```
## Run the Model

python src/main.py

This will:

Load and clean data

Train the regression model

Predict RUL

Save results and residual plot to 'output/'

## 📊 Evaluation Metrics

Mean Squared Error (MSE)

Residual Analysis

Future scope: R² Score, MAE, custom domain thresholds








