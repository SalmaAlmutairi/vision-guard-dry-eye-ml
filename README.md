# Vision Guard: AI-Powered Dry Eye Prediction and Prevention

An end-to-end machine learning project that explores whether **Dry Eye Disease (DED) risk can be predicted using lifestyle and behavioral data** without relying solely on clinical measurements.

The project combines **supervised learning, unsupervised learning, representation learning, and Generative AI** to analyze lifestyle patterns, predict DED risk, identify behavioral clusters, and generate personalized lifestyle recommendations.

> **Note:** This project was developed for research and educational purposes and is not intended for medical diagnosis.

---

## Project Overview

Most Dry Eye Disease prediction studies rely heavily on clinical measurements, medical imaging, or laboratory data.

Vision Guard explores an alternative approach by using lifestyle-related information such as:

- Sleep duration and quality
- Stress level
- Average screen time
- Smart device use before bed
- Blue-light filter usage
- Physical activity and health indicators
- Eye-related symptoms

The project investigates whether these behavioral patterns contain enough information to support DED risk prediction and personalized lifestyle recommendations.

---

## Dataset

The dataset was obtained from **Kaggle** and contains:

- **20,000 records**
- **26 features**
- Numerical and categorical variables
- Self-reported lifestyle, health, environmental, and eye-related information

### Target Distribution

| Class | Samples |
|---|---:|
| DED Positive | 13,037 |
| DED Negative | 6,963 |

The dataset contained **no missing values, no duplicate records, and no significant outliers**.

Most feature correlations were relatively weak, indicating limited linear relationships between lifestyle variables.

Dataset file:

[`data/Dry_Eye_Dataset.csv`](data/Dry_Eye_Dataset.csv)

---

## Machine Learning Pipeline

```text
Lifestyle Dataset
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Encoding + Feature Engineering + Scaling
       ↓
Class Imbalance Handling (SMOTE)
       ↓
 ┌─────────────────────────────┐
 │                             │
Clustering                 Autoencoder
 │                             │
DBSCAN / K-Means /         Latent Feature
Agglomerative / GMM        Representation
 │                             │
 ↓                             ↓
Supervised Classification Models
       ↓
Logistic Regression
XGBoost
LightGBM
       ↓
Model Evaluation
       ↓
Lifestyle Cluster Profiles
       ↓
OpenAI API
       ↓
Personalized Recommendations
```
## Data Preprocessing

The preprocessing pipeline included:

- Standardization of categorical values
- Label encoding for categorical variables
- Splitting the `Blood Pressure` feature into:
  - `Systolic_BP`
  - `Diastolic_BP`
- Min-Max normalization
- Stratified train/validation/test splitting
- SMOTE applied to the training data to address class imbalance

## Unsupervised Learning: Clustering

Four clustering algorithms were evaluated:

- K-Means
- DBSCAN
- Agglomerative Clustering
- Gaussian Mixture Model (GMM)

Feature selection was performed using a **K-Nearest Neighbors graph and Laplacian Score**.

The six selected lifestyle features were:

1. Sleep Duration
2. Sleep Quality
3. Stress Level
4. Average Screen Time
5. Smart Device Before Bed
6. Blue-Light Filter Usage

### Clustering Results

DBSCAN and Agglomerative Clustering achieved the strongest clustering performance:

| Metric | DBSCAN / Agglomerative | K-Means / GMM |
|---|---:|---:|
| Silhouette Score | **0.354** | ~0.302 |
| Davies-Bouldin Index | **1.254** | ~1.466 |
| Number of Clusters | 4 | 5 |

DBSCAN cluster assignments were later incorporated as an additional feature for supervised classification.

## Supervised Learning

Three classification algorithms were evaluated:

- Logistic Regression
- XGBoost
- LightGBM

The dataset was split using a stratified:

- **80% Training**
- **10% Validation**
- **10% Testing**

Hyperparameter optimization was performed using **GridSearchCV with 5-fold cross-validation**.

### Final Test Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 65.10% | 68.87% | 84.82% | 76.01% | 56.03% |
| XGBoost | **70.20%** | 69.89% | 70.20% | 65.61% | **60.00%** |
| LightGBM | 70.05% | **70.00%** | **93.40%** | **80.26%** | 59.71% |

### Selected Model: LightGBM

LightGBM was selected as the strongest model in the post-clustering pipeline because it achieved:

- **93.40% Recall**
- **80.26% F1 Score**
- **70.05% Accuracy**

Its high recall allowed the model to identify a large proportion of DED-positive cases.

However, the AUC-ROC remained around **0.60**, indicating that overall class separation was still limited.

## Autoencoder Experiment

An Autoencoder was explored as an alternative unsupervised representation-learning technique.

### Architecture

```text
Input
  ↓
16 neurons
  ↓
12 neurons
  ↓
5-dimensional latent representation
  ↓
Decoder
  ↓
Reconstructed Input
```

The Autoencoder was trained using:

- Mean Squared Error (MSE)
- Up to 300 epochs
- Early stopping
- 8,000 stratified samples

The mean reconstruction error was approximately:

**0.0981**

Although the Autoencoder successfully learned a compressed representation of the data, the latent features did not improve downstream classification performance.

### Full Reconstructed Dataset — LightGBM

| Metric | Result |
|---|---:|
| Accuracy | 58.45% |
| Precision | 68.67% |
| Recall | 66.67% |
| F1 Score | 67.65% |

This experiment showed that good reconstruction performance does not necessarily produce features that are useful for classification.

## Generative AI Recommendations

To extend the project beyond prediction, lifestyle profiles were generated for the clusters identified by DBSCAN.

The average behavioral characteristics of each cluster were summarized and passed to **GPT through the OpenAI API**.

The model generated personalized lifestyle recommendations related to areas such as:

- Sleep habits
- Stress management
- Screen-time reduction
- Smart-device usage
- Blue-light exposure

This demonstrated how **unsupervised learning and Generative AI** can be combined to transform analytical results into more interpretable, user-friendly recommendations.

---

## Key Findings

The project produced several important findings:

- Lifestyle data contains useful predictive information for DED risk.
- LightGBM achieved the highest recall and F1 score in the proposed post-clustering pipeline.
- DBSCAN and Agglomerative Clustering produced stronger cluster separation than K-Means and GMM.
- Adding clustering information did not necessarily improve overall classification performance.
- Autoencoder representations also did not improve classification performance.
- Weak feature relationships and limited class separability were major challenges in the dataset.
- Combining behavioral data with limited clinical information could improve future performance.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Modeling-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Classification-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Autoencoder-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-API-black)

**Core tools and libraries:**

`Python` • `Pandas` • `NumPy` • `Scikit-learn` • `XGBoost` • `LightGBM` • `TensorFlow/Keras` • `imbalanced-learn` • `Matplotlib` • `OpenAI API`

---

## Repository Structure

```text
vision-guard-dry-eye-ml/
│
├── data/
│   └── Dry_Eye_Dataset.csv
│
├── report/
│   └── Ml_DED_Final_Report.pdf
│
├── clustering_and_classification_01.ipynb
├── autoencoder_classification_02.ipynb
│
├── README.md
└── .gitignore
```

## Project Notebooks

### 1. Clustering & Classification

[`clustering_and_classification_01.ipynb`](clustering_and_classification_01.ipynb)

Includes:

- Exploratory Data Analysis
- Data preprocessing
- Feature selection
- K-Means
- DBSCAN
- Agglomerative Clustering
- Gaussian Mixture Models
- Logistic Regression
- XGBoost
- LightGBM
- GPT-powered recommendations

### 2. Autoencoder & Classification

[`autoencoder_classification_02.ipynb`](autoencoder_classification_02.ipynb)

Includes:

- Autoencoder architecture
- Latent feature extraction
- Reconstruction analysis
- Logistic Regression
- XGBoost
- LightGBM
- Evaluation of reconstructed features

---

## Full Project Report

For a detailed description of the methodology, experiments, and results, see the full project report:

📄 [View Full Project Report](report/Ml_DED_Final_Report.pdf)

---

## Future Improvements

Future work could explore:

- Combining lifestyle and selected clinical features
- More advanced representation-learning techniques
- Longitudinal lifestyle data
- Real-time behavioral monitoring
- Mobile health integration
- Improved personalized recommendation systems

---

## Team

Developed as a team machine learning project by:

**Alanoud Almakadi • Dana Alnemari • Nadine Alsahafi • Sarah Sadik • Salma Almutairi**

---

## Disclaimer

This project is intended for **research and educational purposes only**.

The predictions and generated recommendations should not be considered medical diagnoses or substitutes for professional healthcare advice.
