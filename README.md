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
