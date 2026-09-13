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

