# Online Shoppers Purchasing Intention

Predicting real-time purchase intent in e-commerce environments is a critical challenge for dynamic personalization and customer retention. This repository contains an end-to-end machine learning pipeline applied to the **Online Shoppers Purchasing Intention dataset**, comprising 12,330 web sessions. 

To overcome severe class imbalance and capture complex behavioral patterns, we designed a hybrid framework that integrates unsupervised representation learning with robust supervised classification.

## Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA)
We initiate the project with an exhaustive EDA, exploring Google Analytics metrics and temporal dynamics to deeply understand user behavior.

### 2. Dimensionality Reduction & Feature Engineering
- **Manifold Learning:** We apply Principal Component Analysis (PCA) for linear dimensionality reduction and t-Distributed Stochastic Neighbor Embedding (t-SNE) to map non-linear manifolds.
- **Clustering:** We leverage K-Means clustering on the latent space to engineer a novel categorical feature, `Cluster ID`, representing distinct user navigation profiles.

### 3. Data Preprocessing & Imbalance Handling
To strictly prevent data leakage, a stratified train-test split (80/20) is enforced *prior* to data augmentation. On the training set alone, we apply:
- **Feature Selection:** Information Theory filters (Mutual Information).
- **Oversampling:** SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features).

## Models & Optimization

We benchmark four distinct learning paradigms:
- **Random Forest**
- **XGBoost**
- **Support Vector Machines (SVM)** (with RBF kernels)
- **Deep Multi-Layer Perceptron (MLP)**

**Hyperparameter Tuning:** 
Classical ML algorithms are systematically optimized using Grid Search coupled with a Stratified 5-fold Cross-Validation. The Deep Learning approach (MLP) is improved using stronger regularization and Early Stopping.

## Evaluation

All models are evaluated under a unified framework providing insights into the trade-off between false alarm rates and missed conversion opportunities. Our primary evaluation metrics prioritize:
- Precision-Recall Curves
- F1-Score
- Area Under the ROC (AUC-ROC)
