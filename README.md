# online-shoppers-purchasing-intention
Predicting real-time purchase intent in e-commerce environments is a critical challenge for dynamic personaliza-
tion and customer retention. This study proposes an end-to-end machine learning pipeline applied to the Online
Shoppers Purchasing Intention dataset, comprising 12,330 web sessions. To overcome the severe class imbalance
and capture complex behavioral patterns, we design a hybrid framework integrating unsupervised representation
learning and robust supervised classification. Our methodology initiates with an exhaustive exploratory data
analysis (EDA), including Google Analytics metrics and temporal dynamics. We then apply Principal Com-
ponent Analysis (PCA) for linear dimensionality reduction and t-Distributed Stochastic Neighbor Embedding
(t-SNE) to map non-linear manifolds. Crucially, we leverage K-Means clustering on the latent space to engineer
a novel categorical feature, Cluster ID, representing distinct user navigation profiles. To prevent data leakage,
a strict stratified train-test split (80/20) is enforced prior to applying Information Theory filters (Mutual Infor-
mation) and Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTENC)
on the training set alone. Finally, we benchmark four distinct learning paradigms: Random Forest, XGBoost,
Support Vector Machines (SVM) with radial basis function kernels, and a deep Multi-Layer Perceptron (MLP).
In classical ML algorithms, hyperparameters are systematically optimized using Grid Search coupled with a
Stratified 5-fold Cross-Validation; the only DL algorithm is improved using stronger regularitazion and Early
Stopping. The models are evaluated under a unified framework prioritizing Precision-Recall curves, F1-Score,
and Area Under the ROC (AUC-ROC), providing insights into the trade-off between false alarm rates and missed
conversion opportunities.
