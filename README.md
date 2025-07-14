📊 Project Overview

This project performs an in-depth analysis and modeling exercise on the IMDB Top 250 Movies dataset. Through a series of data processing, visualization, and machine learning tasks, the goal is to understand patterns in movie ratings, genres, and other metadata. Multiple classification and regression models are built to predict ratings and explore cluster behavior among movies.

🎯 Objectives

Handle missing data and clean the dataset
Process and filter genre information
Train and evaluate regression and classification models
Apply KMeans and Hierarchical Clustering for grouping movies
Compare multiple models including:
Linear Regression
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Multinomial Naive Bayes
Use 10-fold cross-validation to assess model performance

📁 Dataset Description

Source: IMDB Top 250 Movies (public dataset)
Key Columns:
movie_id
title
genre
imbd_rating
rank
review_content_count (engineered feature)

🧪 Tasks & Methodology

✅ 1. Missing Data Handling

Checked for missing values across all columns
Printed columns with missing values and counts
Removed all rows with at least one missing value
Calculated dataset size before and after cleaning
Plotted a scatter plot: movie_id vs imbd_rating (8.0 to 9.3)

✅ 2. Genre Processing

Extracted all unique genres in the dataset
Counted and printed genre occurrences
Removed genres with fewer than 10 movies
Final genre list used for downstream modeling

✅ 3. Linear Regression Modeling

Built a Linear Regression model to predict imbd_rating using rank
Split dataset: 70% train, 30% test
Engineered feature: review_content_count
Retrained regression model including this feature
Evaluated with RMSE, MAE, and R² score

✅ 4. K-Means Clustering

Selected numerical features only
Applied K-Means clustering and labeled original data
Calculated cluster-wise means to interpret group characteristics
Visualized clusters using two most distinguishing features
Repeated clustering with higher k values to evaluate separation clarity

✅ 5. Hierarchical Clustering

Performed agglomerative hierarchical clustering
Visualized results using a dendrogram
Analyzed natural groupings of movies without needing to predefine cluster count

✅ 6. K-Nearest Neighbors (KNN) Classifier

Trained a KNN classifier on the labeled movie data
Evaluated model using accuracy score
Tested impact of hyperparameter (k) on classification performance

✅ 7. Gaussian Naive Bayes

Built a Gaussian Naive Bayes classifier
Analyzed model parameters and learned probabilities
Used 10-fold cross-validation to evaluate model robustness

✅ 8. Multinomial Naive Bayes

Created and trained a Multinomial Naive Bayes model
Compared accuracy score with other classifiers

✅ 9. Model Comparison

Compared KNN and Gaussian Naive Bayes classifiers
Reported mean accuracy and standard deviation using 10-fold CV
Identified which model performed better for movie classification

⚙️ Tools & Libraries

Python
pandas, numpy
scikit-learn
seaborn, matplotlib
scipy, sklearn.cluster, sklearn.naive_bayes

📈 Key Outcomes

Outliers and missing data significantly affect model accuracy
rank and review_content_count are strong predictors of imbd_rating
Clustering helped reveal distinct groups of movies with similar characteristics
Gaussian Naive Bayes showed better cross-validated accuracy than KNN in this dataset

👤 Author

Mariam Fatima Burhan
Master of Business Analytics | Macquarie University
