# IMDB Sentiment Analysis using Machine Learning

## Overview
This project performs sentiment analysis on movie reviews from the IMDB dataset.
The goal is to classify reviews as positive or negative using machine learning models.

## Technologies Used
* Python
* Scikit-learn
* XGBoost
* TensorFlow Hub (Universal Sentence Encoder)
* Pandas, NumPy
* Matplotlib, Seaborn
* NLTK

## Workflow
1. Data loading and preprocessing
2. Text cleaning (tokenization, stopword removal)
3. Feature extraction:
 * TF-IDF Vectorization
 * Uniersal Sentence Encoder (USE)
4. Model training:
 * Logistic Regression
 * Support Vector Machine (SVM)
 * Naive Bayes
 * XGBoost
 * Random Forest
5. Model evaluation using:
 * Accuracy
 * Precision, Recall, F1-score

## Results
* Best Accuracy: ~87%
* Logistic Regression and SVM performed the best
* Tree-based models (Random Forest) performed worse on sparse TF-IDF features

## Key Insights
* TF-IDF is highly effective for text classification tasks
* XGBoost performs well but requires more computation
* Universal Sentence Encoder provides richer embeddings but increases runtime
* Simpler models can outperform complex ones on structured text features

## Dataset was too big and cannot be uploaded to this repository, request one via email (mananb2708@gmail.com)

Manan Bhatia (Ti2ni1m)
