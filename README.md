# DLBAIPNLP01-Project-NLP-Sentiment-Analysis-on-Movie-Reviews

# Movie Review Sentiment Analysis Project Report

## 1. Project Overview
The goal of this project was to develop an NLP system to analyze the sentiment of movie reviews, classifying them as either positive or negative. The system involves data collection, preprocessing, feature extraction, model training, evaluation, and prediction.

## 2. Dataset
The project utilized the **Large Movie Review Dataset (often referred to as the IMDb dataset)**, a publicly available dataset commonly used for sentiment analysis tasks. The dataset contains 50,000 movie reviews, split equally into 25,000 for training and 25,000 for testing. Each review is labeled as either positive (1) or negative (0). For the scaling evaluation, we used subsets of the training data of varying sizes (5000, 10000, 15000, 20000, and 25000).

## 3. Methodology

### 3.1 Data Preprocessing
The raw text data underwent several preprocessing steps to prepare it for model training:
- **Lowercase Conversion:** All text was converted to lowercase to ensure consistency and reduce the vocabulary size.
- **Punctuation Removal:** Punctuation marks were removed as they generally do not contribute to the sentiment of a review.
- **Number Removal:** Numerical digits were removed from the text.
- **Whitespace Handling:** Leading/trailing whitespace and multiple spaces within the text were standardized.

### 3.2 Feature Extraction
To convert the cleaned text into a numerical format suitable for machine learning models, the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique was employed. TF-IDF reflects the importance of a word in a document relative to a collection of documents. We used a `TfidfVectorizer` with `max_features=5000` to limit the vocabulary size to the most frequent 5000 terms and `ngram_range=(1, 2)` to include both unigrams and bigrams.

### 3.3 Model Selection and Training
A **Logistic Regression** model was chosen for sentiment analysis. Logistic Regression is a simple yet effective linear model for binary classification tasks. The model was trained on the TF-IDF features extracted from the preprocessed training data subsets.

### 3.4 Model Evaluation
The trained model was evaluated on a separate test set (20% of each dataset subset) using the following performance metrics:
- **Accuracy:** The proportion of correctly classified reviews.
- **Precision:** The ability of the model to correctly identify positive reviews out of all reviews predicted as positive.
- **Recall:** The ability of the model to find all the positive reviews.
- **F1-score:** The harmonic mean of precision and recall, providing a single metric that balances both.

## 4. Results

The performance of the Logistic Regression model was evaluated on different training dataset sizes. The results are summarized in the table below:

| Dataset Size | Accuracy | Precision | Recall | F1-score |
|--------------|----------|-----------|--------|----------|
| 5000 | 0.8600 | 0.8426 | 0.8740 | 0.8580 |
| 10000 | 0.8655 | 0.8675 | 0.8717 | 0.8696 |
| 15000 | 0.8813 | 0.8795 | 0.8865 | 0.8830 |
| 20000 | 0.8805 | 0.8799 | 0.8842 | 0.8820 |
| 25000 | 0.8862 | 0.8759 | 0.8981 | 0.8869 |


### Performance Metrics Visualization
(https://github.com/denizlahi5-cpu/DLBAIPNLP01-Project-NLP-Sentiment-Analysis-on-Movie-Reviews/blob/main/performance_metrics_plot.png)

### Discussion of Results

The evaluation results show that the model's performance generally improves as the training dataset size increases. This is expected as supervised learning models benefit from more training data, allowing them to learn more robust patterns. The F1-score, which balances precision and recall, shows a clear upward trend with increasing dataset size, indicating better overall performance. The accuracy also follows a similar pattern. The precision and recall scores are relatively balanced, suggesting that the model is not heavily biased towards classifying reviews as positive or negative.

## 5. Potential Future Improvements

Several areas can be explored for future improvements:
- **Exploring Different Models:** Evaluate other machine learning models such as Naive Bayes, Support Vector Machines (SVM), or deep learning models (e.g., LSTMs, GRUs, or Transformer-based models like BERT) to potentially achieve higher accuracy.
- **Hyperparameter Tuning:** Optimize the hyperparameters of the chosen model (Logistic Regression) and the TF-IDF vectorizer using techniques like cross-validation and grid search.
- **Advanced Text Preprocessing:** Incorporate more advanced preprocessing steps such as stemming, lemmatization, and stop word removal.
- **Alternative Feature Extraction:** Experiment with other feature extraction techniques like Word Embeddings (Word2Vec, GloVe) or contextual embeddings (BERT embeddings) which can capture semantic relationships between words.
- **Handling Imbalanced Data:** If dealing with datasets where the sentiment distribution is skewed, techniques for handling class imbalance could be implemented.

## 6. Repository Link

The code for this project is available on GitHub at: https://github.com/denizlahi5-cpu/DLBAIPNLP01-Project-NLP-Sentiment-Analysis-on-Movie-Reviews
