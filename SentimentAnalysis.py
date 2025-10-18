"""Task
Develop an NLP system for movie review sentiment analysis. The system should collect data from publicly available datasets (e.g., "Large Movie Review Dataset"), preprocess the text, train a supervised machine learning model (e.g., Naive Bayes, SVM, or Deep Learning) on labeled data, evaluate the model on a test set, and predict the sentiment (positive/negative) of new reviews. Evaluate the system's performance on progressively larger datasets. Provide a link to a GitHub or GitLab repository containing the code and a report with performance metrics.
"""
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\MYDEK\Desktop\NLP_PROJECT\\IMDB Dataset.csv')

# Load the "imdb" dataset
#dataset = load_dataset("imdb")

# Load the training split into a variable
#train_dataset = dataset["train"]

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(train_dataset)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
display(df.head())
print("\nDataset Info:")
display(df.info())

"""
Data preprocessing
Clean the text data by removing noise, handling special characters, and converting text to lowercase.
"""
import re
import string

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# Assuming your text column is named 'text'.
df['review_cleaned'] = df['text'].apply(clean_text)

print("Original vs Cleaned Text:")
display(df[['text', 'review_cleaned']].head())

"""
Feature extraction/encoding
Convert the cleaned text data into numerical features that can be used by a machine learning model.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer
# Setting max_features to limit vocabulary size and ngram_range for bigrams
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit and transform the cleaned text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review_cleaned'])

# Display the shape of the TF-IDF matrix
print("Shape of TF-IDF matrix:")
print(tfidf_matrix.shape)

"""
Model selection and training
Choose and train a suitable sentiment analysis model using the TF-IDF features.
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['label'], test_size=0.2, random_state=42)

# Instantiate and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained successfully.")

"""
Assess the performance of the trained sentiment analysis model using appropriate metrics by making predictions on the test set and calculating accuracy, precision, recall, and F1-score.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

"""
Sentiment prediction
Use the trained model to predict the sentiment (positive/negative) of new, unseen movie reviews.
"""
# 1. Define a list of new movie review strings
new_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "The acting was terrible and the plot made no sense. A complete waste of time.",
    "It was an okay movie, not great but not bad either.",
    "Highly recommend this film, a true masterpiece.",
    "I couldn't even finish watching it, so boring."
]

# 2. Apply the clean_text function to each new review
cleaned_new_reviews = [clean_text(review) for review in new_reviews]

# 3. Use the fitted tfidf_vectorizer to transform the cleaned new reviews
new_reviews_tfidf = tfidf_vectorizer.transform(cleaned_new_reviews)

# 4. Use the trained model to predict the sentiment labels
predicted_sentiments = model.predict(new_reviews_tfidf)

# 5. Print the original new reviews and their predicted sentiment labels
sentiment_map = {0: "negative", 1: "positive"}

print("Sentiment predictions for new reviews:")
for i, review in enumerate(new_reviews):
    predicted_label = predicted_sentiments[i]
    predicted_sentiment = sentiment_map[predicted_label]
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {predicted_sentiment}\n")

"""
Scaling and optimization
"""
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Reload the "imdb" dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
df_full = pd.DataFrame(train_dataset)

# 2. Access the 'train' split of the dataset (already done by loading train_dataset)

# 3. Create a list of dataset sizes to evaluate
dataset_sizes = [5000, 10000, 15000, 20000, 25000]

# Dictionary to store performance metrics for each size
performance_metrics = {}

# 4. Iterate through the list of dataset sizes
for size in dataset_sizes:
    print(f"\nEvaluating performance on dataset size: {size}")

    # a. Select a subset of the training data of the current size
    df_subset = df_full.sample(n=size, random_state=42).reset_index(drop=True)

    # b. Apply the clean_text function to the 'text' column of the subset
    df_subset['review_cleaned'] = df_subset['text'].apply(clean_text)

    # c. Split the subset into training and testing sets (e.g., 80% train, 20% test)
    X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(
        df_subset['review_cleaned'], df_subset['label'], test_size=0.2, random_state=42
    )

    # d. Re-instantiate and fit a TfidfVectorizer on the cleaned text of the current training subset
    tfidf_vectorizer_subset = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf_subset = tfidf_vectorizer_subset.fit_transform(X_train_subset)

    # e. Transform both the training and testing subsets using the fitted vectorizer
    X_test_tfidf_subset = tfidf_vectorizer_subset.transform(X_test_subset)

    # f. Re-instantiate and train a LogisticRegression model on the vectorized training data
    model_subset = LogisticRegression()
    model_subset.fit(X_train_tfidf_subset, y_train_subset)

    # g. Make predictions on the vectorized testing data
    y_pred_subset = model_subset.predict(X_test_tfidf_subset)

    # h. Calculate and store the accuracy, precision, recall, and F1-score for the current dataset size
    accuracy_subset = accuracy_score(y_test_subset, y_pred_subset)
    precision_subset = precision_score(y_test_subset, y_pred_subset)
    recall_subset = recall_score(y_test_subset, y_pred_subset)
    f1_subset = f1_score(y_test_subset, y_pred_subset)

    performance_metrics[size] = {
        'accuracy': accuracy_subset,
        'precision': precision_subset,
        'recall': recall_subset,
        'f1_score': f1_subset
    }

    # i. Print the performance metrics for the current dataset size
    print(f"Accuracy: {accuracy_subset:.4f}")
    print(f"Precision: {precision_subset:.4f}")
    print(f"Recall: {recall_subset:.4f}")
    print(f"F1-score: {f1_subset:.4f}")

# Optional: Print all collected performance metrics
print("\nPerformance metrics across different dataset sizes:")
display(pd.DataFrame(performance_metrics).T)



# performance_metrics = {
#     5000: {'accuracy': 0.86, 'precision': 0.8426, 'recall': 0.8740, 'f1_score': 0.8580},
#     10000: {'accuracy': 0.8655, 'precision': 0.8675, 'recall': 0.8717, 'f1_score': 0.8696},
#     15000: {'accuracy': 0.8813, 'precision': 0.8795, 'recall': 0.8865, 'f1_score': 0.8830},
#     20000: {'accuracy': 0.8805, 'precision': 0.8799, 'recall': 0.8842, 'f1_score': 0.8820},
#     25000: {'accuracy': 0.8862, 'precision': 0.8759, 'recall': 0.8981, 'f1_score': 0.8869}
# }


### Performance Metrics Visualization

# Create a plot for performance metrics
sizes = list(performance_metrics.keys())
accuracy_scores = [metrics['accuracy'] for metrics in performance_metrics.values()]
precision_scores = [metrics['precision'] for metrics in performance_metrics.values()]
recall_scores = [metrics['recall'] for metrics in performance_metrics.values()]
f1_scores = [metrics['f1_score'] for metrics in performance_metrics.values()]

plt.figure(figsize=(10, 6))
plt.plot(sizes, accuracy_scores, marker='o', label='Accuracy')
plt.plot(sizes, precision_scores, marker='o', label='Precision')
plt.plot(sizes, recall_scores, marker='o', label='Recall')
plt.plot(sizes, f1_scores, marker='o', label='F1-score')
plt.title('Performance Metrics vs. Dataset Size')
plt.xlabel('Dataset Size')
plt.ylabel('Score')
plt.xticks(sizes)
plt.grid(True)
plt.legend()
plt.tight_layout()

