import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset (assuming you have it in a CSV format)
# For simplicity, we'll use a small dataset with just two columns: 'review' and 'sentiment'
# Replace 'your_dataset.csv' with your actual dataset file
df = pd.read_csv('"D:\Project\SentimentAnalysisonMovieReviews\movie_reviews_dataset.csv"')

# Data Preprocessing
# Convert text to lowercase
df['review'] = df['review'].str.lower()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Convert text data into numerical data (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Predictions
y_pred = model.predict(X_test_vectors)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Simple Prediction Example
sample_review = ["This movie was fantastic! I loved the storyline and the characters."]
sample_vector = vectorizer.transform(sample_review)
sample_prediction = model.predict(sample_vector)
print(f"Sample Review Sentiment: {sample_prediction[0]}")
