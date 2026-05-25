import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('reviews.csv')

# Clean text function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    return text

# Apply cleaning
df['cleaned_review'] = df['review'].apply(clean_text)

# Features and labels
X = df['cleaned_review']
y = df['sentiment']

# Convert text to numbers
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# User input
user_review = input("Enter a review: ")

# Clean input
cleaned = clean_text(user_review)

# Convert to vector
vectorized = vectorizer.transform([cleaned])

# Predict
prediction = model.predict(vectorized)

print("Predicted Sentiment:", prediction[0])