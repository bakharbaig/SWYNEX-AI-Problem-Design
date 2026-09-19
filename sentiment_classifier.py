"""
SWYNEX - AI Problem Design
Student Feedback Sentiment Classifier

Run:
    python sentiment_classifier.py

This program trains a simple machine-learning model and predicts
whether student feedback is Positive, Negative, or Neutral.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = pd.read_csv("dataset.csv")

X = data["feedback"]
y = data["sentiment"]

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 3. Create the AI/ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("=" * 55)
print("SWYNEX - Student Feedback Sentiment Classifier")
print("=" * 55)
print(f"Test Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 6. Test new feedback
new_feedback = [
    "The teacher explained everything very clearly.",
    "The classes were confusing and difficult.",
    "The assignment was submitted today."
]

print("\nSample Predictions:")
for text in new_feedback:
    result = model.predict([text])[0]
    print(f'Feedback: "{text}"')
    print(f"Prediction: {result}\n")

# 7. Interactive prediction
while True:
    user_text = input("Enter feedback (or type 'exit' to stop): ").strip()

    if user_text.lower() == "exit":
        print("Program ended.")
        break

    if not user_text:
        print("Please enter some feedback.\n")
        continue

    result = model.predict([user_text])[0]
    print(f"Predicted Sentiment: {result}\n")
