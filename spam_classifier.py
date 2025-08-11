import pandas as pd
import string

def preprocess_text(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return text

df = pd.read_csv("spam.csv",encoding = "latin-1")[["v1","v2"]]
df.columns = ["label","text"]

df['text'] = df['text'].apply(preprocess_text)

print(df.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size = 0.2,random_state = 42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

from sklearn.naive_bayes import MultinomialNB

# Create the model
model = MultinomialNB()

# Train the model on training data
model.fit(x_train_vec, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = model.predict(x_test_vec)

# Calculate and print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print detailed classification report (precision, recall, F1-score)
print(classification_report(y_test, y_pred))

def predict_spam(text):

    text = preprocess_text(text)
    text_vec= vectorizer.transform(['text'])
    prediction = model.predict(text_vec)
    return prediction[0]


sample_text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
print("Prediction:", predict_spam(sample_text))

import joblib

joblib.dump(model,"spam_classifier_model.joblib")
joblib.dump(vectorizer,"tfidf_vectorizer.joblib")
print("model and vectorizer saved")