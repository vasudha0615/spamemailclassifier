import joblib
import string

model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def preprocess_text(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return text

def predict_spam(text):
    text = preprocess_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]


sample_text = input("enter the meassage to classify:")
print("prediction:",predict_spam(sample_text))