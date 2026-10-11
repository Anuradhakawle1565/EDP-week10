import os, re, joblib

model = joblib.load(os.path.join("model", "spam_classifier.pkl"))
vectorizer = joblib.load(os.path.join("model", "tfidf_vectorizer.pkl"))

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("=== SPAM MESSAGE PREDICTOR ===")
print("Type a message. Type 'exit' to stop.\n")

while True:
    message = input("Enter message: ")
    if message.lower().strip() == "exit":
        break
    prediction = model.predict(vectorizer.transform([clean(message)]))[0]
    confidence = model.predict_proba(vectorizer.transform([clean(message)]))[0].max()
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.2%}\n")
