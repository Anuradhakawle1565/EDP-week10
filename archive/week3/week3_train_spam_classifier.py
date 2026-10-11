import os, re, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = os.path.join("data", "spam.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("=== WEEK 3: SPAM CLASSIFIER ===")
data = pd.read_csv(DATA_PATH).dropna(subset=["label", "message"])
data["label"] = data["label"].str.lower().str.strip()
data["clean_message"] = data["message"].astype(str).apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    data["clean_message"], data["label"], test_size=0.20,
    random_state=42, stratify=data["label"]
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("\n--- Evaluation ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, pos_label='spam', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, pos_label='spam', zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred, pos_label='spam', zero_division=0):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
ConfusionMatrixDisplay(cm, display_labels=["HAM", "SPAM"]).plot()
plt.title("Spam Classifier - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

joblib.dump(model, os.path.join(MODEL_DIR, "spam_classifier.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("\nModel and TF-IDF vectorizer saved in model/")
print("\n--- Sample Predictions ---")
for msg in [
    "Congratulations you have won a free prize!",
    "Can you send me the project notes?",
    "Claim your free cash reward now!",
    "Are we meeting after class?"
]:
    pred = model.predict(vectorizer.transform([preprocess_text(msg)]))[0]
    print(f"{pred.upper():4} -> {msg}")

print("\nWeek 3 project completed!")
