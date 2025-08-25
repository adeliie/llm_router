import pickle
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

print("Loading dataset...")
# Load dataset
dataset = load_dataset("DevQuasar/llm_router_dataset-synth")
train_data = dataset["train"]
test_data = dataset["test"]

# Extract queries and labels
X_train = [entry["prompt"] for entry in train_data]
y_train = [entry["label"] for entry in train_data]

X_test = [entry["prompt"] for entry in test_data]
y_test = [entry["label"] for entry in test_data]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SVM hyperparameters to test
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search for best SVM
grid = GridSearchCV(SVC(probability=True), param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
grid.fit(X_train_tfidf, y_train)

print("Best SVM parameters:", grid.best_params_)

# Best SVM model
svm_model = grid.best_estimator_

# Prediction on test set
y_pred = svm_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
model_file = "models/svm_llm_router.pkl"
with open(model_file, "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "svm": svm_model
    }, f)

print(f"Model saved to '{model_file}'")
