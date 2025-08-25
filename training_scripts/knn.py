import pickle
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
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

# Grid search for best K in KNN
param_grid = {
    'n_neighbors': [5, 10, 20, 30, 40, 50],
    "weights": ["uniform", "distance"],
    'metric': ['cosine', 'euclidean']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1_weighted', cv=5)
grid.fit(X_train_tfidf, y_train)

print("Best KNN parameters:", grid.best_params_)

# Best KNN model
knn = grid.best_estimator_

# Prediction on test set
y_pred = knn.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
model_file = "models/knn_llm_router_bestk.pkl"
with open(model_file, "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "knn": knn
    }, f)

print(f"Model saved to '{model_file}'")
