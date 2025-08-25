import pickle
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# === Model lists ===
strong_models = [
    "gpt-4-0125-preview", "gpt-4-1106-preview",
    "gpt-4-0314", "gpt-4-0613", "mistral-medium",
    "claude-1", "qwen1.5-72b-chat"
]

weak_models = [
    "claude-2.0", "mixtral-8x7b-instruct-v0.1", "claude-2.1",
    "gemini-pro-dev-api", "gpt-3.5-turbo-0314", "gpt-3.5-turbo-0613",
    "gemini-pro", "gpt-3.5-turbo-0125", "claude-instant-1",
    "yi-34b-chat", "starling-lm-7b-alpha", "wizardlm-70b",
    "vicuna-33b", "tulu-2-dpo-70b", "nous-hermes-2-mixtral-8x7b-dpo",
    "llama-2-70b-chat", "openchat-3.5", "llama2-70b-steerlm-chat",
    "pplx-70b-online", "dolphin-2.2.1-mistral-7b", "gpt-3.5-turbo-1106",
    "deepseek-llm-67b-chat", "openhermes-2.5-mistral-7b",
    "openchat-3.5-0106", "wizardlm-13b", "mistral-7b-instruct-v0.2",
    "solar-10.7b-instruct-v1.0", "zephyr-7b-beta", "zephyr-7b-alpha",
    "codellama-34b-instruct", "mpt-30b-chat", "llama-2-13b-chat",
    "vicuna-13b", "qwen1.5-7b-chat", "pplx-7b-online", "falcon-180b-chat",
    "llama-2-7b-chat", "guanaco-33b", "qwen-14b-chat"
]

# === Load Arena Human Preference dataset ===
print("Loading Arena Human Preference dataset...")
dataset = load_dataset("lmarena-ai/arena-human-preference-55k")
train_dataset = dataset["train"]

# Manual split: 80% train, 20% test
split = train_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
test_data = split["test"]

# === Map models => label (0=weak, 1=strong) ===
def map_label(model):
    if model in strong_models:
        return 1
    elif model in weak_models:
        return 0
    else:
        return None

X_train, y_train = [], []
X_test, y_test = [], []

def process_split(split, X, y):
    for entry in split:
        lbl_a = map_label(entry["model_a"])
        lbl_b = map_label(entry["model_b"])
        if lbl_a is None or lbl_b is None:
            continue
        if entry['winner_model_a'] and lbl_a + lbl_b == 1:
            X.append(entry["prompt"])
            y.append(lbl_a)
        if entry['winner_model_b'] and lbl_a + lbl_b == 1:
            X.append(entry["prompt"])
            y.append(lbl_b)

process_split(train_data, X_train, y_train)
process_split(test_data, X_test, y_test)

print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === k-NN with GridSearch ===
param_grid = {
    'n_neighbors': [5, 10, 20, 30, 40, 50],
    "weights": ["uniform", "distance"],
    "metric": ["cosine", "euclidean", "manhattan"]
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, scoring="f1_weighted", cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train_tfidf, y_train)

print("Best k-NN parameters:", grid.best_params_)
knn = grid.best_estimator_

# === Evaluation ===
y_pred = knn.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# === Save model and vectorizer ===
model_file = "models/knn_arena.pkl"
with open(model_file, "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "knn": knn
    }, f)
