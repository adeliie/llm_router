from huggingface_hub import login
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import f1_score
import os

# --- Login Hugging Face via token depuis variable d'environnement ---
# Assure-toi de définir HF_TOKEN dans ton terminal : export HF_TOKEN="ton_token"
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

# === Charger dataset ===
dataset_id = "lmarena-ai/arena-human-preference-55k"
raw_dataset = load_dataset(dataset_id, split="train")

print(f"Taille brute: {len(raw_dataset)}")

# === Définition des modèles Strong/Weak ===
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

# === Mapping Strong vs Weak ===
def map_label(example):
    if example["winner_model_a"] == 1:
        if example["model_a"] in strong_models:
            example["labels"] = 1
        elif example["model_a"] in weak_models:
            example["labels"] = 0
        else:
            example["labels"] = None
    elif example["winner_model_b"] == 1:
        if example["model_b"] in strong_models:
            example["labels"] = 1
        elif example["model_b"] in weak_models:
            example["labels"] = 0
        else:
            example["labels"] = None
    else:
        example["labels"] = None  # tie ou inconnu
    return {"prompt": example["prompt"], "labels": example["labels"]}

# Appliquer mapping
dataset = raw_dataset.map(map_label)

# Filtrer les exemples sans label
dataset = dataset.filter(lambda x: x["labels"] is not None)

print(f"Dataset après mapping: {len(dataset)}")

# === Split train/test ===
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, test_ds = train_test["train"], train_test["test"]

print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
print(train_ds[0])

# --- Tokenisation ---
model_id = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512

def tokenize(batch):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True)

cols_to_remove = ["id", "model_a", "model_b", "response_a", "response_b",
                  "winner_model_a", "winner_model_b", "winner_tie", "prompt"]

train_ds = train_ds.map(tokenize, batched=True, remove_columns=cols_to_remove)
test_ds = test_ds.map(tokenize, batched=True, remove_columns=cols_to_remove)

# --- Charger le modèle ---
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# --- Métrique F1 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds, average="weighted")}

# --- Arguments d'entraînement ---
training_args = TrainingArguments(
    output_dir="router_arena",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("models/router_arena")
tokenizer.save_pretrained("models/router_arena")
