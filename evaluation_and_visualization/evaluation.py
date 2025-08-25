import matplotlib.pyplot as plt
from datasets import load_dataset
import ollama
import time
import numpy as np
import pickle
import json
import random
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ========================================
# DATASET CONFIGURATION
# ========================================

def get_dataset_config():
    """
    Configuration of supported datasets.
    Returns a dictionary with the configuration of the chosen dataset.
    """
    datasets_config = {
        "alpaca": {
            "name": "tatsu-lab/alpaca",
            "subset": "default",
            "split": "train",
            "question_field": "instruction",
            "answer_field": "output",
            "description": "Alpaca dataset with instructions and responses"
        },
        "gsm8k": {
            "name": "gsm8k",
            "subset": "main",
            "split": "test",
            "question_field": "question",
            "answer_field": "answer",
            "description": "Grade School Math 8K problems"
        },
        "mmlu": {
            "name": "cais/mmlu",
            "subset": "all",
            "split": "validation",
            "question_field": "question",
            "answer_field": "answer",
            "description": "Multi-domain multiple-choice benchmark"
        },
        "squad_v2": {
            "name": "squad_v2",
            "subset": None,
            "split": "validation",
            "question_field": "question",
            "answer_field": "answers",
            "answer_processor": lambda ans: ans["text"][0] if isinstance(ans, dict) and "text" in ans and ans["text"] else "",
            "description": "Stanford Question Answering Dataset v2"
        }
    }
    
    # CHOOSE THE DATASET HERE
    selected_dataset = "gsm8k"  # Change this value to use another dataset
    
    if selected_dataset not in datasets_config:
        raise ValueError(f"Dataset '{selected_dataset}' not supported. Available datasets: {list(datasets_config.keys())}")
    
    return datasets_config[selected_dataset], selected_dataset

def load_and_process_dataset(config, num_samples=100):
    """
    Load and process the dataset according to its configuration.
    """
    print(f"Loading dataset: {config['description']}")
    
    # Load dataset
    if config["subset"]:
        ds = load_dataset(config["name"], config["subset"])[config["split"]]
    else:
        ds = load_dataset(config["name"])[config["split"]]
    
    # Select a sample
    if len(ds) > num_samples:
        ds = ds.select(range(num_samples))
    
    print(f"Number of samples loaded: {len(ds)}")
    return ds

def extract_qa_pair(sample, config):
    """
    Extract the question-answer pair from a sample according to the dataset configuration.
    """
    try:
        # Extract question
        question = sample.get(config["question_field"], "")
        if not question or not question.strip():
            return None, None
        
        # Extract answer
        answer_raw = sample.get(config["answer_field"], "")
        
        # Apply answer processor if defined
        if "answer_processor" in config and config["answer_processor"]:
            answer = config["answer_processor"](answer_raw)
        else:
            answer = answer_raw
        
        # Ensure answer is not empty
        if not answer or (isinstance(answer, str) and not answer.strip()):
            return None, None
            
        return str(question).strip(), str(answer).strip()
    
    except Exception as e:
        print(f"Error during QA extraction: {e}")
        return None, None

# ========================================
# DATASET LOADING AND CONFIGURATION
# ========================================

dataset_config, dataset_name = get_dataset_config()
print(f"Selected dataset: {dataset_name}")
print(f"Description: {dataset_config['description']}")

ds = load_and_process_dataset(dataset_config, num_samples=30)

# ========================================
# LOADING PRETRAINED MODELS
# ========================================

# Load KNN
try:
    with open("models/knn_llm_router_bestk.pkl", "rb") as f:
        knn_data = pickle.load(f)
    knn_vectorizer = knn_data["vectorizer"]
    knn_model = knn_data["knn"]
    print("KNN router loaded successfully")
except FileNotFoundError:
    print("KNN file not found, this router will be ignored")
    knn_model = None

# Load SVM
try:
    with open("models/svm_llm_router.pkl", "rb") as f:
        svm_data = pickle.load(f)
    svm_vectorizer = svm_data["vectorizer"]
    svm_model = svm_data["svm"]
    print("SVM router loaded successfully")
except FileNotFoundError:
    print("SVM file not found, this router will be ignored")
    svm_model = None

try:
    with open("models/knn_arena.pkl", "rb") as f:
        knn_arena_data = pickle.load(f)
    knn_vectorizer_arena = knn_arena_data["vectorizer"]
    knn_model_arena = knn_arena_data["knn"]
    print("KNN Arena router loaded successfully")
except FileNotFoundError:
    print("KNN Arena file not found, this router will be ignored")
    knn_model_arena = None

try:
    model_dir = "models/router_arena"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_bert = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model_bert.eval()
except Exception as e:
    print(f"Error loading BERT router model: {e}")
    model = None

# ========================================
# LLM FUNCTIONS
# ========================================

def query_small_llm(prompt):
    """Query the small LLM"""
    try:
        resp = ollama.generate(model="tinyllama", prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        print(f"Error with TinyLlama: {e}")
        return "Generation error"

def query_big_llm(prompt):
    """Query the big LLM"""
    try:
        resp = ollama.generate(model="llama3", prompt=prompt)
        return resp['response'].strip()
    except Exception as e:
        print(f"Error with Llama3: {e}")
        return "Generation error"

def score_response(question, answer, ground_truth, dataset_name="general"):
    """
    Score a response against ground truth.
    Adapts evaluation prompt depending on dataset type.
    """
    eval_prompts = {
        "gsm8k": f"""You have to say how accurate is a mathematical answer.
The question is : "{question}"
The Student respond: "{answer}"
the correct answer is : "{ground_truth}"
You have to give a score from 0 to 10 based on mathematical correctness. 
Just answer with a single number which is the mark of the student out of 10 and nothing else""",
        
        "mmlu": f"""You are evaluating a multiple choice answer.
Question: "{question}"
Answer: "{answer}"
Correct answer: "{ground_truth}"
Score 10 if the anwer is the same as the correct answer, 0 if wrong. Just answer with a single number.""",
        
        "general": f"""You are a strict judge evaluating the quality of an answer.
Question: "{question}"
Proposed answer: "{answer}"
Reference answer: "{ground_truth}"
Give a score from 0 to 10. Just answer with a single number."""
    }
    
    eval_prompt = eval_prompts.get(dataset_name, eval_prompts["general"])
    
    try:
        resp = ollama.generate(model="llama3", prompt=eval_prompt)
        import re
        numbers = re.findall(r'\d+\.?\d*', resp['response'])
        print(numbers)
        return float(numbers[0]) if numbers else 0.0
    except Exception as e:
        print(f"Error during scoring: {e}")
        return 0.0

# ========================================
# PROCESSING FUNCTIONS
# ========================================

def get_win_probs(model, vectorizer, dataset, config):
    """Compute win probabilities for a routing model"""
    if model is None or vectorizer is None:
        return [], [], []
    
    win_probs = []
    questions = []
    ground_truths = []
    
    for sample in dataset:
        try:
            q, gt = extract_qa_pair(sample, config)
            
            if q is None or gt is None:
                print(f"Sample ignored: empty question or answer")
                continue
                
            Xq = vectorizer.transform([q])
            probs = model.predict_proba(Xq)[0]
            
            if 1 in model.classes_:
                big_index = list(model.classes_).index(1)
                win_probs.append(probs[big_index])
                questions.append(q)
                ground_truths.append(gt)
            else:
                print(f"Class 1 not found in model classes: {model.classes_}")
                
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"Probabilities computed for {len(win_probs)} samples")
    return win_probs, questions, ground_truths

def get_bert_win_probs(model, tokenizer, dataset, config):
    """Compute win probabilities for the BERT routing model"""
    if model is None or tokenizer is None:
        return [], [], []
    
    win_probs = []
    questions = []
    ground_truths = []
    
    for sample in dataset:
        try:
            q, gt = extract_qa_pair(sample, config)
            
            if q is None or gt is None:
                print(f"Sample ignored: empty question or answer")
                continue
                
            inputs = tokenizer(q, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).squeeze().tolist()
            
            win_probs.append(probs[1])
            questions.append(q)
            ground_truths.append(gt)
                
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"Probabilities computed for {len(win_probs)} samples")
    return win_probs, questions, ground_truths

def router_scores(win_probs, questions, ground_truths, score_cache, dataset_name, label=""):
    """Compute average scores for different thresholds tau"""
    if not questions:
        print(f"No question available for router {label}, skipping...")
        return []

    thresholds = np.unique(
        np.concatenate([
            np.linspace(0, 0.2, 40),
            np.linspace(0.2, 0.6, 20),
            np.linspace(0.8, 1, 40)
        ])
    )

    results = []
    for tau in thresholds:
        all_scores = []
        big_calls = 0
        
        for q, gt, wp in zip(questions, ground_truths, win_probs):
            if wp > tau:
                score = 10.0
                big_calls += 1
            else:
                score = score_cache.get(q, 0.0)
            all_scores.append(score)
            
        results.append({
            'router': label,
            'tau': tau,
            'pct_big': (big_calls / len(questions)) * 100 if len(questions) > 0 else 0,
            'avg_score': np.mean(all_scores) if all_scores else 0
        })
    
    return results

# ========================================
# MAIN PROCESSING
# ========================================

cache_suffix = f"_{dataset_name}"
small_cache_file = f"caching_results/small_llm_cache{cache_suffix}.pkl"
score_cache_file = f"caching_results/score_cache{cache_suffix}.pkl"

results_file = f"caching_results/router_results{cache_suffix}.json"

import os

if os.path.exists(small_cache_file) and os.path.exists(score_cache_file):
    print("Loading existing caches...")
    with open(small_cache_file, "rb") as f:
        small_llm_cache = pickle.load(f)
    with open(score_cache_file, "rb") as f:
        score_cache = pickle.load(f)
        print(score_cache)
    print(f"Cache loaded: {len(small_llm_cache)} responses, {len(score_cache)} scores")
else:
    print("Initial computation of responses and scores...")
    small_llm_cache = {}
    score_cache = {}
    
    for i, sample in enumerate(ds):
        try:
            print(f"Pre-computation: {i+1}/{len(ds)}")
            q, gt = extract_qa_pair(sample, dataset_config)
            
            if q is None or gt is None:
                print(f"Sample {i} ignored: empty question or answer")
                continue
                
            ans = query_small_llm(q)
            small_llm_cache[q] = ans
            score_cache[q] = score_response(q, ans, gt, dataset_name)
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error during pre-computation of sample {i}: {e}")
            continue
    
    with open(small_cache_file, "wb") as f:
        pickle.dump(small_llm_cache, f)
    with open(score_cache_file, "wb") as f:
        pickle.dump(score_cache, f)
    print("New caches saved")

print("Computing router performances...")

all_results = {}

routers = [
    ("KNN", knn_model, knn_vectorizer),
    ("SVM", svm_model, svm_vectorizer),
    ("KNN Arena", knn_model_arena, knn_vectorizer_arena),
    ("BERT Router", model_bert, tokenizer)
]

for router_name, model, vectorizer in routers:
    if model is not None and vectorizer is not None:
        print(f"Processing router: {router_name}")
        if router_name == "BERT Router":
            probs, questions, ground_truths = get_bert_win_probs(model, vectorizer, ds, dataset_config)
        else:
            probs, questions, ground_truths = get_win_probs(model, vectorizer, ds, dataset_config)
        
        if probs:
            results = router_scores(probs, questions, ground_truths, score_cache, dataset_name, router_name)
            all_results[router_name] = results
        else:
            print(f"No results for router {router_name}")
    else:
        print(f"Router {router_name} not available")

if all_results:
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved in '{results_file}'")

    print(f"\nEvaluation finished on dataset {dataset_name}")
    print(f"Dataset: {dataset_config['description']}")
    print(f"Number of processed samples: {len(score_cache)}")
    
else:
    print("No router available for evaluation")
