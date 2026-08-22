"""
BankIQ - NLP-Powered Banking Query Resolution Engine
An NLP + Machine Learning based banking query resolution system.

Pipeline: clean text -> TF-IDF vectorize -> Naive Bayes intent
classification -> confidence check -> risk-aware response lookup -> Gradio UI.

Run with: python banking_chatbot.py
"""

import os
import random
import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "banking_support.csv"
MODEL_PATH = "banking_chatbot_model.pkl"
VECTORIZER_PATH = "banking_chatbot_vectorizer.pkl"

CONFIDENCE_THRESHOLD = 0.06
TOLL_FREE_NUMBER = "1800-419-1234"

# Categories that are conversational rather than banking-informational.
# These are matched with simple rule-based patterns (see detect_small_talk)
# rather than the ML classifier - they're a small, fixed set of phrasings,
# and stopword removal tends to strip them down to nothing anyway (e.g.
# "who are you" loses every single word to the stopword list).
CONVERSATIONAL_CATEGORIES = {"identity", "greeting", "goodbye", "thanks", "general_help"}

SMALL_TALK_TRIGGERS = {
    "greeting": ["hi", "hii", "hello", "hey", "yo", "good morning", "good afternoon", "good evening", "whats up", "what's up"],
    "goodbye": ["bye", "goodbye", "see you", "take care", "good night", "gtg"],
    "thanks": ["thanks", "thank you", "thank u", "tq", "ty", "appreciate it"],
    "identity": ["who are you", "what are you", "what is your name", "whats your name", "tell me about yourself", "are you a bot", "are you human", "are you ai"],
    "general_help": ["what can you do", "how can you help", "what services", "what do you do", "help me with", "how does this work"],
}

# A question shares no vocabulary at all with anything the model was
# trained on - it's very unlikely to be a banking question.
OUT_OF_SCOPE_MESSAGES = [
    "That doesn't look like a banking-related question, so I can't help with it here. You might want to try a general web search instead.",
    "This falls outside banking topics I'm trained on. I can't answer that one — a general search engine would be a better fit.",
]

# A question uses some recognizable words, but the model isn't confident
# enough in any category to answer, and the topic doesn't look urgent.
LOW_CONFIDENCE_MESSAGES = [
    "I'm not fully sure what you're asking. Could you rephrase that, maybe with a bit more detail?",
    "I couldn't confidently match that to a banking topic I know. Try asking it a different way?",
]

# A question uses recognizable words and its best-guess category is
# high-priority (fraud, lost card, KYC, etc.), but confidence is still too
# low to safely act on. This is the one case where we point to a human.
UNRESOLVED_URGENT_MESSAGE = (
    "This sounds like it could be an urgent banking matter, and I'm not confident enough "
    f"in my answer to guide you safely. Please call our 24x7 helpline at {TOLL_FREE_NUMBER} "
    "so a bank representative can help directly."
)

ESCALATION_NOTICE = (
    f"\n\n⚠️ This is a high-priority banking matter. For immediate help, "
    f"please call our 24x7 helpline at {TOLL_FREE_NUMBER}."
)

BOT_NAME = "BankIQ"

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)


def detect_small_talk(text):
    """
    Checks the raw (lightly normalized) question against a small set of
    fixed conversational patterns, before any stopword removal happens.
    Returns the matching category, or None if this looks like a real
    banking question that should go through the ML pipeline instead.
    """
    normalized = re.sub(r"[^a-z\s']", " ", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    for category, triggers in SMALL_TALK_TRIGGERS.items():
        for trigger in triggers:
            if re.search(rf"\b{re.escape(trigger)}\b", normalized):
                return category
    return None


# ---------------------------------------------------------------------------
# Load data and build lookups
# ---------------------------------------------------------------------------

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    df["clean_question"] = df["question"].apply(clean_text)
    return df


def build_priority_lookup(df):
    return df.drop_duplicates("category").set_index("category")["priority"].to_dict()


def build_vectorizer():
    return TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=1)


# ---------------------------------------------------------------------------
# Train, evaluate, and persist the model
# ---------------------------------------------------------------------------

def train_and_save_model(df):
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["clean_question"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    eval_model = MultinomialNB()
    eval_model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, eval_model.predict(X_test))
    print(f"Held-out test accuracy: {test_accuracy * 100:.1f}% "
          f"({len(y_test)} unseen questions, {y.nunique()} categories)")

    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    return model, vectorizer


def load_or_train_model(df):
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("Loading saved model and vectorizer...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    print("No saved model found. Training a new one...")
    return train_and_save_model(df)


# ---------------------------------------------------------------------------
# Predict intent and generate a response
# ---------------------------------------------------------------------------

def get_response(question, model, vectorizer, df, priority_lookup, threshold=CONFIDENCE_THRESHOLD):
    small_talk_category = detect_small_talk(question)
    if small_talk_category is not None:
        options = df.loc[df["category"] == small_talk_category, "response"].unique().tolist()
        response = random.choice(options) if len(options) > 1 else options[0]
        return response, small_talk_category, 1.0

    cleaned = clean_text(question)
    vector = vectorizer.transform([cleaned])

    if vector.sum() == 0:
        return random.choice(OUT_OF_SCOPE_MESSAGES), None, 0.0

    probabilities = model.predict_proba(vector)[0]
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]
    predicted_category = model.classes_[best_index]

    if confidence < threshold:
        if priority_lookup.get(predicted_category) == "high":
            return UNRESOLVED_URGENT_MESSAGE, predicted_category, confidence
        return random.choice(LOW_CONFIDENCE_MESSAGES), predicted_category, confidence

    matches = df.loc[df["category"] == predicted_category, "response"]

    if predicted_category in CONVERSATIONAL_CATEGORIES:
        options = matches.unique().tolist()
        response = random.choice(options) if len(options) > 1 else options[0]
        return response, predicted_category, confidence

    response = matches.iloc[0]
    if priority_lookup.get(predicted_category) == "high":
        response = response + ESCALATION_NOTICE

    return response, predicted_category, confidence


# ---------------------------------------------------------------------------
# Gradio chat interface
# ---------------------------------------------------------------------------

def build_chat_function(model, vectorizer, df, priority_lookup):
    def chat_fn(message, history):
        response, _category, _confidence = get_response(
            message, model, vectorizer, df, priority_lookup
        )
        return response
    return chat_fn


def launch_app(model, vectorizer, df, priority_lookup):
    chat_fn = build_chat_function(model, vectorizer, df, priority_lookup)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title=f"🏦 {BOT_NAME}",
        description=(
            "Ask me about accounts, cards, loans, UPI, KYC, net banking, "
            "and other everyday banking topics. I'm an AI assistant trained "
            "on common support questions — I can't access live account data. "
            "For urgent issues like a lost card or suspected fraud, I'll "
            "point you to our 24x7 helpline when I can't fully resolve it."
        ),
        textbox=gr.Textbox(
            placeholder="e.g. How do I check my account balance?",
            scale=7,
        ),
        examples=[
            "How do I check my account balance?",
            "My debit card got blocked, what should I do?",
            "What documents are needed to open an account?",
            "How can I transfer money using UPI?",
            "I think someone made a fraudulent transaction on my account",
        ],
    )
    demo.launch(share=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = load_dataset()
    priority_lookup = build_priority_lookup(df)
    model, vectorizer = load_or_train_model(df)
    launch_app(model, vectorizer, df, priority_lookup)


if __name__ == "__main__":
    main()
