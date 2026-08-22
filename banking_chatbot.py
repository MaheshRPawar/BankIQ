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
import tempfile

import gradio as gr
import joblib
import nltk
import pandas as pd
from gtts import gTTS
from gtts.tts import gTTSError
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

# A question is "vague" if it's just a bare topic mention - a couple of
# words with no verb or question word attached, like "saving account" or
# "loan" on their own - rather than an actual question about that topic.
# We ask the user to be specific instead of guessing what they meant.
VAGUE_QUERY_MAX_WORDS = 3

ACTION_WORDS = {
    "how", "what", "why", "when", "where", "who", "which", "can", "could", "do", "does",
    "is", "are", "will", "would", "should", "need", "want", "please", "help", "explain",
    "show", "tell", "get", "check", "open", "close", "block", "unblock", "apply", "update",
    "change", "reset", "transfer", "pay", "deposit", "withdraw", "activate", "deactivate",
    "cancel", "register", "link", "download", "print", "report", "request", "add", "remove",
    "increase", "decrease", "convert", "know", "find", "locate", "contact", "call", "verify",
    "confirm", "issue", "complain", "complaint", "dispute", "lost", "stolen", "missing",
    "wrong", "error", "problem", "fail", "failed", "not", "working", "decommission", "email",
    "occur", "occurred", "unauthorized", "delay", "delayed", "pending", "reject", "rejected",
    "decline", "declined", "send", "receive", "make", "set", "setup", "enable", "disable",
    "renew", "upgrade", "downgrade", "order", "track", "file", "submit", "provide", "give",
    "procedure", "process", "steps", "way", "ways", "method", "eligibility", "eligible",
}

VAGUE_QUERY_MESSAGE_TEMPLATE = (
    'I\'m not sure what you\'d like to know about "{query}". '
    "Could you specify your question a bit more — for example, what you'd "
    "like to do or find out about it?"
)

BOT_NAME = "BankIQ"

# Indian-accent English via Google's TTS endpoint (tld="co.in" is what
# selects the accent - gTTS otherwise defaults to a US-accented voice).
TTS_LANGUAGE = "en"
TTS_ACCENT_TLD = "co.in"

# Emoji/symbols read awkwardly (or get skipped oddly) by TTS engines, so
# they're stripped before handing text to gTTS. This does not affect the
# text shown in the chat window, only the audio version of it.
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+"
)

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


def is_vague_query(text):
    """
    True if this looks like a bare topic mention ("saving account", "loan",
    "kyc") rather than an actual question about that topic. A handful of
    words with no verb or question word attached gives no real signal of
    what the user actually wants to know, so instead of guessing which of
    several possible questions they meant, we ask them to be specific.
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens or len(tokens) > VAGUE_QUERY_MAX_WORDS:
        return False
    return not any(token in ACTION_WORDS for token in tokens)


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

    if is_vague_query(question):
        clarification = VAGUE_QUERY_MESSAGE_TEMPLATE.format(query=question.strip())
        return clarification, None, 0.0

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
# Text-to-speech (read aloud) - opt-in only, triggered by a button
# ---------------------------------------------------------------------------

def synthesize_speech(text):
    """
    Converts text to an Indian-accent English MP3 using gTTS and returns
    the file path, or None if speech generation failed (e.g. no internet).
    Requires an internet connection - gTTS calls Google's TTS service, it
    does not run offline.
    """
    if not text:
        return None

    cleaned = EMOJI_PATTERN.sub("", text).strip()
    if not cleaned:
        return None

    try:
        tts = gTTS(text=cleaned, lang=TTS_LANGUAGE, tld=TTS_ACCENT_TLD)
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False, dir=tempfile.gettempdir()
        )
        tts.save(temp_file.name)
        return temp_file.name
    except (gTTSError, Exception) as error:
        print(f"Text-to-speech failed: {error}")
        return None


# ---------------------------------------------------------------------------
# Gradio chat interface
# ---------------------------------------------------------------------------

# Refocuses the message box after every send, so the user can keep typing
# without clicking back into it - Gradio's own DOM update after clearing
# the textbox value otherwise drops focus.
FOCUS_INPUT_JS = """
() => {
    const el = document.querySelector('#chat_input textarea');
    if (el) { el.focus(); }
}
"""

EXAMPLE_QUESTIONS = [
    "How do I check my account balance?",
    "My debit card got blocked, what should I do?",
    "What documents are needed to open an account?",
    "How can I transfer money using UPI?",
    "I think someone made a fraudulent transaction on my account",
]


def launch_app(model, vectorizer, df, priority_lookup):

    def respond(message, chat_history):
        if not message or not message.strip():
            return chat_history, "", "", gr.update()
        response, _category, _confidence = get_response(
            message, model, vectorizer, df, priority_lookup
        )
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        # hide the example-questions block once a real conversation starts,
        # so it's only ever seen before the first message
        return chat_history, "", response, gr.update(visible=False)

    def read_last_reply(last_response):
        audio_path = synthesize_speech(last_response)
        if audio_path is None:
            gr.Warning(
                "Couldn't generate audio right now. This feature needs an "
                "internet connection."
            )
        return audio_path

    with gr.Blocks(title=f"🏦 {BOT_NAME}") as demo:
        gr.Markdown(f"## 🏦 {BOT_NAME}")
        gr.Markdown(
            "Ask me about accounts, cards, loans, UPI, KYC, net banking, "
            "and other everyday banking topics. I'm an AI assistant trained "
            "on common support questions — I can't access live account data. "
            "For urgent issues like a lost card or suspected fraud, I'll "
            "point you to our 24x7 helpline when I can't fully resolve it."
        )

        # created now (so gr.Examples and the input row below can both
        # reference it) but not displayed until msg_box.render() is called
        # further down - this lets it visually end up right above the
        # Send button, below the chat window, while still being wired up
        # to the example buttons defined above it
        msg_box = gr.Textbox(
            placeholder="e.g. How do I check my account balance?",
            scale=8,
            show_label=False,
            elem_id="chat_input",
            render=False,
        )

        # Shown immediately, above the fold, with no chat window in the
        # way - visible only until the first message is sent, then hidden.
        with gr.Column(visible=True) as examples_block:
            gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=msg_box, label="Try asking:")

        # Read-aloud controls sit directly above the chat window, not
        # buried below the input box, so they're the first thing visible
        # next to whatever the bot just said.
        with gr.Row():
            read_aloud_btn = gr.Button("🔊 Read last reply", scale=1)
            audio_output = gr.Audio(
                autoplay=True, show_label=False, interactive=False, scale=3
            )

        chatbot = gr.Chatbot(height=420, label=None)
        last_response_state = gr.State("")

        with gr.Row():
            msg_box.render()
            send_btn = gr.Button("Send", scale=1, variant="primary")

        submit_outputs = [chatbot, msg_box, last_response_state, examples_block]

        msg_box.submit(respond, [msg_box, chatbot], submit_outputs).then(
            fn=None, js=FOCUS_INPUT_JS
        )
        send_btn.click(respond, [msg_box, chatbot], submit_outputs).then(
            fn=None, js=FOCUS_INPUT_JS
        )
        read_aloud_btn.click(
            read_last_reply, [last_response_state], [audio_output]
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
