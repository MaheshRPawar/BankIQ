# 🏦 Banking ChatBot

A machine learning–based Natural Language Banking Query Resolution System built with **Python**, **Scikit-learn**, **NLP**, and **Gradio**.

The system is trained on a labeled banking dataset to identify a customer's intent from a free-text question and return the matching support response — including automatically flagging high-priority issues (lost/stolen card, fraud, KYC) with the 24x7 helpline number.

---

## 📖 About the Project

This project demonstrates how traditional machine learning techniques can classify customer intent and retrieve the right response, without relying on external AI APIs or large language models.

It is **not** a generative chatbot. It does not "understand" conversation the way an LLM does — it classifies each incoming question into one of a fixed set of known banking categories, using patterns learned from labeled examples, and returns the pre-written response for that category.

Technical description: **a Natural Language Banking Query Resolution System using NLP-based text preprocessing, TF-IDF feature extraction, and Multinomial Naive Bayes intent classification.**

---

## ✨ Key Features

* 🤖 Machine learning–based intent classification (TF-IDF + Multinomial Naive Bayes)
* 💬 Interactive chat interface built with Gradio
* 🏦 Covers 39 banking intents across accounts, cards, loans, UPI, KYC, and more
* 🛡️ Confidence-based fallback for questions the model can't confidently classify
* 🚨 **Risk-aware escalation** — high-priority categories (lost/stolen card, fraud, KYC, complaints) automatically surface the 24x7 helpline number alongside the normal answer
* 📚 Dataset cleaned of templated/duplicate phrasing for more realistic training data
* ⚡ Automatic model training on first run, cached afterward

---

## 🛠️ Technology Stack

| Component         | Technology              |
| ------------------ | ------------------------ |
| Language           | Python                   |
| Machine Learning    | Scikit-learn              |
| NLP                | TF-IDF + NLTK             |
| Model              | Multinomial Naive Bayes   |
| Data Handling       | Pandas                    |
| User Interface      | Gradio                    |
| Model Persistence   | Joblib                    |

---

## 📂 Repository Structure

```text
Banking-ChatBot/
│
├── banking_chatbot.py            # main app, minimal comments
├── banking_chatbot_explained.py  # identical logic, fully commented for study/viva
├── banking_support.csv           # training data (question, category, priority, response)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/MaheshRPawar/Banking-ChatBot.git
cd Banking-ChatBot
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the chatbot

```bash
python banking_chatbot.py
```

If no trained model files exist yet, the app trains one automatically before launching the chat interface. This creates `banking_chatbot_model.pkl` and `banking_chatbot_vectorizer.pkl` locally — these are intentionally excluded from git via `.gitignore`, since they're regenerated from the dataset.

---

## 🚨 How Risk-Level Escalation Works

Every category in the dataset carries a `priority` value (`low` / `medium` / `high`), assigned per banking topic — for example `fraud_reporting`, `card_block`, `kyc_update`, `credit_card`, and `complaints_feedback` are all `high`.

When the chatbot confidently classifies a question into a `high` priority category, it appends a short notice pointing the user to the 24x7 helpline **in addition to** the normal self-service answer — it never replaces the useful information with just a phone number. Lower-priority categories (balance checks, branch timings, interest rates) get the plain answer with no extra notice.

Genuinely unclear questions (below the confidence threshold) always get the fallback message, which also includes the helpline number.

---

## ⚙️ How It Works

```text
User Input
     │
     ▼
Text Cleaning & Preprocessing
     │
     ▼
TF-IDF Vectorization (unigrams + bigrams)
     │
     ▼
Multinomial Naive Bayes Model
     │
     ▼
Intent Prediction + Confidence Check
     │
     ▼
Response Lookup + Risk-Level Check
     │
     ▼
Chat Interface (Gradio)
```

---

## 🔄 Updating the Dataset

1. Open `banking_support.csv`
2. Add or edit banking questions, categories, priorities, and responses
3. Delete `banking_chatbot_model.pkl` and `banking_chatbot_vectorizer.pkl` if they exist, so the app retrains
4. Run `python banking_chatbot.py`

---

## 📌 Known Limitations

* With 39 categories sharing overlapping vocabulary (e.g. `debit_card` vs `card_block`), the model's confidence scores for correct and incorrect predictions overlap — there's no threshold that perfectly separates them. `CONFIDENCE_THRESHOLD = 0.06` was chosen by analyzing the held-out test set as the best balance point, not a value that guarantees correctness. See the comments in `banking_chatbot_explained.py` for the full analysis.
* Held-out test accuracy is roughly 60% across 39 categories — well above the ~2.6% random baseline, but this is a bag-of-words classifier, not a semantic one, so paraphrased or unusual phrasing can still be missed.
* The `fraud_reporting` response includes its own dedicated fraud hotline number, separate from the general helpline used in the escalation notice — worth reconciling into a single number if this becomes a real deployment.

---

## 📌 Future Enhancements

* Transformer-based NLP models
* Semantic search using embeddings
* Multilingual support
* Retrieval-Augmented Generation (RAG)
* Integration with real banking APIs

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**. It is not affiliated with any real bank or financial institution and does not access live customer accounts, transactions, or banking systems.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Mahesh Pawar**

GitHub: **https://github.com/MaheshRPawar**
