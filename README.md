
# Customer Feedback Intelligence System

An end-to-end NLP system that analyzes customer reviews using **classical machine learning, free large language models, and sentence embeddings** to extract actionable business insights.

---

##  Project Overview

This project processes large-scale customer reviews to perform:
- **Sentiment Analysis** (ML vs LLM comparison)
- **Topic Modeling** (LDA)
- **Named Entity Recognition (NER)**
- **Semantic Similarity Search using Embeddings**

The system is modular, reproducible, and demonstrated through an interactive **Streamlit dashboard**.

---

##  Key Features

- **Classical NLP**
  - TF-IDF + Logistic Regression
  - Naive Bayes baseline
  - LDA topic modeling
  - spaCy-based entity extraction

- **LLM-Based Intelligence (Free & Local)**
  - Zero-shot sentiment classification using Hugging Face models
  - No paid APIs or API keys required

- **Embeddings**
  - Sentence Transformers (`all-MiniLM-L6-v2`)
  - Semantic similarity search
  - Review clustering

- **Engineering Best Practices**
  - Modular `src/` architecture
  - Notebook → production code workflow
  - End-to-end test pipeline
  - Streamlit POC for visualization

---

##  Project Structure

```

customer-feedback-ai/
│
├── data/
│   ├── amazon_reviews.csv
│   └── clean_reviews.csv
│
├── notebooks/
│   ├── eda_preprocessing.ipynb
│   ├── topic_modeling.ipynb
│   └── ner.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── sentiment_ml.py
│   ├── topic_modeling.py
│   ├── ner.py
│   ├── embeddings.py
│   └── llm_tasks.py
│
├── app.py              # Streamlit app
├── test_pipeline.py    # End-to-end validation script
├── requirements.txt
├── .gitignore
└── README.md

````

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

##  Verify the Pipeline

Run the complete system sanity check:

```bash
python test_pipeline.py
```

This validates:

* Preprocessing
* Sentiment ML models
* Topic modeling
* NER
* Embeddings & similarity search
* Free LLM sentiment inference

---

##  Run the Streamlit App

```bash
streamlit run app.py
```

The dashboard allows:

* ML vs LLM sentiment comparison
* Topic exploration
* Entity extraction
* Semantic similarity search

---

##  Model Comparison Summary

| Task               | Classical ML          | LLM (Free)              |
| ------------------ | --------------------- | ----------------------- |
| Sentiment Accuracy | High (dominant class) | Better on neutral/mixed |
| Interpretability   | High                  | Medium                  |
| Cost               | Low                   | Higher compute          |
| Latency            | Fast                  | Slower                  |
| Flexibility        | Limited               | High                    |

---

