# test_pipeline.py
# End-to-end sanity test for Customer Feedback Intelligence System

print("\n========== STARTING PIPELINE TEST ==========\n")

# --------------------------------------------------
# STEP 1: Preprocessing
# --------------------------------------------------
print("1️⃣ Testing preprocessing...")

from src.preprocessing import load_and_prepare_data

df = load_and_prepare_data("data/clean_reviews.csv")

assert "clean_review" in df.columns, "❌ clean_review column missing"
assert len(df) > 0, "❌ DataFrame is empty"

print("✅ Preprocessing OK")
print(df[["review_text", "clean_review"]].head(2), "\n")


# --------------------------------------------------
# STEP 2: Classical Sentiment ML
# --------------------------------------------------
print("2️⃣ Testing classical ML sentiment...")

from src.sentiment_ml import (
    train_tfidf_vectorizer,
    train_logistic_regression
)

# Create sentiment labels
y = df["rating"].apply(
    lambda r: "negative" if r <= 2 else "neutral" if r == 3 else "positive"
)
X = df["clean_review"]

vectorizer, X_vec = train_tfidf_vectorizer(X)
model = train_logistic_regression(X_vec, y)

preds = model.predict(X_vec[:5])

print("✅ ML Sentiment OK")
print("Predictions:", preds, "\n")


# --------------------------------------------------
# STEP 3: Topic Modeling
# --------------------------------------------------
print("3️⃣ Testing topic modeling (LDA)...")

from src.topic_modeling import train_lda_model, get_topics

lda, count_vec = train_lda_model(df["clean_review"][:1000])
topics = get_topics(lda, count_vec.get_feature_names_out())

print("✅ Topic Modeling OK")
for i, topic in enumerate(topics):
    print(f"Topic {i+1}:", topic)
print()


# --------------------------------------------------
# STEP 4: Named Entity Recognition
# --------------------------------------------------
print("4️⃣ Testing NER...")

from src.ner import extract_entities

sample_text = df["clean_review"].iloc[0]
entities = extract_entities(sample_text)

print("✅ NER OK")
print("Entities:", entities, "\n")


# --------------------------------------------------
# STEP 5: Embeddings & Similarity
# --------------------------------------------------
print("5️⃣ Testing embeddings & similarity search...")

from src.embeddings import generate_embeddings, find_similar_reviews

texts = df["clean_review"].head(50).tolist()
embeddings = generate_embeddings(texts)

query = "slow write speed memory card"
similar_reviews = find_similar_reviews(
    query_text=query,
    corpus_texts=texts,
    corpus_embeddings=embeddings,
    top_k=3
)

print("✅ Embeddings OK")
for score, review in [(r[1], r[0]) for r in similar_reviews]:
    print(f"Score: {score:.3f} | Review: {review[:80]}")
print()


# --------------------------------------------------
# STEP 6: Free LLM Sentiment (Hugging Face)
# --------------------------------------------------
print("6️⃣ Testing free LLM sentiment...")

from src.llm_tasks import llm_sentiment

llm_result = llm_sentiment(df["review_text"].iloc[0])

print("✅ LLM Sentiment OK")
print("LLM Prediction:", llm_result, "\n")


print("========== ALL TESTS PASSED SUCCESSFULLY ==========\n")
