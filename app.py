import streamlit as st
import pandas as pd

from src.preprocessing import load_and_prepare_data
from src.sentiment_ml import (
    train_tfidf_vectorizer,
    train_logistic_regression
)
from src.topic_modeling import train_lda_model, get_topics
from src.ner import extract_entities
from src.embeddings import generate_embeddings, find_similar_reviews
from src.llm_tasks import llm_sentiment


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Feedback Intelligence System",
    layout="wide"
)

st.title(" Customer Feedback Intelligence System")
st.caption("Classical NLP + LLMs + Embeddings (End-to-End Demo)")


# --------------------------------------------------
# Load data (cached)
# --------------------------------------------------
@st.cache_data
def load_data():
    return load_and_prepare_data("data/clean_reviews.csv")


df = load_data()


# --------------------------------------------------
# Train ML models (cached)
# --------------------------------------------------
@st.cache_resource
def train_sentiment_model(texts, ratings):
    y = ratings.apply(
        lambda r: "negative" if r <= 2 else "neutral" if r == 3 else "positive"
    )
    vectorizer, X = train_tfidf_vectorizer(texts)
    model = train_logistic_regression(X, y)
    return vectorizer, model


vectorizer, sentiment_model = train_sentiment_model(
    df["clean_review"],
    df["rating"]
)


# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview",
    " Sentiment",
    " Topics",
    " Entities",
    " Similarity"
])


# =========================
# TAB 1 — Overview
# =========================
with tab1:
    st.subheader("Dataset Overview")
    st.write("Total reviews:", len(df))
    st.write("Rating distribution:")
    st.bar_chart(df["rating"].value_counts().sort_index())

    st.dataframe(df[["review_text", "rating"]].sample(5))


# =========================
# TAB 2 — Sentiment
# =========================
with tab2:
    st.subheader("Sentiment Analysis")

    review = st.text_area(
        "Enter a customer review:",
        height=150
    )

    if st.button("Analyze Sentiment"):
        if review.strip():
            # ML sentiment
            review_clean = review.lower()
            review_vec = vectorizer.transform([review_clean])
            ml_pred = sentiment_model.predict(review_vec)[0]

            # LLM sentiment
            llm_pred = llm_sentiment(review)

            st.success(f" ML Prediction: **{ml_pred}**")
            st.info(f" LLM Prediction: **{llm_pred}**")

        else:
            st.warning("Please enter a review.")


# =========================
# TAB 3 — Topic Modeling
# =========================
with tab3:
    st.subheader("Topic Modeling (LDA)")

    lda, count_vec = train_lda_model(df["clean_review"][:1000])
    topics = get_topics(
        lda,
        count_vec.get_feature_names_out()
    )

    for i, topic in enumerate(topics):
        st.markdown(f"**Topic {i+1}:** {', '.join(topic)}")


# =========================
# TAB 4 — Entity Extraction
# =========================
with tab4:
    st.subheader("Named Entity Recognition")

    review = st.text_area(
        "Enter review text for entity extraction:",
        height=150,
        key="ner_text"
    )

    if st.button("Extract Entities"):
        if review.strip():
            entities = extract_entities(review)
            if entities:
                st.write(entities)
            else:
                st.info("No relevant entities found.")
        else:
            st.warning("Please enter text.")


# =========================
# TAB 5 — Similarity
# =========================
with tab5:
    st.subheader("Semantic Similarity Search")

    sample_df = df.head(100)
    texts = sample_df["clean_review"].tolist()
    embeddings = generate_embeddings(texts)

    query = st.text_input(
        "Enter a query (e.g., 'slow write speed memory card')"
    )

    if st.button("Find Similar Reviews"):
        if query.strip():
            results = find_similar_reviews(
                query,
                texts,
                embeddings
            )

            for review, score in results:
                st.markdown(f"**Score:** {score:.3f}")
                st.write(review)
                st.markdown("---")
        else:
            st.warning("Please enter a query.")
