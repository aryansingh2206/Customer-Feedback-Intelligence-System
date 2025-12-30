from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Load once (important for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Generate sentence embeddings for a list of texts.
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )
    return embeddings


def find_similar_reviews(
    query_text,
    corpus_texts,
    corpus_embeddings,
    top_k=5
):
    """
    Find top-k semantically similar reviews.
    """
    query_embedding = model.encode([query_text])
    similarities = cosine_similarity(
        query_embedding,
        corpus_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        (corpus_texts[i], similarities[i])
        for i in top_indices
    ]


def cluster_reviews(embeddings, n_clusters=5):
    """
    Cluster reviews using KMeans on embeddings.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )
    labels = kmeans.fit_predict(embeddings)
    return labels
