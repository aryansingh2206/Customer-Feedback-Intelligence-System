from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def train_lda_model(texts, n_topics=5):
    vectorizer = CountVectorizer(
        max_df=0.9,
        min_df=20,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda.fit(X)

    return lda, vectorizer


def get_topics(model, feature_names, top_words=10):
    topics = []
    for topic in model.components_:
        topics.append(
            [feature_names[i] for i in topic.argsort()[-top_words:][::-1]]
        )
    return topics
