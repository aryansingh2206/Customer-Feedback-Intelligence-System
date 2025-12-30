from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def train_tfidf_vectorizer(texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def train_logistic_regression(X, y):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model


def train_naive_bayes(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    return model
