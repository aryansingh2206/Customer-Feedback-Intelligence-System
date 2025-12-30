from transformers import pipeline

zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

LABELS = ["positive", "neutral", "negative"]


def llm_sentiment(text: str) -> str:
    result = zero_shot_classifier(text, LABELS)
    return result["labels"][0]
