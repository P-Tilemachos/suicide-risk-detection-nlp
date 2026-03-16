import joblib
import re
import string

model = joblib.load("models/suicide_classifier_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    label = "suicide risk" if pred == 1 else "non-suicide"

    return label, prob


if __name__ == "__main__":

    example = "I feel hopeless and empty"

    label, prob = predict(example)

    print(label)
    print(prob)
