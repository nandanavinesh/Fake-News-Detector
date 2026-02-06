# ===============================
# Fake News Detection (ISOT Dataset)
# ===============================

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ===============================
# 1. Load ISOT Dataset
# ===============================

fake = pd.read_csv("News_Dataset/Fake.csv")
real = pd.read_csv("News_Dataset/True.csv")

fake["label"] = 0   # fake
real["label"] = 1   # real

data = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# 2. Text Cleaning Function
# ===============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)     # remove punctuation/numbers
    return text

data["content"] = data["text"].apply(clean_text)

# ===============================
# 3. Features & Labels
# ===============================

X = data["content"]
y = data["label"]

# ===============================
# 4. Train-Test Split (VERY IMPORTANT)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 5. TF-IDF Vectorization
# ===============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ===============================
# 6. Train Model
# ===============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ===============================
# 7. Evaluate Model
# ===============================

y_pred = model.predict(X_test_tfidf)

print("\nModel Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred)*100))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
