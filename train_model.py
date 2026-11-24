"""Train and evaluate sentiment models. Saves best pipeline to model.joblib."""
import pandas as pd, joblib, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score

def clean(t):
    t = t.lower()
    return re.sub(r"[^a-z0-9\s!']", " ", t)

df = pd.read_csv("data/reviews.csv").drop_duplicates("review_text")
df["clean"] = df["review_text"].apply(clean)
X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], df["sentiment"], test_size=0.2,
                                          stratify=df["sentiment"], random_state=42)
models = {
  "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
  "NaiveBayes": MultinomialNB(),
}
best, best_f1 = None, -1
for name, clf in models.items():
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    f1 = f1_score(y_te, pred, average="macro")
    print(f"\n=== {name} (macro F1 = {f1:.3f}) ===")
    print(classification_report(y_te, pred))
    if f1 > best_f1: best, best_f1, best_name, best_pred = pipe, f1, name, pred

ConfusionMatrixDisplay.from_predictions(y_te, best_pred, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}"); plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
joblib.dump(best, "model.joblib")
print(f"\nSaved best model: {best_name} (macro F1 {best_f1:.3f})")
