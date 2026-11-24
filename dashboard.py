"""Streamlit dashboard: run with  streamlit run dashboard.py"""
import re, joblib, pandas as pd, plotly.express as px, streamlit as st
from collections import Counter

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="wide")
COLORS = {"Positive": "#2e9e5b", "Neutral": "#e0a800", "Negative": "#d64545"}
ASPECTS = {"battery": ["battery", "charge", "charging"], "camera": ["camera", "photo"],
           "price": ["price", "value", "money", "expensive", "pricey", "worth"],
           "delivery": ["delivery", "packed", "box", "late"], "sound": ["sound", "bass"],
           "comfort": ["comfortable", "ears", "comfort"], "performance": ["performance", "fast", "lag"],
           "display": ["display", "screen", "colors"], "tracking": ["tracking", "steps", "heart"],
           "design": ["design", "strap", "looks", "sleek"]}
STOP = set("the a an and is are was it i to of for in on this that with at as be so too very after".split())

@st.cache_data
def load():
    d = pd.read_csv("data/reviews.csv", parse_dates=["date"]); d["month"] = d["date"].dt.to_period("M").dt.to_timestamp(); return d
@st.cache_resource
def model(): return joblib.load("model.joblib")

df = load()
st.title("💬 Product Review Sentiment Analyzer")

with st.sidebar:
    st.header("Filters")
    prods = st.multiselect("Product", sorted(df["product"].unique()), default=sorted(df["product"].unique()))
    rng = st.date_input("Date range", (df["date"].min(), df["date"].max()))
f = df[df["product"].isin(prods)]
if len(rng) == 2: f = f[(f["date"] >= pd.Timestamp(rng[0])) & (f["date"] <= pd.Timestamp(rng[1]))]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", len(f)); c2.metric("Avg rating", f"{f['rating'].mean():.2f} ⭐")
c3.metric("% Positive", f"{(f['sentiment']=='Positive').mean()*100:.1f}%")
c4.metric("% Negative", f"{(f['sentiment']=='Negative').mean()*100:.1f}%")

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Trends", "Aspects & Keywords", "Compare products", "Live predictor"])

with t1:
    a, b = st.columns(2)
    a.plotly_chart(px.pie(f, names="sentiment", color="sentiment", color_discrete_map=COLORS, hole=.45, title="Sentiment split"), use_container_width=True)
    b.plotly_chart(px.histogram(f, x="rating", color="sentiment", color_discrete_map=COLORS, title="Rating distribution"), use_container_width=True)
    st.dataframe(f.sort_values("date", ascending=False).head(50), use_container_width=True)

with t2:
    m = f.groupby(["month", "sentiment"]).size().reset_index(name="count")
    st.plotly_chart(px.line(m, x="month", y="count", color="sentiment", color_discrete_map=COLORS, markers=True, title="Reviews per month"), use_container_width=True)
    pct = f.assign(pos=(f["sentiment"] == "Positive")).groupby("month")["pos"].mean().mul(100).reset_index()
    st.plotly_chart(px.line(pct, x="month", y="pos", markers=True, title="% positive over time"), use_container_width=True)

with t3:
    rows = []
    for _, r in f.iterrows():
        for asp, kws in ASPECTS.items():
            if any(k in r["review_text"].lower() for k in kws): rows.append((asp, r["sentiment"]))
    ad = pd.DataFrame(rows, columns=["aspect", "sentiment"]).value_counts().reset_index(name="count")
    st.plotly_chart(px.bar(ad, x="aspect", y="count", color="sentiment", color_discrete_map=COLORS, barmode="group", title="Aspect-level sentiment (keyword based)"), use_container_width=True)
    a, b = st.columns(2)
    for col, s in ((a, "Positive"), (b, "Negative")):
        words = Counter(w for t in f[f["sentiment"] == s]["review_text"] for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP and len(w) > 3)
        wd = pd.DataFrame(words.most_common(12), columns=["word", "count"])
        col.plotly_chart(px.bar(wd, x="count", y="word", orientation="h", title=f"Top words - {s}", color_discrete_sequence=[COLORS[s]]), use_container_width=True)

with t4:
    cmp = df.groupby(["product", "sentiment"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(cmp, x="product", y="count", color="sentiment", color_discrete_map=COLORS, barmode="stack"), use_container_width=True)
    st.dataframe(df.groupby("product").agg(avg_rating=("rating", "mean"), reviews=("review_id", "count")).round(2))

with t5:
    txt = st.text_area("Paste a review", "Battery life is great but the camera is blurry in low light.")
    if st.button("Analyze") and txt.strip():
        pipe = model(); pred = pipe.predict([txt.lower()])[0]; proba = pipe.predict_proba([txt.lower()])[0]
        st.subheader(f"Prediction: :{ {'Positive':'green','Negative':'red','Neutral':'orange'}[pred] }[{pred}]")
        st.plotly_chart(px.bar(x=pipe.classes_, y=proba, labels={"x": "Class", "y": "Probability"}, color=pipe.classes_, color_discrete_map=COLORS), use_container_width=True)
        found = [a for a, k in ASPECTS.items() if any(w in txt.lower() for w in k)]
        st.write("Aspects mentioned:", ", ".join(found) or "none detected")
