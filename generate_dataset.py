"""Generates a synthetic product-review dataset (data/reviews.csv).
Swap in a real Kaggle/Amazon dataset with the same columns to use real data."""
import random, pandas as pd
random.seed(42)

PRODUCTS = {
 "NovaPhone X": {"battery": ("The battery easily lasts a full day", "Battery drains way too fast", "Battery life is average"),
                 "camera": ("Camera photos are stunning in daylight", "Camera is blurry in low light", "Camera is decent for the price"),
                 "price": ("Great value for money", "Overpriced for what you get", "Price is about what I expected"),
                 "delivery": ("Delivery was quick and well packed", "Delivery was late and the box was damaged", "Delivery took the usual time")},
 "BeatBuds Pro": {"sound": ("Sound quality is crisp with deep bass", "Sound is tinny and distorted at high volume", "Sound is okay, nothing special"),
                  "battery": ("Battery lasts for days on one charge", "Battery dies within two hours", "Battery is fine for daily use"),
                  "comfort": ("Very comfortable even after hours", "They hurt my ears after twenty minutes", "Comfort is acceptable"),
                  "price": ("Totally worth the price", "Too expensive for the quality", "Fairly priced")},
 "AeroBook Air": {"performance": ("Performance is blazing fast for coding", "Performance lags with multiple tabs open", "Performance is fine for basic tasks"),
                  "display": ("The display is bright and sharp", "The display has terrible washed out colors", "The display is average"),
                  "battery": ("Battery easily gets me through a workday", "Battery barely lasts three hours", "Battery life is decent"),
                  "price": ("Excellent value for a laptop this good", "Way too pricey", "Price is reasonable")},
 "FitPulse Watch": {"battery": ("Battery runs a full week", "Battery needs charging every night", "Battery is okay"),
                    "tracking": ("Fitness tracking is accurate and motivating", "Step and heart rate tracking are wildly inaccurate", "Tracking is acceptable"),
                    "design": ("Looks sleek and premium on the wrist", "Strap feels cheap and itchy", "Design is plain but fine"),
                    "price": ("Amazing features at this price", "Not worth the money", "Price is average")},
}
OPEN = {"pos": ["Loving it!", "Highly recommended.", "Very happy with this purchase.", "Absolutely great."],
        "neg": ["Very disappointed.", "Would not recommend.", "Total waste of money.", "Regret buying this."],
        "neu": ["It is okay overall.", "Does the job.", "Mixed feelings.", "Average product."]}

rows = []
dates = pd.date_range("2025-01-01", "2026-10-31", freq="D")
for i in range(1200):
    prod = random.choice(list(PRODUCTS))
    label = random.choices(["pos", "neg", "neu"], [0.5, 0.3, 0.2])[0]
    aspects = random.sample(list(PRODUCTS[prod]), random.choice([1, 2, 3]))
    parts = [random.choice(OPEN[label])]
    for a in aspects:
        # mostly consistent with label, some mixed reviews
        k = {"pos": 0, "neg": 1, "neu": 2}[label]
        if random.random() < 0.2: k = random.choice([0, 1, 2])
        parts.append(PRODUCTS[prod][a][k] + ".")
    random.shuffle(parts)
    rating = {"pos": random.choice([4, 5, 5]), "neg": random.choice([1, 1, 2]), "neu": 3}[label]
    d = random.choice(dates)
    # simulate a software-update dip/boost for the phone
    rows.append({"review_id": i + 1, "product": prod, "date": d.date(), "rating": rating,
                 "review_text": " ".join(parts),
                 "sentiment": {"pos": "Positive", "neg": "Negative", "neu": "Neutral"}[label]})
pd.DataFrame(rows).to_csv("data/reviews.csv", index=False)
print("Saved data/reviews.csv with", len(rows), "reviews")
