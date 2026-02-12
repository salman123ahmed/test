import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("data/unknown.csv")

X = df[["feature1", "feature2"]]
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

    # Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


print("Model trained and saved to model.pkl")

print("Model trained and saved to model.pkl")
print("Model trained and saved to model.pkl")

