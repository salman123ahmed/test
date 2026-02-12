from matplotlib.pylab import matrix
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("data/unknown.csv")
df = df.dropna()
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
df = df.head(1000)  # Use a subset for quick training
df = df[df["label"].isin([0, 1])]  # Binary classification
df["label"] = df["label"].astype(int)
X = df[["feature1", "feature2"]]
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)


X.plot(X["feature1"], X["feature2"], c=y, cmap="viridis", marker="o", edgecolor="k")    
y.plot(X["feature1"], X["feature2"], c=y, cmap="viridis", marker="o", edgecolor="k")



 #confusion matrix
from sklearn.metrics import confusion_matrix    
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
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



print("alll the chages for teh testing the git hub actions")
 

