import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("train/landmarks.csv")

# Check before cleaning
print(f"Total rows before cleaning: {len(df)}")
print(f"NaN rows: {df.isnull().any(axis=1).sum()}")

# Drop rows with any NaN value
df = df.dropna()

print(f"Total rows after cleaning: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.2%}")

with open("models/gesture_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/gesture_classifier.pkl")