import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pickle
import os

df = pd.read_csv("train/landmarks_new.csv")

print(f"Total rows before cleaning: {len(df)}")
print(f"NaN rows: {df.isnull().any(axis=1).sum()}")
df = df.dropna()
print(f"Total rows after cleaning: {len(df)}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- Metrics ---
print("\n" + "=" * 50)
print(f"Overall Accuracy: {acc:.2%}")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
labels = model.classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Confusion Matrix — Accuracy: {acc:.2%}", fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to results/confusion_matrix.png")
plt.show()

# --- Feature Importance ---
importances = model.feature_importances_
axes = ['x', 'y', 'z']
feature_names = [f"{ax}{i}" for i in range(21) for ax in axes]

top_n = 20
indices = np.argsort(importances)[::-1][:top_n]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(top_n), importances[indices])
ax.set_xticks(range(top_n))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', fontsize=9)
ax.set_title(f"Top {top_n} Most Important Landmarks", fontsize=13)
ax.set_ylabel("Importance Score")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
print("Feature importance saved to results/feature_importance.png")
plt.show()

# --- Per-class Accuracy ---
print("\nPer-class Accuracy:")
for label in labels:
    mask = y_test == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {label:<30} {class_acc:.2%}  ({mask.sum()} samples)")

# --- Save model ---
with open("models/gesture_classifier_new.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to models/gesture_classifier.pkl")