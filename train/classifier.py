import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pickle
import os

df = pd.read_csv("train/landmarks_two_hand.csv")

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
feature_names = (
    [f"L_{ax}{i}" for i in range(21) for ax in ['x', 'y', 'z']] +
    [f"R_{ax}{i}" for i in range(21) for ax in ['x', 'y', 'z']]
)

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




# --------------------------------------------------------------------------------------------------------------
# Cross Evaluation
# --------------------------------------------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print("\nCross Validation Results:")
print(f"  Scores per fold: {[f'{s:.2%}' for s in cv_scores]}")
print(f"  Mean:  {cv_scores.mean():.2%}")
print(f"  Std:   {cv_scores.std():.2%}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
bars = ax.bar(folds, cv_scores * 100, color='steelblue', edgecolor='black', width=0.5)

# Mean line
ax.axhline(cv_scores.mean() * 100, color='red', linestyle='--', linewidth=1.5,
           label=f"Mean: {cv_scores.mean():.2%}")

# Value labels on bars
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{score:.2%}", ha='center', va='bottom', fontsize=10)

ax.set_title("5-Fold Cross Validation Accuracy", fontsize=13)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(cv_scores.min() * 100 - 2, 101)
ax.legend()
plt.tight_layout()
plt.savefig("results/cross_validation.png", dpi=150)
print("Cross validation graph saved to results/cross_validation.png")
plt.show()
