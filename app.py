import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Dataset
# ==========================================

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add Species Column
df["species"] = iris.target
df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

# ==========================================
# Features and Target
# ==========================================

X = df.drop("species", axis=1)
y = df["species"]

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 50)

print(f"\nModel Accuracy : {accuracy*100:.2f}%")

# ==========================================
# Histogram
# ==========================================

X.hist(figsize=(10,8), bins=20)

plt.suptitle("Distribution of Iris Flower Features", fontsize=16)

plt.tight_layout()

plt.show()

# ==========================================
# Pair Plot
# ==========================================

sns.pairplot(
    df,
    hue="species",
    diag_kind="hist"
)

plt.show()

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# Classification Report
# ==========================================

print("\n")
print("="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(classification_report(y_test, y_pred))

# ==========================================
# Feature Importance
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("="*60)
print("FEATURE IMPORTANCE")
print("="*60)

print(feature_importance)

plt.figure(figsize=(8,5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()

# ==========================================
# User Prediction
# ==========================================

print("\n")
print("=" * 50)
print("PREDICT YOUR OWN FLOWER")
print("=" * 50)

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

new_flower = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=X.columns
)

prediction = model.predict(new_flower)

print("\nPredicted Flower Species :", prediction[0])