# Task 5: Decision Tree and Random Forest Classifier on Heart Disease Dataset

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from the Task-5 folder
data = pd.read_csv("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-5//heart.csv")

# Split data into input features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 1. Decision Tree Classifier
# --------------------------

# Create and train the model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate
dt_predictions = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("Classification Report (Decision Tree):\n", classification_report(y_test, dt_predictions))

# Visualize the decision tree (simplified depth for clarity)
plt.figure(figsize=(15, 7))
plot_tree(dt_model, filled=True, max_depth=2, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.title("Decision Tree (Depth=2)")
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-5//decision_tree.png")
plt.show()

# --------------------------
# 2. Decision Tree with max_depth=3 (Control Overfitting)
# --------------------------
dt_depth3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth3.fit(X_train, y_train)
depth3_preds = dt_depth3.predict(X_test)
print("Accuracy with max_depth=3:", accuracy_score(y_test, depth3_preds))

# --------------------------
# 3. Random Forest Classifier
# --------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report (Random Forest):\n", classification_report(y_test, rf_preds))

# --------------------------
# 4. Feature Importance
# --------------------------
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-5//feature_importance.png")
plt.show()

# --------------------------
# 5. Cross Validation Score
# --------------------------
dt_scores = cross_val_score(dt_model, X, y, cv=5)
rf_scores = cross_val_score(rf_model, X, y, cv=5)
print("Decision Tree Cross-Validation Accuracy:", dt_scores.mean())
print("Random Forest Cross-Validation Accuracy:", rf_scores.mean())

# 6.CSV File saving
data.to_csv("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-5//heart.csv", index=False)
