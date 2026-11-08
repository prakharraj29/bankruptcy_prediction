import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# 1. Load dataset
print("Loading data.csv...")
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
target_col = "Bankrupt?"

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Stratified train-test split (20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Export test_data.csv (features + target column)
test_data = X_test.copy()
test_data[target_col] = y_test
test_data.to_csv("test_data.csv", index=False)
print("Saved test_data.csv successfully.")

# 4. Standard Scaling
os.makedirs("models", exist_ok=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# 5. Define 5 mandatory classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, class_weight='balanced', max_iter=2500, random_state=42)
}

# 6. Train, Evaluate & Save
results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, average='weighted'), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    })
    
    file_name = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{file_name}")

results_df = pd.DataFrame(results)
print("\n=== Model Comparison Metrics ===")
print(results_df.to_string(index=False))
