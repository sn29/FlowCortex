import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# -----------------------
# Data loader
# -----------------------
def load_clean_data(folder, marker_x, marker_y):
    dfs = []
    for fname in os.listdir(folder):
        if fname.endswith("_clean.csv"):
            df = pd.read_csv(os.path.join(folder, fname))
            df = df[df["Phenotype_Label"].isin([0, 1])]  # only Q1 (1) and Q3 (0)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# -----------------------
# Training
# -----------------------
def train_models(df, marker_x, marker_y, test_size=0.2, random_state=42):
    X = df[[marker_x, marker_y]].values
    y = df["Phenotype_Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_preds = logreg.predict(X_test)
    logreg_probs = logreg.predict_proba(X_test)[:, 1]

    # SVM
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_probs = svm.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, logreg_preds))
    print("AUC:", roc_auc_score(y_test, logreg_probs))

    print("\n=== SVM (RBF Kernel) ===")
    print(classification_report(y_test, svm_preds))
    print("AUC:", roc_auc_score(y_test, svm_probs))

    return logreg, svm


# -----------------------
# Save models
# -----------------------
def save_models(logreg, svm, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(logreg, os.path.join(out_dir, "logreg_model.joblib"))
    joblib.dump(svm, os.path.join(out_dir, "svm_model.joblib"))
    print(f"\n✅ Models saved in {out_dir}/")


# -----------------------
# CLI entry
# -----------------------
if __name__ == "__main__":
    clean_folder = "/content/data/FlowSense_Testing/Clean_Labeled"
    marker_x, marker_y = "Alexa.647.A", "FITC.A"

    df_all = load_clean_data(clean_folder, marker_x, marker_y)
    logreg, svm = train_models(df_all, marker_x, marker_y)
    save_models(logreg, svm)
