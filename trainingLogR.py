import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the paired data
df = pd.read_csv('paired_data.csv')

# Define all available numerical features
feature_cols = [
    'similarity_score',
    'experience_years',
    'title_match_score',
    'resume_length',
    'job_length'
]

# Ensure the columns exist and fill any missing values
for col in feature_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)
    else:
        print(f"Warning: {col} not found in dataset!")

X = df[feature_cols]
y = df['label']

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# ColumnTransformer with scaling
transform = ColumnTransformer([
    ('scaler', StandardScaler(), feature_cols)
])

# Pipeline with Logistic Regression
logit_pipe = Pipeline([
    ('preprocess', transform),
    ('classifier', LogisticRegression(
        solver='saga',
        max_iter=5000,
        class_weight='balanced',
        n_jobs=-1,
        verbose=1,
        random_state=42
    ))
])

# Train the model
logit_pipe.fit(X_train, y_train)

# Predict
y_pred = logit_pipe.predict(X_test)
y_proba = logit_pipe.predict_proba(X_test)[:, 1]

# === Evaluation ===
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
