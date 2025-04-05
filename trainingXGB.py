import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from xgboost import XGBClassifier

# === Load and prepare data ===
df = pd.read_csv('paired_data.csv')

# Feature columns
feature_cols = [
    'similarity_score',
    'experience_years',
    'title_match_score',
    'resume_length',
    'job_length'
]

# Handle missing
for col in feature_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

X = df[feature_cols]
y = df['label']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Preprocessor (scaling only, for numerical) ===
transform = ColumnTransformer([
    ('scaler', StandardScaler(), feature_cols)
], remainder='passthrough')

# === Define base XGB model ===
xgb_base = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# === Hyperparameter grid ===
param_grid = {
    'n_estimators': [100, 200],          # Enough trees for XGB to learn patterns, not too much to delay
    'max_depth': [3, 5],                 # 3 is fast/safe, 5 gives room for more depth
    'learning_rate': [0.1, 0.2],         # Higher rates reduce training rounds (faster convergence)
    'subsample': [0.8, 1.0],             # Controls overfitting, 0.8 is a common default
    'colsample_bytree': [0.8, 1.0],      # Column sampling for tree splits (boosts generalization)
    'gamma': [0, 0.1],                   # Minimal cost for split. 0.1 slows growth slightly
}

# === CV object ===
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# === Grid search ===
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    refit=True
)

# === Pipeline ===
pipe = Pipeline([
    ('preprocess', transform),
    ('classifier', grid_search)
])

# === Fit ===
pipe.fit(X_train, y_train)

# === Predict and Evaluate ===
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix - GridSearch XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC AUC
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve - GridSearch XGBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Feature Importances ===
best_model = pipe.named_steps['classifier'].best_estimator_
importances = best_model.feature_importances_
plt.barh(feature_cols, importances)
plt.title("Feature Importances - Best XGBoost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
