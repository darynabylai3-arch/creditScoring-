"""Обучение модели кредитного скоринга (CatBoost) на credit_risk_dataset.csv.

Запуск:  python train.py [путь_к_csv]
Результат: model.cbm и meta.json рядом со скриптом.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "credit_risk_dataset.csv"

NUM = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CAT = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]

df = pd.read_csv(CSV)

# Чистка выбросов (в датасете встречаются возраст 144 и стаж 123 года)
df = df[(df["person_age"] <= 100) & (df["person_emp_length"].fillna(0) <= 60)]

# В ноутбуке было fillna(df[col].mean) без скобок - это подставляло метод, а не число.
# Здесь заполняем медианой, а значения сохраняем для работы приложения.
medians = {c: float(df[c].median()) for c in ["person_emp_length", "loan_int_rate"]}
for c, v in medians.items():
    df[c] = df[c].fillna(v)

X, y = df[FEATURES].copy(), df["loan_status"]
for c in CAT:
    X[c] = X[c].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    loss_function="Logloss", eval_metric="AUC",
    random_seed=42, verbose=100,
)
model.fit(
    X_tr, y_tr, cat_features=CAT,
    eval_set=(X_te, y_te), early_stopping_rounds=50,
)

auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
print(f"ROC-AUC на тесте: {auc:.4f}")

model.save_model(str(BASE / "model.cbm"))

meta = {
    "features": FEATURES,
    "cat_features": CAT,
    "medians": medians,
    "auc": round(auc, 4),
    "ref_medians": {c: float(X[c].median()) for c in NUM},
    "categories": {c: sorted(X[c].unique().tolist()) for c in CAT},
}
(BASE / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print("Сохранено: model.cbm, meta.json")
