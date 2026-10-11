"""FastAPI-приложение кредитного скоринга.

Запуск:  uvicorn main:app --reload
"""
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
if not (BASE / "model.cbm").exists():
    raise SystemExit("Не найден model.cbm - сначала выполните: python train.py")

model = CatBoostClassifier()
model.load_model(str(BASE / "model.cbm"))
meta = json.loads((BASE / "meta.json").read_text())
FEATURES, CAT = meta["features"], meta["cat_features"]

app = FastAPI(title="Кредитный скоринг")

# Человекочитаемые названия признаков
LABELS = {
    "person_age": "Возраст",
    "person_income": "Годовой доход",
    "person_home_ownership": "Жильё",
    "person_emp_length": "Стаж работы",
    "loan_intent": "Цель кредита",
    "loan_grade": "Кредитный рейтинг (грейд)",
    "loan_amnt": "Сумма кредита",
    "loan_int_rate": "Процентная ставка",
    "loan_percent_income": "Доля кредита от дохода",
    "cb_person_default_on_file": "Просрочки в истории",
    "cb_person_cred_hist_length": "Длина кредитной истории",
}
HOME = {"RENT": "аренда", "OWN": "своё жильё", "MORTGAGE": "ипотека", "OTHER": "другое"}
INTENT = {
    "PERSONAL": "личные нужды", "EDUCATION": "образование", "MEDICAL": "лечение",
    "VENTURE": "бизнес", "HOMEIMPROVEMENT": "ремонт", "DEBTCONSOLIDATION": "рефинансирование долгов",
}


class Applicant(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0, le=40)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=60)


def value_text(f: str, row: dict) -> str:
    v = row[f]
    if f == "person_age":
        return f"{v} лет"
    if f in ("person_income", "loan_amnt"):
        return f"{v:,.0f}".replace(",", " ")
    if f == "person_emp_length":
        return f"{v:g} лет"
    if f == "cb_person_cred_hist_length":
        return f"{v} лет"
    if f == "loan_int_rate":
        return f"{v:g}%"
    if f == "loan_percent_income":
        return f"{v * 100:.0f}% дохода"
    if f == "person_home_ownership":
        return HOME.get(v, v)
    if f == "loan_intent":
        return INTENT.get(v, v)
    if f == "cb_person_default_on_file":
        return "были" if v == "Y" else "не было"
    return str(v)


def advice(f: str, good: bool, row: dict) -> str:
    """Короткое пояснение для пользователя."""
    if f == "loan_percent_income":
        return ("Нагрузка на доход умеренная." if good
                else "Кредит слишком велик относительно дохода - попробуйте уменьшить сумму.")
    if f == "person_income":
        return "Доход поддерживает такой кредит." if good else "Доход невысок для запрашиваемых условий."
    if f == "loan_grade":
        return "Хороший рейтинг снижает риск." if good else "Низкий рейтинг заметно повышает риск."
    if f == "loan_int_rate":
        return "Ставка невысокая." if good else "Высокая ставка делает платежи тяжёлыми и сигналит о риске."
    if f == "cb_person_default_on_file":
        return "Чистая кредитная история." if good else "Прошлые просрочки сильно ухудшают оценку."
    if f == "person_home_ownership":
        return "Тип жилья повышает надёжность." if good else "Аренда жилья считается менее стабильной."
    if f == "person_emp_length":
        return "Стабильный стаж работы." if good else "Небольшой стаж работы."
    if f == "cb_person_cred_hist_length":
        return "Достаточно длинная кредитная история." if good else "Короткая кредитная история."
    if f == "loan_amnt":
        return "Сумма умеренная." if good else "Крупная сумма повышает риск."
    if f == "loan_intent":
        return "Цель кредита считается надёжной." if good else "Для такой цели риск невозврата выше."
    if f == "person_age":
        return "Возраст положительно влияет на оценку." if good else "Возраст слегка снижает оценку."
    return ""


@app.post("/api/score")
def score(a: Applicant):
    row = a.model_dump()
    row["loan_percent_income"] = round(a.loan_amnt / a.person_income, 4)
    X = pd.DataFrame([row])[FEATURES]
    for c in CAT:
        X[c] = X[c].astype(str)

    p_default = float(model.predict_proba(X)[0, 1])
    chance = 1 - p_default

    # SHAP: вклад каждого признака в лог-шансы ДЕФОЛТА (последний столбец - base value)
    shap = model.get_feature_importance(Pool(X, cat_features=CAT), type="ShapValues")[0][:-1]

    factors = []
    for f, s in zip(FEATURES, shap):
        good = s < 0  # снижает риск дефолта => хорошо
        factors.append({
            "feature": f,
            "label": LABELS[f],
            "value": value_text(f, row),
            "impact": round(float(-s), 3),  # >0 помогает одобрению, <0 мешает
            "good": bool(good),
            "text": advice(f, good, row),
        })
    factors.sort(key=lambda d: abs(d["impact"]), reverse=True)

    if chance >= 0.75:
        verdict, level = "Высокий шанс одобрения", "good"
    elif chance >= 0.5:
        verdict, level = "Средний шанс одобрения", "mid"
    else:
        verdict, level = "Низкий шанс одобрения", "bad"

    return {
        "chance": round(chance * 100, 1),
        "default_risk": round(p_default * 100, 1),
        "score": int(round(300 + 550 * chance)),  # условная шкала 300-850
        "verdict": verdict,
        "level": level,
        "loan_percent_income": round(row["loan_percent_income"] * 100, 1),
        "factors": factors,
    }


@app.get("/api/meta")
def get_meta():
    return {"auc": meta.get("auc"), "categories": meta["categories"]}


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "templates" / "index.html").read_text(encoding="utf-8")
