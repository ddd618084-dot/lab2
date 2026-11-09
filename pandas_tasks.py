"""Задания первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic."""
    df = pd.read_csv(data.path)

    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}

    passengers_over_30 = int((df["Age"] > 30).sum())

    mean_age_by_class = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Age"].mean().items()
    }

    survival_rate_by_class = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Survived"].mean().items()
    }

    top_fare = [
        float(x)
        for x in df["Fare"].dropna().sort_values(ascending=False).head(5).tolist()
    ]

    return TitanicSummary(
        missing_values=missing_values,
        passengers_over_30=passengers_over_30,
        mean_age_by_class=mean_age_by_class,
        survival_rate_by_class=survival_rate_by_class,
        top_fare=top_fare,
    )
