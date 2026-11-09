"""Задания второй части лабораторной: Корреляционный анализ."""
from __future__ import annotations

import pandas as pd
import numpy as np

from grader_contracts.correlation_tasks import BrainDataInput, BrainCorrelationSummary


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Загружает brainsize.txt и возвращает корреляционную сводку."""
    df = pd.read_csv(data.path, sep="\t", na_values=["NA"])
    df.columns = [c.strip() for c in df.columns]

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    target = "MRI_Count"

    for col in features + [target]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    gender_col = "Gender" if "Gender" in df.columns else "Sex"
    df[gender_col] = df[gender_col].astype(str).str.strip().str.lower()

    males = df[df[gender_col].isin(["male", "m", "man"])]
    females = df[df[gender_col].isin(["female", "f", "woman"])]

    def get_corr(sub_df):
        res = {}
        for f in features:
            pair = sub_df[[f, target]].dropna()
            if len(pair) >= 2:
                corr = pair[f].corr(pair[target], method="pearson")
                res[f] = float(corr) if pd.notna(corr) else float("nan")
            else:
                res[f] = float("nan")
        return res

    male_correlations = get_corr(males)
    female_correlations = get_corr(females)

    strongest_feature = None
    max_abs_corr = -1.0

    for f in features:
        for corr in (male_correlations[f], female_correlations[f]):
            if not np.isnan(corr) and abs(corr) > max_abs_corr:
                max_abs_corr = abs(corr)
                strongest_feature = f

    return BrainCorrelationSummary(
        male_correlations=male_correlations,
        female_correlations=female_correlations,
        strongest_mri_feature=strongest_feature,
    )
