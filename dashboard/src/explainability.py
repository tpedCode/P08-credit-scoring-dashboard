import joblib
import pandas as pd
import shap

from pathlib import Path
from src.business_labels import get_feature_label, get_feature_description


# ==================================================
# CHEMINS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "models" / "feature_names.pkl"


# ==================================================
# CHARGEMENT MODELE
# ==================================================

def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, feature_names


# ==================================================
# SHAP LOCAL
# ==================================================

def compute_local_shap(client_features: pd.DataFrame, top_n: int = 10):
    """
    Calcule les contributions SHAP locales pour un client.

    Paramètres :
    - client_features : DataFrame d'une seule ligne contenant les 246 features modèle ;
    - top_n : nombre de variables principales à retourner.

    Retour :
    - DataFrame trié par importance absolue des contributions SHAP.
    """

    model, feature_names = load_model_and_features()

    client_features = client_features.reindex(
        columns=feature_names,
        fill_value=0
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(client_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_array = shap_values[0]

    local_explanation = pd.DataFrame({
        "feature": feature_names,
        "feature_label": [get_feature_label(feature) for feature in feature_names],
        "feature_description": [get_feature_description(feature) for feature in feature_names],
        "value": client_features.iloc[0].values,
        "shap_value": shap_array
    })

    local_explanation["abs_shap_value"] = local_explanation["shap_value"].abs()

    local_explanation["impact"] = local_explanation["shap_value"].apply(
        lambda value: "Augmente le risque" if value > 0 else "Réduit le risque"
    )

    local_explanation = (
        local_explanation
        .sort_values("abs_shap_value", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return local_explanation


# ==================================================
# SEPARATION RISQUE / PROTECTION
# ==================================================

def split_shap_impacts(local_explanation: pd.DataFrame):
    """
    Sépare les variables selon leur effet local sur la prédiction.

    Retour :
    - contributions_risk : variables qui augmentent le risque ;
    - contributions_protective : variables qui réduisent le risque.
    """

    contributions_risk = (
        local_explanation[local_explanation["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
        .reset_index(drop=True)
    )

    contributions_protective = (
        local_explanation[local_explanation["shap_value"] < 0]
        .sort_values("shap_value", ascending=True)
        .reset_index(drop=True)
    )

    return contributions_risk, contributions_protective


# ==================================================
# PHRASE METIER
# ==================================================

def build_shap_business_sentence(row):
    """
    Construit une phrase métier simple pour une variable SHAP.
    """

    feature_label = row["feature_label"]
    impact = row["impact"].lower()
    description = row["feature_description"]

    return (
        f"{feature_label} : {description} "
        f"Pour ce client, cette variable {impact} estimé par le modèle."
    )