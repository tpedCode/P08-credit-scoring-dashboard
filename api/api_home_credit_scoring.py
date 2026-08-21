# ==================================================
# API FASTAPI - HOME CREDIT SCORING
# ==================================================
"""
API de scoring crédit Home Credit.

Principe métier :
- recevoir les données déjà préparées par le Feature Engineering ;
- calculer la probabilité de défaut ;
- appliquer le seuil métier optimisé ;
- retourner ACCEPTED ou REFUSED ;
- indiquer la complétude des données ;
- expliquer la prédiction avec SHAP.

Principe technique :
- FastAPI expose les endpoints ;
- Pydantic valide les données reçues ;
- Joblib charge le modèle et ses artefacts ;
- Pandas reconstruit les variables attendues ;
- LightGBM calcule la probabilité ;
- SHAP explique les facteurs influençant la prédiction.

L'API ne reconstruit pas les agrégations historiques ni les variables dérivées :
elles sont supposées avoir été calculées en amont.
"""

# ==================================================
# IMPORTS
# ==================================================
from pathlib import Path
from typing import Any
import logging
import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ==================================================
# LOGS
# ==================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ==================================================
# CHEMINS DES ARTEFACTS
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"

# ==================================================
# CHARGEMENT DU MODELE ET DES ARTEFACTS
# ==================================================
try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
except Exception as error:
    raise RuntimeError(f"Erreur lors du chargement des artefacts du modèle : {error}")

# ==================================================
# EXPLICABILITE SHAP
# ==================================================
try:
    explainer = shap.TreeExplainer(model)
except Exception as error:
    raise RuntimeError(f"Erreur lors de l'initialisation de SHAP : {error}")

# ==================================================
# CATEGORIES DE VARIABLES
# ==================================================
REQUIRED_FEATURES = [
    "PAYMENT_RATE",
    "EXT_SOURCE_MEAN",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_ANNUITY"
]

RECOMMENDED_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE"
]

OPTIONAL_FEATURES = [
    feature for feature in feature_names
    if feature not in REQUIRED_FEATURES and feature not in RECOMMENDED_FEATURES
]

# ==================================================
# IMPORTANCES DU MODELE
# ==================================================
if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(feature_names):
    feature_importance_map = dict(zip(feature_names, model.feature_importances_))
else:
    feature_importance_map = {feature: 1 for feature in feature_names}

total_feature_importance = sum(feature_importance_map.values()) or len(feature_names)

# ==================================================
# SCHEMA D'UN CLIENT
# ==================================================
class ClientData(BaseModel):
    """
    Données d'un client.

    Les 5 variables principales sont documentées explicitement.
    Les autres variables du modèle peuvent être envoyées directement
    et sont récupérées grâce à extra="allow".
    """
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "PAYMENT_RATE": 0.15,
                "EXT_SOURCE_MEAN": 0.55,
                "DAYS_BIRTH": -14000,
                "DAYS_EMPLOYED": -2000,
                "AMT_ANNUITY": 25000
            }
        }
    )

    PAYMENT_RATE: float = Field(
        ...,
        description="Ratio de paiement calculé en amont."
    )
    EXT_SOURCE_MEAN: float = Field(
        ...,
        description="Moyenne des scores externes calculée en amont."
    )
    DAYS_BIRTH: float = Field(
        ...,
        description="Âge encodé selon la convention Home Credit."
    )
    DAYS_EMPLOYED: float = Field(
        ...,
        description="Ancienneté professionnelle encodée."
    )
    AMT_ANNUITY: float = Field(
        ...,
        ge=0,
        description="Montant de l'annuité, positif ou nul."
    )

    @field_validator(
        "PAYMENT_RATE",
        "EXT_SOURCE_MEAN",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_ANNUITY",
        mode="before"
    )
    @classmethod
    def validate_numeric_required_features(cls, value: Any):
        if isinstance(value, bool):
            raise ValueError("La valeur doit être numérique, pas booléenne.")
        if not isinstance(value, (int, float)):
            raise ValueError("La valeur doit être numérique.")
        return value

# ==================================================
# SCHEMA DE REQUETE
# ==================================================
class PredictionRequest(BaseModel):
    requested_by: str = Field(
        default="anonymous",
        description="Identifiant déclaratif du demandeur."
    )
    clients: list[ClientData] = Field(
        ...,
        min_length=1,
        description="Clients à traiter."
    )

# ==================================================
# INITIALISATION FASTAPI
# ==================================================
app = FastAPI(
    title="Home Credit Scoring API",
    description="API de scoring crédit avec prédiction et explicabilité SHAP.",
    version="1.4.1"
)

# ==================================================
# ENDPOINT D'ACCUEIL
# ==================================================
@app.get("/")
def home():
    return {
        "status": "ok",
        "model": model.__class__.__name__,
        "n_features": len(feature_names),
        "n_required_features": len(REQUIRED_FEATURES),
        "n_recommended_features": len(RECOMMENDED_FEATURES),
        "business_threshold": round(threshold, 6),
        "shap_available": True
    }

# ==================================================
# ENDPOINT DE SANTE
# ==================================================
@app.get("/health")
def health():
    return {"status": "healthy"}

# ==================================================
# VALIDATION DES VARIABLES
# ==================================================
def validate_client_features(client_data: dict):
    # On ignore les éventuels champs techniques ajoutés par Pydantic
    # et on vérifie uniquement les variables réellement reçues.
    unknown_features = [key for key in client_data if key not in feature_names]
    if unknown_features:
        raise ValueError(f"Variables inconnues du modèle : {unknown_features}")

    for key, value in client_data.items():
        if isinstance(value, bool):
            raise ValueError(f"La variable '{key}' doit être numérique, pas booléenne.")
        if not isinstance(value, (int, float)):
            raise ValueError(f"La variable '{key}' doit être numérique.")

# ==================================================
# PREPARATION DES DONNEES
# ==================================================
def prepare_client_features(client_data: dict):
    known_features = {
        key: value for key, value in client_data.items()
        if key in feature_names
    }

    missing_recommended_features = [
        feature for feature in RECOMMENDED_FEATURES
        if feature not in known_features
    ]

    n_missing_optional_features = sum(
        feature not in known_features
        for feature in OPTIONAL_FEATURES
    )

    X = pd.DataFrame([known_features])
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    X = X.reindex(columns=feature_names, fill_value=0)

    return (
        X,
        known_features,
        missing_recommended_features,
        n_missing_optional_features
    )

# ==================================================
# COUVERTURE DES VARIABLES
# ==================================================
def compute_feature_coverage(known_features: dict):
    covered_importance = sum(
        feature_importance_map.get(feature, 0)
        for feature in known_features
    )

    feature_coverage_rate = (
        covered_importance / total_feature_importance * 100
    )

    if feature_coverage_rate >= 80:
        level = "HIGH"
    elif feature_coverage_rate >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"

    return round(feature_coverage_rate, 2), level

# ==================================================
# WARNING DE COMPLETUDE
# ==================================================
def build_warning(
    level: str,
    coverage: float,
    missing_recommended: list[str],
    n_missing_optional: int
):
    if level == "LOW":
        warning = (
            f"Complétude pondérée : {coverage:.2f}% (< 50%). "
            "La couverture des variables importantes est faible. "
            "La prédiction doit être interprétée avec prudence."
        )
    elif level == "MEDIUM":
        warning = (
            f"Complétude pondérée : {coverage:.2f}% (entre 50% et 80%). "
            "La couverture des variables importantes est partielle."
        )
    else:
        warning = (
            f"Complétude pondérée : {coverage:.2f}% (≥ 80%). "
            "La couverture des variables importantes est jugée satisfaisante."
        )

    if missing_recommended:
        warning += (
            f" Variables fortement recommandées absentes : "
            f"{missing_recommended}."
        )

    warning += (
        f" Nombre de variables optionnelles absentes : "
        f"{n_missing_optional}."
    )

    return warning

# ==================================================
# PREDICTION D'UN CLIENT
# ==================================================
def predict_single_client(client: ClientData, client_index: int):
    client_data = client.model_dump()
    validate_client_features(client_data)

    X, known_features, missing_recommended, n_missing_optional = (
        prepare_client_features(client_data)
    )

    coverage, level = compute_feature_coverage(known_features)

    default_probability = float(
        model.predict_proba(X)[0, 1]
    )

    decision = (
        "REFUSED"
        if default_probability >= threshold
        else "ACCEPTED"
    )

    warning = build_warning(
        level,
        coverage,
        missing_recommended,
        n_missing_optional
    )

    return {
        "client_index": client_index,
        "default_probability": round(default_probability, 6),
        "business_threshold": round(threshold, 6),
        "decision": decision,
        "feature_coverage_rate": coverage,
        "data_completeness_level": level,
        "missing_recommended_features": missing_recommended,
        "n_missing_optional_features": n_missing_optional,
        "warning": warning
    }

# ==================================================
# CALCUL SHAP
# ==================================================
def compute_shap_explanation(X: pd.DataFrame, top_n: int = 10):
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        values = shap_values[1]
        base_value = (
            float(expected_value[1])
            if isinstance(expected_value, (list, tuple))
            else float(expected_value)
        )
    else:
        values = shap_values

        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]

        if hasattr(expected_value, "__len__") and not isinstance(expected_value, str):
            try:
                base_value = float(expected_value[1])
            except (IndexError, TypeError):
                base_value = float(expected_value[0])
        else:
            base_value = float(expected_value)

    values = values[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
        "feature_value": X.iloc[0].values
    })

    shap_df["abs_shap_value"] = shap_df["shap_value"].abs()

    shap_df = (
        shap_df
        .sort_values("abs_shap_value", ascending=False)
        .head(top_n)
    )

    contributions = []

    for _, row in shap_df.iterrows():
        value = float(row["shap_value"])

        if value > 0:
            effect = "INCREASES_DEFAULT_RISK"
        elif value < 0:
            effect = "DECREASES_DEFAULT_RISK"
        else:
            effect = "NO_EFFECT"

        contributions.append({
            "feature": str(row["feature"]),
            "feature_value": float(row["feature_value"]),
            "shap_value": round(value, 6),
            "effect": effect
        })

    return {
        "base_value": round(base_value, 6),
        "contributions": contributions
    }

# ==================================================
# ENDPOINT /PREDICT
# ==================================================
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        predictions = [
            predict_single_client(client, index)
            for index, client in enumerate(request.clients)
        ]

        decisions = [
            prediction["decision"]
            for prediction in predictions
        ]

        qualities = [
            prediction["data_completeness_level"]
            for prediction in predictions
        ]

        logger.info(
            "requested_by=%s | endpoint=/predict | n_clients=%s | decisions=%s | qualities=%s",
            request.requested_by,
            len(request.clients),
            decisions,
            qualities
        )

        return {
            "requested_by": request.requested_by,
            "n_clients": len(request.clients),
            "predictions": predictions
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        logger.exception("Erreur inattendue sur /predict")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {error}"
        )

# ==================================================
# ENDPOINT /SHAP
# ==================================================
@app.post("/shap")
def shap_endpoint(request: PredictionRequest):
    try:
        explanations = []

        for index, client in enumerate(request.clients):
            client_data = client.model_dump()
            validate_client_features(client_data)

            X, known_features, missing_recommended, n_missing_optional = (
                prepare_client_features(client_data)
            )

            coverage, level = compute_feature_coverage(
                known_features
            )

            default_probability = float(
                model.predict_proba(X)[0, 1]
            )

            decision = (
                "REFUSED"
                if default_probability >= threshold
                else "ACCEPTED"
            )

            shap_result = compute_shap_explanation(
                X,
                top_n=10
            )

            warning = build_warning(
                level,
                coverage,
                missing_recommended,
                n_missing_optional
            )

            explanations.append({
                "client_index": index,
                "default_probability": round(default_probability, 6),
                "business_threshold": round(threshold, 6),
                "decision": decision,
                "feature_coverage_rate": coverage,
                "data_completeness_level": level,
                "base_value": shap_result["base_value"],
                "shap_contributions": shap_result["contributions"],
                "missing_recommended_features": missing_recommended,
                "n_missing_optional_features": n_missing_optional,
                "warning": warning
            })

        logger.info(
            "requested_by=%s | endpoint=/shap | n_clients=%s",
            request.requested_by,
            len(request.clients)
        )

        return {
            "requested_by": request.requested_by,
            "n_clients": len(request.clients),
            "explanations": explanations
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        logger.exception("Erreur inattendue sur /shap")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'explication SHAP : {error}"
        )