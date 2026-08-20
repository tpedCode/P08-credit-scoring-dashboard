# ==================================================
# API FASTAPI - HOME CREDIT SCORING
# ==================================================

"""
API FASTAPI - HOME CREDIT SCORING

Objectif :
- exposer le modèle de scoring crédit sous forme d'API ;
- recevoir les données préparées d'un ou plusieurs clients ;
- valider les données nécessaires à la prédiction ;
- calculer une probabilité de défaut ;
- appliquer le seuil métier optimisé lors de l'entraînement ;
- retourner une décision exploitable par les équipes métier.

Postulat de fonctionnement :
- l'API intervient après le pipeline de Feature Engineering ;
- les variables dérivées et les agrégations historiques sont calculées en amont ;
- les données transmises à l'API sont donc déjà préparées selon le Feature Engineering
  défini lors de l'entraînement ;
- l'API ne reconstruit pas les variables dérivées ni les agrégations historiques ;
- les sources nécessaires au Feature Engineering complet sont traitées en amont,
  notamment Application, Bureau, Previous Application et Installments Payments ;
- l'API réalise uniquement la validation des données, leur préparation finale
  et la prédiction ;
- les 246 variables finales attendues par le modèle sont définies par feature_names.

Choix métiers :
- utiliser une probabilité de défaut pour évaluer le risque de chaque client ;
- appliquer le seuil métier optimisé lors de l'entraînement plutôt que le seuil
  standard de 0,5 ;
- retourner une décision ACCEPTED ou REFUSED ;
- distinguer la prédiction du modèle de la couverture des données fournies ;
- informer l'utilisateur lorsque certaines variables recommandées ou importantes
  ne sont pas fournies ;
- limiter la journalisation des données afin de réduire l'exposition des données clients.

Choix techniques :
- FastAPI expose les endpoints de l'API ;
- Pydantic valide le format et les types des données reçues ;
- Joblib charge le modèle et les artefacts sauvegardés ;
- Pandas reconstruit le DataFrame attendu par le modèle ;
- les valeurs infinies sont remplacées par NaN ;
- les valeurs NaN sont remplacées par 0, conformément au traitement réalisé
  pendant l'entraînement ;
- les colonnes sont alignées selon feature_names ;
- les 246 variables finales sont transmises au modèle LightGBM ;
- aucune standardisation n'est appliquée car le modèle final a été entraîné
  avec scale=False ;
- le modèle utilise predict_proba pour produire la probabilité de défaut ;
- le seuil métier optimisé est appliqué à cette probabilité ;
- les importances globales du modèle sont utilisées pour estimer la couverture
  pondérée des variables fournies.

Entrée / Sortie :
- entrée : requête JSON contenant les données préparées d'un ou plusieurs clients ;
- sortie : réponse JSON contenant, pour chaque client, la probabilité de défaut,
  le seuil métier, la décision, le niveau de complétude des données et les warnings éventuels.
"""

from pathlib import Path
from typing import Any

import logging
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==================================================
# CONFIGURATION DES LOGS
# ==================================================

# Les logs permettent de suivre l'activité minimale de l'API.
# Les données clients complètes ne sont volontairement pas enregistrées.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==================================================
# CHEMINS DES ARTEFACTS
# ==================================================

# Le fichier actuel est situé dans :
# P07/api/api_home_credit_scoring.py
#
# parents[1] permet de remonter à la racine du projet :
# P07/

BASE_DIR = Path(__file__).resolve().parents[1]

# Artefacts produits lors de la phase de modélisation.
# Ils permettent à l'API de charger directement le modèle et ses paramètres.

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"


# ==================================================
# CHARGEMENT DES ARTEFACTS DU MODELE
# ==================================================

try:
    # Modèle final retenu lors de la phase de modélisation.
    model = joblib.load(MODEL_PATH)

    # Liste exacte des variables utilisées lors de l'entraînement.
    feature_names = joblib.load(FEATURES_PATH)

    # Seuil métier optimisé lors de l'entraînement.
    threshold = float(joblib.load(THRESHOLD_PATH))

except Exception as error:
    raise RuntimeError(
        f"Erreur lors du chargement des artefacts du modèle : {error}"
    )


# ==================================================
# CATEGORIES DE VARIABLES
# ==================================================

# Catégorie 1 : variables obligatoires.
#
# Ces variables ont été sélectionnées à partir de l'importance globale
# du modèle et représentent un compromis entre :
# - importance prédictive ;
# - disponibilité métier ;
# - facilité d'obtention lors d'un appel API.
#
# Elles sont déclarées comme champs obligatoires dans ClientData.

REQUIRED_FEATURES = [
    "PAYMENT_RATE",
    "EXT_SOURCE_MEAN",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_ANNUITY"
]


# Catégorie 2 : variables fortement recommandées.
#
# Leur présence augmente la couverture des variables importantes du modèle.
# Leur absence ne bloque pas la prédiction mais est signalée dans la réponse.

RECOMMENDED_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE"
]


# Catégorie 3 : variables optionnelles.
#
# Toutes les autres variables attendues par le modèle sont considérées comme
# optionnelles. Elles peuvent notamment provenir des agrégations réalisées
# à partir de l'historique de crédit et des remboursements.
#
# Les variables absentes seront ajoutées lors de la préparation du DataFrame
# et complétées par 0, conformément au traitement appliqué pendant l'entraînement.

OPTIONAL_FEATURES = [
    feature
    for feature in feature_names
    if feature not in REQUIRED_FEATURES
    and feature not in RECOMMENDED_FEATURES
]


# ==================================================
# IMPORTANCES GLOBALES DU MODELE
# ==================================================

# Les importances globales permettent de calculer une couverture pondérée.
# L'objectif est de donner davantage de poids aux variables les plus importantes
# dans le modèle plutôt que de simplement compter le nombre de variables fournies.

if (
    hasattr(model, "feature_importances_")
    and len(model.feature_importances_) == len(feature_names)
):
    feature_importance_map = dict(
        zip(
            feature_names,
            model.feature_importances_
        )
    )

else:
    # Si le modèle ne fournit pas d'importances, chaque variable reçoit
    # le même poids afin de conserver un calcul de couverture possible.
    feature_importance_map = {
        feature: 1
        for feature in feature_names
    }

total_feature_importance = sum(feature_importance_map.values())

if total_feature_importance == 0:
    total_feature_importance = len(feature_names)


# ==================================================
# SCHEMA D'UN CLIENT
# ==================================================

class ClientData(BaseModel):
    """
    Objectif :
    - définir le format attendu pour les données d'un client ;
    - garantir la présence des variables obligatoires ;
    - contrôler les types des variables obligatoires avant prédiction.

    Choix métiers :
    - rendre obligatoires les variables importantes et facilement exploitables ;
    - permettre la transmission de variables recommandées et optionnelles lorsqu'elles sont disponibles ;
    - considérer les variables dérivées comme PAYMENT_RATE et EXT_SOURCE_MEAN
      déjà calculées en amont du pipeline d'API.

    Choix techniques :
    - Pydantic valide automatiquement les champs obligatoires ;
    - extra="allow" autorise la transmission des autres variables du modèle ;
    - les variables supplémentaires sont ensuite contrôlées par rapport à feature_names ;
    - les booléens sont refusés car ils ne constituent pas des valeurs numériques valides
      pour les variables du modèle.

    Entrée / Sortie :
    - entrée : données JSON d'un client ;
    - sortie : objet ClientData validé ou erreur 422 en cas de donnée invalide.
    """

    model_config = ConfigDict(extra="allow")

    PAYMENT_RATE: float = Field(
        ...,
        description="Ratio de paiement déjà calculé pendant le Feature Engineering."
    )

    EXT_SOURCE_MEAN: float = Field(
        ...,
        description="Moyenne des scores externes déjà calculée pendant le Feature Engineering."
    )

    DAYS_BIRTH: float = Field(
        ...,
        description="Âge du client encodé selon la convention du dataset Home Credit."
    )

    DAYS_EMPLOYED: float = Field(
        ...,
        description="Ancienneté professionnelle encodée selon la convention du dataset Home Credit."
    )

    AMT_ANNUITY: float = Field(
        ...,
        ge=0,
        description="Montant de l'annuité. Doit être positif ou nul."
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
        """
        Objectif :
        - vérifier que les variables obligatoires sont numériques.

        Choix métiers :
        - empêcher l'utilisation d'une valeur textuelle ou booléenne pour le scoring ;
        - garantir un minimum de cohérence avant l'appel au modèle.

        Choix techniques :
        - effectuer le contrôle avant la conversion Pydantic ;
        - refuser les booléens et les chaînes de caractères ;
        - autoriser uniquement les valeurs numériques.

        Entrée / Sortie :
        - entrée : valeur reçue pour une variable obligatoire ;
        - sortie : valeur validée ou erreur de validation.
        """

        if isinstance(value, bool):
            raise ValueError(
                "La valeur doit être numérique, pas booléenne."
            )

        if not isinstance(value, (int, float)):
            raise ValueError(
                "La valeur doit être numérique."
            )

        return value


# ==================================================
# SCHEMA DE REQUETE
# ==================================================

class PredictionRequest(BaseModel):
    """
    Objectif :
    - définir le contrat d'entrée de l'endpoint de prédiction ;
    - permettre de scorer un ou plusieurs clients avec une seule requête.

    Choix métiers :
    - utiliser un format commun pour les prédictions unitaires et multiples ;
    - identifier de manière déclarative le demandeur de la prédiction ;
    - faciliter le suivi des appels dans les logs ;
    - préciser que requested_by ne constitue pas un mécanisme d'authentification ;
    - considérer que les données transmises sont déjà préparées en amont.

    Choix techniques :
    - la requête contient toujours une liste de clients ;
    - min_length=1 empêche les requêtes sans client ;
    - requested_by est optionnel et vaut "anonymous" par défaut.

    Entrée / Sortie :
    - entrée : requête JSON contenant requested_by et clients ;
    - sortie : objet PredictionRequest validé ou erreur 422 en cas de format invalide.
    """

    requested_by: str = Field(
        default="anonymous",
        description=(
            "Identifiant déclaratif de la personne ou du système ayant demandé la prédiction."
        )
    )

    clients: list[ClientData] = Field(
        ...,
        min_length=1,
        description="Liste des clients à scorer."
    )


# ==================================================
# INITIALISATION DE L'API
# ==================================================

app = FastAPI(
    title="Home Credit Scoring API",
    description=(
        "API de scoring crédit permettant de calculer la probabilité de défaut d'un ou "
        "plusieurs clients et de retourner une décision métier basée sur un seuil optimisé."
    ),
    version="1.2.0"
)


# ==================================================
# ENDPOINT D'ACCUEIL
# ==================================================

@app.get("/")
def home():
    """
    Objectif :
    - exposer les principales informations relatives à l'API et au modèle chargé ;
    - permettre de vérifier rapidement la configuration utilisée pour le scoring.

    Choix métiers :
    - rendre visible le seuil métier utilisé pour les décisions ;
    - fournir des informations générales sur les variables utilisées par le modèle.

    Choix techniques :
    - utiliser un endpoint GET simple ;
    - ne nécessiter aucune donnée client ;
    - retourner le nom du modèle, le nombre de variables et le seuil métier.

    Entrée / Sortie :
    - entrée : aucune ;
    - sortie : dictionnaire contenant les informations générales de l'API et du modèle.
    """

    return {
        "status": "ok",
        "model": model.__class__.__name__,
        "n_features": len(feature_names),
        "n_required_features": len(REQUIRED_FEATURES),
        "n_recommended_features": len(RECOMMENDED_FEATURES),
        "business_threshold": round(threshold, 6)
    }


# ==================================================
# ENDPOINT DE SANTE
# ==================================================

@app.get("/health")
def health():
    """
    Objectif :
    - vérifier que l'API est disponible et répond correctement.

    Choix métiers :
    - aucun traitement métier ni aucune prédiction ne sont réalisés ;
    - fournir uniquement une information de disponibilité du service.

    Choix techniques :
    - utiliser un endpoint GET indépendant des données clients ;
    - fournir une réponse minimale adaptée aux contrôles de disponibilité.

    Entrée / Sortie :
    - entrée : aucune ;
    - sortie : dictionnaire contenant le statut de santé de l'API.
    """

    return {
        "status": "healthy"
    }


# ==================================================
# VALIDATION DES VARIABLES
# ==================================================

def validate_client_features(client_data: dict):
    """
    Objectif :
    - vérifier que les variables fournies appartiennent au périmètre du modèle ;
    - vérifier que les variables transmises sont numériques avant la prédiction.

    Choix métiers :
    - empêcher l'utilisation de variables inconnues du modèle ;
    - éviter qu'une donnée non numérique soit utilisée pour le scoring ;
    - garantir que les variables transmises appartiennent au périmètre utilisé
      lors de l'entraînement.

    Choix techniques :
    - comparer les variables reçues à feature_names ;
    - rejeter les variables inconnues ;
    - refuser les valeurs booléennes et non numériques ;
    - effectuer ces contrôles avant l'appel à predict_proba.

    Entrée / Sortie :
    - entrée : dictionnaire contenant les données d'un client ;
    - sortie : aucune sortie directe ;
    - exception : ValueError si une variable inconnue ou une valeur non numérique
      est détectée.
    """

    unknown_features = [
        key
        for key in client_data.keys()
        if key not in feature_names
    ]

    if unknown_features:
        raise ValueError(
            f"Variables inconnues du modèle : {unknown_features}"
        )

    for key, value in client_data.items():

        if isinstance(value, bool):
            raise ValueError(
                f"La variable '{key}' doit être numérique, pas booléenne."
            )

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"La variable '{key}' doit être numérique."
            )


# ==================================================
# PREPARATION DES DONNEES POUR LE MODELE
# ==================================================

def prepare_client_features(client_data: dict):
    """
    Objectif :
    - préparer les données d'un client dans le format attendu par le modèle ;
    - appliquer le traitement des valeurs infinies et manquantes utilisé pendant l'entraînement ;
    - aligner les variables selon l'ordre attendu par le modèle.

    Postulat de fonctionnement :
    - les variables dérivées et les agrégations historiques ont déjà été calculées
      avant l'appel à l'API ;
    - la fonction ne réalise donc pas le Feature Engineering complet ;
    - les variables absentes sont ajoutées au DataFrame et complétées par 0.

    Choix métiers :
    - permettre la prédiction lorsque des variables recommandées ou optionnelles
      ne sont pas disponibles ;
    - signaler les variables recommandées absentes ;
    - informer du nombre de variables optionnelles absentes.

    Choix techniques :
    - conserver uniquement les variables connues du modèle ;
    - remplacer les valeurs infinies par NaN puis les NaN par 0 ;
    - ajouter les variables absentes avec une valeur de 0 ;
    - reconstruire un DataFrame contenant les 246 variables attendues ;
    - utiliser feature_names pour garantir l'ordre des colonnes ;
    - ne réaliser aucune standardisation.

    Entrée / Sortie :
    - entrée : dictionnaire contenant les données d'un client ;
    - sortie :
        - X : DataFrame contenant les 246 variables attendues par le modèle ;
        - known_features : variables du modèle effectivement fournies ;
        - missing_recommended_features : variables recommandées absentes ;
        - n_missing_optional_features : nombre de variables optionnelles absentes.
    """

    known_features = {
        key: value
        for key, value in client_data.items()
        if key in feature_names
    }

    missing_recommended_features = [
        feature
        for feature in RECOMMENDED_FEATURES
        if feature not in known_features
    ]

    missing_optional_features = [
        feature
        for feature in OPTIONAL_FEATURES
        if feature not in known_features
    ]

    n_missing_optional_features = len(missing_optional_features)

    X = pd.DataFrame([known_features])

    # Même traitement que pendant l'entraînement :
    # infini → NaN → 0
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(0)

    # Les variables absentes sont ajoutées avec 0 afin de reconstruire
    # exactement l'espace de variables attendu par le modèle.
    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    return (
        X,
        known_features,
        missing_recommended_features,
        n_missing_optional_features
    )


# ==================================================
# QUALITE DE COMPLETUDE PONDEREE
# ==================================================

def compute_feature_coverage(known_features: dict):
    """
    Objectif :
    - mesurer la couverture des variables du modèle effectivement fournies ;
    - pondérer cette couverture selon l'importance globale des variables.

    Choix métiers :
    - distinguer la couverture des données de la performance ou de la fiabilité
      de la prédiction ;
    - donner davantage de poids aux variables importantes pour le modèle ;
    - fournir un indicateur permettant d'interpréter la complétude des données.

    Choix techniques :
    - utiliser feature_importances_ du modèle LightGBM ;
    - calculer la somme des importances des variables fournies ;
    - rapporter cette valeur à l'importance totale du modèle ;
    - classer la couverture selon trois niveaux :
        - LOW : < 50 % ;
        - MEDIUM : 50 % à moins de 80 % ;
        - HIGH : ≥ 80 %.

    Entrée / Sortie :
    - entrée : dictionnaire des variables effectivement fournies ;
    - sortie :
        - feature_coverage_rate : pourcentage d'importance du modèle couverte ;
        - data_completeness_level : niveau de complétude LOW, MEDIUM ou HIGH.
    """

    covered_importance = sum(
        feature_importance_map.get(feature, 0)
        for feature in known_features
    )

    feature_coverage_rate = (
        covered_importance / total_feature_importance
    ) * 100

    if feature_coverage_rate >= 80:
        data_completeness_level = "HIGH"

    elif feature_coverage_rate >= 50:
        data_completeness_level = "MEDIUM"

    else:
        data_completeness_level = "LOW"

    return round(feature_coverage_rate, 2), data_completeness_level


# ==================================================
# WARNING METIER
# ==================================================

def build_warning(
    data_completeness_level: str,
    feature_coverage_rate: float,
    missing_recommended_features: list[str],
    n_missing_optional_features: int
):
    """
    Objectif :
    - générer un message permettant d'interpréter la couverture des données utilisées
      pour la prédiction.

    Choix métiers :
    - informer l'utilisateur lorsque la couverture des variables importantes est faible
      ou partielle ;
    - distinguer les variables recommandées absentes des variables optionnelles absentes ;
    - fournir un indicateur de complétude sans le présenter comme une mesure de performance
      ou de fiabilité du modèle.

    Choix techniques :
    - utiliser le niveau de complétude et le taux de couverture pondérée calculés précédemment ;
    - appliquer les seuils suivants :
        - LOW : < 50 % ;
        - MEDIUM : 50 % à moins de 80 % ;
        - HIGH : ≥ 80 % ;
    - ajouter la liste des variables recommandées absentes ;
    - indiquer le nombre de variables optionnelles absentes.

    Entrée / Sortie :
    - entrée :
        - data_completeness_level : niveau de complétude ;
        - feature_coverage_rate : taux de couverture pondérée ;
        - missing_recommended_features : variables recommandées absentes ;
        - n_missing_optional_features : nombre de variables optionnelles absentes.
    - sortie : chaîne de caractères contenant le warning destiné à l'utilisateur.
    """

    if data_completeness_level == "LOW":

        warning = (
            f"Complétude pondérée : {feature_coverage_rate:.2f}% "
            "(< 50%). "
            "La prédiction repose sur une faible couverture des variables "
            "importantes du modèle. Les données fournies sont incomplètes "
            "et le résultat doit être interprété avec prudence."
        )

    elif data_completeness_level == "MEDIUM":

        warning = (
            f"Complétude pondérée : {feature_coverage_rate:.2f}% "
            "(entre 50% et 80%). "
            "La couverture des variables importantes du modèle est partielle. "
            "Certaines variables utilisées lors de l'entraînement ne sont pas fournies."
        )

    else:

        warning = (
            f"Complétude pondérée : {feature_coverage_rate:.2f}% "
            "(≥ 80%). "
            "La couverture des variables importantes du modèle est jugée satisfaisante."
        )

    if missing_recommended_features:
        warning += (
            f" Variables fortement recommandées absentes : "
            f"{missing_recommended_features}."
        )

    warning += (
        f" Nombre de variables optionnelles absentes : "
        f"{n_missing_optional_features}."
    )

    return warning


# ==================================================
# PREDICTION POUR UN CLIENT
# ==================================================

def predict_single_client(client: ClientData, client_index: int):
    """
    Objectif :
    - calculer la probabilité de défaut d'un client ;
    - appliquer le seuil métier optimisé ;
    - produire une décision de scoring ;
    - fournir des informations sur la couverture des données.

    Postulat de fonctionnement :
    - les données reçues sont issues du Feature Engineering réalisé en amont ;
    - les variables dérivées et les agrégations historiques ne sont pas recalculées
      dans cette fonction ;
    - le modèle reçoit un DataFrame contenant les 246 variables attendues,
      certaines pouvant avoir été complétées par 0.

    Choix métiers :
    - utiliser la probabilité de défaut produite par le modèle ;
    - appliquer le seuil métier optimisé lors de l'entraînement ;
    - retourner ACCEPTED lorsque la probabilité est inférieure au seuil ;
    - retourner REFUSED lorsque la probabilité est supérieure ou égale au seuil ;
    - informer l'utilisateur du niveau de couverture des variables fournies.

    Choix techniques :
    - convertir l'objet Pydantic en dictionnaire ;
    - valider les variables reçues ;
    - préparer les 246 variables attendues par le modèle ;
    - calculer la couverture pondérée des variables fournies ;
    - utiliser predict_proba du modèle LightGBM ;
    - appliquer le seuil métier sauvegardé lors de l'entraînement ;
    - construire le warning associé à la complétude des données.

    Entrée / Sortie :
    - entrée :
        - client : données validées d'un client ;
        - client_index : index du client dans la requête.
    - sortie : dictionnaire contenant la probabilité de défaut, le seuil métier,
      la décision, la couverture des variables, le niveau de complétude,
      les variables recommandées absentes et le warning éventuel.
    """

    client_data = client.model_dump()

    validate_client_features(client_data)

    (
        X,
        known_features,
        missing_recommended_features,
        n_missing_optional_features
    ) = prepare_client_features(client_data)

    feature_coverage_rate, data_completeness_level = compute_feature_coverage(
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

    warning = build_warning(
        data_completeness_level=data_completeness_level,
        feature_coverage_rate=feature_coverage_rate,
        missing_recommended_features=missing_recommended_features,
        n_missing_optional_features=n_missing_optional_features
    )

    return {
        "client_index": client_index,
        "default_probability": round(default_probability, 6),
        "business_threshold": round(threshold, 6),
        "decision": decision,
        "feature_coverage_rate": feature_coverage_rate,
        "data_completeness_level": data_completeness_level,
        "missing_recommended_features": missing_recommended_features,
        "n_missing_optional_features": n_missing_optional_features,
        "warning": warning
    }


# ==================================================
# ENDPOINT DE PREDICTION
# ==================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Objectif :
    - recevoir une requête contenant un ou plusieurs clients ;
    - orchestrer la validation et le scoring de chaque client ;
    - retourner une réponse structurée contenant les résultats de prédiction.

    Choix métiers :
    - permettre le scoring individuel ou par lot ;
    - appliquer le même modèle et le même seuil métier à l'ensemble des clients ;
    - retourner une décision et des informations de complétude pour chaque client ;
    - limiter les informations enregistrées dans les logs afin de ne pas stocker
      les données clients complètes.

    Choix techniques :
    - utiliser Pydantic pour valider la structure de la requête ;
    - traiter les clients individuellement via predict_single_client() ;
    - regrouper les résultats dans une réponse JSON ;
    - enregistrer uniquement des informations minimales dans les logs ;
    - retourner les erreurs de validation avec un statut HTTP 422 ;
    - retourner les erreurs inattendues avec un statut HTTP 500.

    Entrée / Sortie :
    - entrée :
        - request : objet PredictionRequest contenant l'identifiant déclaratif
          du demandeur et la liste des clients.
    - sortie :
        - requested_by : identifiant déclaratif du demandeur ;
        - n_clients : nombre de clients traités ;
        - predictions : liste des résultats de prédiction pour chaque client.
    """

    try:
        predictions = []

        for index, client in enumerate(request.clients):
            prediction = predict_single_client(
                client=client,
                client_index=index
            )
            predictions.append(prediction)

        decisions = [
            prediction["decision"]
            for prediction in predictions
        ]

        qualities = [
            prediction["data_completeness_level"]
            for prediction in predictions
        ]

        # Les logs contiennent uniquement des informations générales
        # afin de limiter l'exposition des données clients.
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
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {error}"
        )