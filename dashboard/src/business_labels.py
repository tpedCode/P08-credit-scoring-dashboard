FEATURE_LABELS = {
    "PAYMENT_RATE": {
        "label": "Ratio annuité / crédit",
        "description": (
            "Mesure le poids de l'annuité par rapport au montant total du crédit. "
            "Un ratio plus élevé peut indiquer une charge de remboursement plus importante."
        ),
        "unit": "%"
    },
    "EXT_SOURCE_MEAN": {
        "label": "Score externe moyen",
        "description": (
            "Synthèse des scores externes disponibles. "
            "Un score plus faible peut être associé à un risque plus élevé."
        ),
        "unit": ""
    },
    "DAYS_BIRTH": {
        "label": "Âge du client",
        "description": (
            "Âge du client encodé en nombre de jours dans les données d'origine. "
            "La valeur est transformée en âge lisible dans le dashboard."
        ),
        "unit": "ans"
    },
    "PREV_CNT_PAYMENT_MEAN": {
        "label": "Durée moyenne des crédits précédents",
        "description": (
            "Nombre moyen d'échéances observé dans les précédentes demandes de crédit."
        ),
        "unit": ""
    },
    "EXT_SOURCE_3": {
        "label": "Score externe 3",
        "description": (
            "Score externe utilisé par le modèle comme indicateur complémentaire de risque."
        ),
        "unit": ""
    },
    "PREV_APP_CREDIT_PERC_MEAN": {
        "label": "Ratio demandé / accordé sur crédits précédents",
        "description": (
            "Compare les montants demandés et accordés lors de précédentes demandes de crédit."
        ),
        "unit": ""
    },
    "DAYS_EMPLOYED": {
        "label": "Ancienneté professionnelle",
        "description": (
            "Ancienneté professionnelle du client, encodée en jours dans les données d'origine."
        ),
        "unit": "ans"
    },
    "DAYS_ID_PUBLISH": {
        "label": "Ancienneté du document d'identité",
        "description": (
            "Ancienneté de la dernière modification ou publication du document d'identité."
        ),
        "unit": "jours"
    },
    "DAYS_LAST_PHONE_CHANGE": {
        "label": "Ancienneté du dernier changement de téléphone",
        "description": (
            "Nombre de jours depuis le dernier changement de numéro de téléphone déclaré."
        ),
        "unit": "jours"
    },
    "EXT_SOURCE_2": {
        "label": "Score externe 2",
        "description": (
            "Score externe utilisé par le modèle comme indicateur complémentaire de solvabilité."
        ),
        "unit": ""
    },
    "DAYS_REGISTRATION": {
        "label": "Ancienneté de l'inscription",
        "description": (
            "Ancienneté de l'enregistrement administratif du client dans les données."
        ),
        "unit": "jours"
    },
    "EXT_SOURCE_1": {
        "label": "Score externe 1",
        "description": (
            "Score externe utilisé par le modèle comme indicateur complémentaire de solvabilité."
        ),
        "unit": ""
    },
    "INSTAL_DPD_MEAN": {
        "label": "Retard moyen de paiement historique",
        "description": (
            "Retard moyen observé dans les paiements historiques du client. "
            "Un retard plus élevé peut être associé à un risque plus important."
        ),
        "unit": "jours"
    },
    "BURO_DAYS_CREDIT_MEAN": {
        "label": "Ancienneté moyenne des crédits externes",
        "description": (
            "Ancienneté moyenne des crédits observés dans l'historique bureau."
        ),
        "unit": "jours"
    },
    "AMT_ANNUITY": {
        "label": "Montant de l'annuité",
        "description": (
            "Montant de l'annuité liée au crédit demandé."
        ),
        "unit": "€"
    },
    "BURO_AMT_CREDIT_SUM_MEAN": {
        "label": "Montant moyen des crédits externes",
        "description": (
            "Montant moyen des crédits observés dans l'historique externe du client."
        ),
        "unit": "€"
    },
    "ANNUITY_INCOME_RATIO": {
        "label": "Ratio annuité / revenu",
        "description": (
            "Mesure le poids de l'annuité par rapport au revenu déclaré."
        ),
        "unit": "%"
    },
    "CREDIT_INCOME_RATIO": {
        "label": "Ratio crédit / revenu",
        "description": (
            "Mesure le poids du crédit demandé par rapport au revenu déclaré."
        ),
        "unit": ""
    },
    "PREV_AMT_APPLICATION_MEAN": {
        "label": "Montant moyen demandé précédemment",
        "description": (
            "Montant moyen demandé lors de précédentes demandes de crédit."
        ),
        "unit": "€"
    },
    "REGION_POPULATION_RELATIVE": {
        "label": "Densité relative de la région",
        "description": (
            "Indicateur relatif à la population de la région du client."
        ),
        "unit": ""
    }
}


def get_feature_label(feature_name):
    return FEATURE_LABELS.get(feature_name, {}).get("label", feature_name)


def get_feature_description(feature_name):
    return FEATURE_LABELS.get(feature_name, {}).get(
        "description",
        "Aucune description métier disponible pour cette variable."
    )


def get_feature_unit(feature_name):
    return FEATURE_LABELS.get(feature_name, {}).get("unit", "")
