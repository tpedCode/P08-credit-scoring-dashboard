import os
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.business_labels import get_feature_label


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Prêt à dépenser — Decision insights",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# URL DES API
# --------------------------------------------------

SCORING_API_URL = os.getenv(
    "SCORING_API_URL",
    "http://127.0.0.1:8000"
)

DATA_API_URL = os.getenv(
    "DATA_API_URL",
    "http://127.0.0.1:8001"
)


# --------------------------------------------------
# COULEURS
# --------------------------------------------------

ORANGE = "#FFB24A"
BLUE = "#06006C"
LIGHT_ORANGE = "#FFF1DC"
LIGHT_BLUE = "#E9E8FF"
LIGHT_GREEN = "#E8F5E9"
LIGHT_RED = "#FDECEC"


# ==================================================
# CSS
# ==================================================

st.markdown(
    f"""
    <style>

        [data-testid="stSidebar"] {{
            display: none;
        }}

        [data-testid="collapsedControl"] {{
            display: none;
        }}

        [data-testid="stTabs"] [role="tablist"] {{
            gap: 8px;
        }}

        [data-testid="stTabs"] button {{
            color: {BLUE};
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            padding: 10px 18px;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] {{
            background-color: {BLUE};
            color: white;
        }}

        .stButton > button {{
            background-color: {ORANGE};
            color: {BLUE};
            border: 1px solid {ORANGE};
            font-weight: 700;
            border-radius: 8px;
        }}

        .stButton > button:hover {{
            background-color: {BLUE};
            color: white;
            border-color: {BLUE};
        }}

        h1, h2, h3 {{
            color: {BLUE};
        }}

        .decision-accepted {{
            background-color: {LIGHT_GREEN};
            border-left: 6px solid #2E7D32;
            padding: 1rem;
            border-radius: 8px;
        }}

        .decision-refused {{
            background-color: {LIGHT_RED};
            border-left: 6px solid #C62828;
            padding: 1rem;
            border-radius: 8px;
        }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# VARIABLES METIER
# ==================================================

FEATURES_TO_COMPARE = {

    "AMT_INCOME_TOTAL": {
        "label": "Revenu total",
        "format": "amount",
        "description": (
            "Revenu total déclaré par le client."
        )
    },

    "AMT_CREDIT": {
        "label": "Montant du crédit",
        "format": "amount",
        "description": (
            "Montant du crédit demandé."
        )
    },

    "AMT_ANNUITY": {
        "label": "Annuité",
        "format": "amount",
        "description": (
            "Montant de l'annuité liée au crédit."
        )
    },

    "PAYMENT_RATE": {
        "label": "Ratio annuité / crédit",
        "format": "percent",
        "description": (
            "Part de l'annuité par rapport "
            "au montant total du crédit."
        )
    },

    "CREDIT_INCOME_RATIO": {
        "label": "Ratio crédit / revenu",
        "format": "number",
        "description": (
            "Poids du montant du crédit "
            "par rapport au revenu total."
        )
    },

    "ANNUITY_INCOME_RATIO": {
        "label": "Ratio annuité / revenu",
        "format": "percent",
        "description": (
            "Poids de l'annuité "
            "par rapport au revenu total."
        )
    },

    "EXT_SOURCE_MEAN": {
        "label": "Score externe moyen",
        "format": "number",
        "description": (
            "Score externe synthétique utilisé "
            "comme indicateur de solvabilité."
        )
    },

    "INSTAL_DPD_MEAN": {
        "label": "Retard moyen de paiement",
        "format": "number",
        "description": (
            "Retard moyen observé "
            "dans les paiements historiques."
        )
    },

    "INSTAL_DPD_MAX": {
        "label": "Retard maximum de paiement",
        "format": "number",
        "description": (
            "Retard maximum observé "
            "dans les paiements historiques."
        )
    },

    "PREV_CNT_PAYMENT_MEAN": {
        "label": "Durée moyenne des crédits précédents",
        "format": "number",
        "description": (
            "Nombre moyen d'échéances "
            "des précédentes demandes de crédit."
        )
    },

    "BURO_AMT_CREDIT_SUM_MEAN": {
        "label": "Montant moyen des crédits externes",
        "format": "amount",
        "description": (
            "Montant moyen des crédits observés "
            "dans l'historique bureau."
        )
    }
}


# ==================================================
# FORMATAGE
# ==================================================

def format_amount(value):

    if pd.isna(value):
        return "Non disponible"

    return f"{value:,.0f} €".replace(",", " ")


def format_percent(value):

    if pd.isna(value):
        return "Non disponible"

    return f"{value:.2%}"


def format_number(value, decimals=2):

    if pd.isna(value):
        return "Non disponible"

    return f"{value:.{decimals}f}"


def format_years_from_days(value):

    if pd.isna(value) or value == 0:
        return "Non disponible"

    return f"{abs(value) / 365:.1f} ans"


def format_by_type(value, value_type):

    if value_type == "amount":
        return format_amount(value)

    if value_type == "percent":
        return format_percent(value)

    return format_number(value)


def translate_decision(decision):

    return {
        "REFUSED": "Crédit refusé",
        "ACCEPTED": "Crédit accepté"
    }.get(
        decision,
        decision
    )


def translate_completeness(level):

    return {
        "HIGH": "Élevée",
        "MEDIUM": "Moyenne",
        "LOW": "Faible"
    }.get(
        level,
        level
    )


def format_shap_value(value):

    return f"{value:+.4f}"


# ==================================================
# SESSION STATE
# ==================================================

def init_session_state():

    defaults = {
        "selected_client_id": None,
        "client_profile": None,
        "prediction": None,
        "local_explanation": None,
        "risk_contributions": None,
        "protective_contributions": None
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def reset_client_results_if_needed(client_id):

    if st.session_state.selected_client_id != client_id:

        st.session_state.selected_client_id = client_id

        st.session_state.client_profile = None
        st.session_state.prediction = None
        st.session_state.local_explanation = None
        st.session_state.risk_contributions = None
        st.session_state.protective_contributions = None


# ==================================================
# API — DONNEES
# ==================================================

@st.cache_data(ttl=300)
def get_client_profile(client_id):

    response = requests.get(
        f"{DATA_API_URL}/clients/{client_id}",
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "client" in data:
        return data["client"]

    return data


@st.cache_data(ttl=300)
def get_population(
    features,
    sample_size=5000
):

    response = requests.get(
        f"{DATA_API_URL}/population",
        params={
            "features": ",".join(features),
            "limit": sample_size
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "clients" not in data:

        raise ValueError(
            "La réponse de l'API Données "
            "ne contient pas la clé 'clients'."
        )

    population = pd.DataFrame(
        data["clients"]
    )

    missing_features = [
        feature
        for feature in features
        if feature not in population.columns
    ]

    if missing_features:

        raise ValueError(
            "Variables absentes de la population : "
            f"{missing_features}"
        )

    return population[features]


@st.cache_data(ttl=300)
def get_global_feature_importance():

    response = requests.get(
        f"{DATA_API_URL}/feature-importance",
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "features" not in data:

        raise ValueError(
            "La réponse de l'API Données "
            "ne contient pas la clé 'features'."
        )

    return pd.DataFrame(
        data["features"]
    )


# ==================================================
# API — SCORING
# ==================================================

def call_prediction_api(profile):

    client_features = {
        key: value
        for key, value in profile.items()
        if key != "SK_ID_CURR"
    }

    payload = {
        "requested_by": "streamlit_dashboard",
        "clients": [client_features]
    }

    response = requests.post(
        f"{SCORING_API_URL}/predict",
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    prediction = data["predictions"][0]

    return prediction


# ==================================================
# EXTRACTION SHAP
# ==================================================

def extract_local_explanation(prediction):

    shap_data = (
        prediction.get("local_explanation")
        or prediction.get("shap")
        or prediction.get("shap_values")
    )

    if shap_data is None:
        return None

    if isinstance(shap_data, dict):

        shap_data = (
            shap_data.get("features")
            or shap_data.get("data")
            or shap_data
        )

    dataframe = pd.DataFrame(shap_data)

    if dataframe.empty:
        return None

    return dataframe


def split_shap_impacts(data):

    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = data.copy()

    if "shap_value" not in data.columns:
        return pd.DataFrame(), pd.DataFrame()

    if "feature_label" not in data.columns:

        if "feature" in data.columns:

            data["feature_label"] = (
                data["feature"]
                .apply(get_feature_label)
            )

        else:

            data["feature_label"] = "Variable"

    if "feature_description" not in data.columns:

        data["feature_description"] = ""

    if "impact" not in data.columns:

        data["impact"] = (
            data["shap_value"]
            .apply(
                lambda value:
                "Augmente le risque"
                if value > 0
                else "Réduit le risque"
            )
        )

    risk = (
        data[data["shap_value"] > 0]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .copy()
    )

    protective = (
        data[data["shap_value"] < 0]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .copy()
    )

    return risk, protective


# ==================================================
# ANALYSE METIER
# ==================================================

def get_position_label(
    value,
    population,
    higher_is_good=None
):

    population = population.dropna()

    if (
        pd.isna(value)
        or population.empty
    ):

        return "information non disponible"

    q1 = population.quantile(0.25)
    q3 = population.quantile(0.75)
    median_value = population.median()

    if value < q1:

        position = (
            "inférieur à la majorité des clients"
        )

    elif value > q3:

        position = (
            "supérieur à la majorité des clients"
        )

    else:

        position = (
            "proche de la zone centrale "
            "de la population"
        )

    if higher_is_good is None:
        return position

    if higher_is_good:

        impact = (
            "plutôt favorable"
            if value > median_value
            else "moins favorable"
        )

    else:

        impact = (
            "susceptible d'augmenter le risque"
            if value > median_value
            else "plutôt favorable"
        )

    return f"{position} ; cet élément est {impact}"


def build_client_summary(
    profile,
    population
):

    income = profile.get(
        "AMT_INCOME_TOTAL"
    )

    credit = profile.get(
        "AMT_CREDIT"
    )

    annuity = profile.get(
        "AMT_ANNUITY"
    )

    payment_rate = profile.get(
        "PAYMENT_RATE"
    )

    credit_income_ratio = profile.get(
        "CREDIT_INCOME_RATIO"
    )

    ext_source_mean = profile.get(
        "EXT_SOURCE_MEAN"
    )

    instal_dpd_mean = profile.get(
        "INSTAL_DPD_MEAN"
    )

    income_position = get_position_label(
        income,
        population["AMT_INCOME_TOTAL"],
        higher_is_good=True
    )

    credit_position = get_position_label(
        credit,
        population["AMT_CREDIT"]
    )

    payment_rate_position = get_position_label(
        payment_rate,
        population["PAYMENT_RATE"],
        higher_is_good=False
    )

    ext_source_position = get_position_label(
        ext_source_mean,
        population["EXT_SOURCE_MEAN"],
        higher_is_good=True
    )

    delay_position = get_position_label(
        instal_dpd_mean,
        population["INSTAL_DPD_MEAN"],
        higher_is_good=False
    )

    return (
        f"Le client dispose d'un revenu total de "
        f"{format_amount(income)}. "
        f"Ce revenu est {income_position}. "

        f"Le montant du crédit demandé est de "
        f"{format_amount(credit)} ; "
        f"il est {credit_position}. "

        f"L'annuité est de "
        f"{format_amount(annuity)}. "

        f"Le ratio annuité / crédit est de "
        f"{format_percent(payment_rate)} ; "
        f"il est {payment_rate_position}. "

        f"Le ratio crédit / revenu est de "
        f"{format_number(credit_income_ratio)}. "

        "Il permet d'apprécier le poids du "
        "crédit demandé par rapport aux "
        "ressources déclarées. "

        f"Le score externe moyen est de "
        f"{format_number(ext_source_mean)} ; "
        f"il est {ext_source_position}. "

        f"Le retard moyen de paiement historique "
        f"est de {format_number(instal_dpd_mean)} ; "
        f"il est {delay_position}."
    )


def build_distribution_summary(
    feature,
    label,
    value,
    population,
    value_type
):

    clean_population = (
        population[feature]
        .dropna()
    )

    if (
        pd.isna(value)
        or clean_population.empty
    ):

        return (
            f"Aucune comparaison fiable n'est "
            f"disponible pour la variable "
            f"« {label} »."
        )

    median_value = (
        clean_population.median()
    )

    percentile = (
        (clean_population < value).mean()
        * 100
    )

    return (
        f"Pour « {label} », la valeur du client "
        f"est {format_by_type(value, value_type)}. "

        f"La médiane de la population est "
        f"{format_by_type(median_value, value_type)}. "

        f"Le client se situe au-dessus d'environ "
        f"{percentile:.1f} % des clients."
    )


def build_bivariate_analysis_summary(
    x_label,
    y_label,
    client_x,
    client_y,
    x_type,
    y_type
):

    return (
        f"Le graphique compare "
        f"« {x_label} » et « {y_label} » "
        "dans la population de référence. "

        f"Le client présente "
        f"{format_by_type(client_x, x_type)} "
        f"pour « {x_label} » et "
        f"{format_by_type(client_y, y_type)} "
        f"pour « {y_label} ». "

        "Le point du client permet de situer "
        "son profil par rapport à la population."
    )


def get_decision_message(
    decision,
    probability,
    threshold
):

    distance_abs = abs(
        probability - threshold
    )

    if decision == "REFUSED":

        return (
            f"Le client présente une probabilité "
            f"de défaut estimée à "
            f"{probability:.2%}. "

            f"Le seuil métier est de "
            f"{threshold:.2%}. "

            f"Le risque estimé dépasse le seuil "
            f"de {distance_abs:.2%}. "

            "Le dossier est donc classé "
            "comme refusé par le modèle."
        )

    return (
        f"Le client présente une probabilité "
        f"de défaut estimée à "
        f"{probability:.2%}. "

        f"Le seuil métier est de "
        f"{threshold:.2%}. "

        f"Le risque estimé est inférieur au seuil "
        f"de {distance_abs:.2%}. "

        "Le dossier est donc classé "
        "comme accepté par le modèle."
    )


# ==================================================
# TABLEAUX
# ==================================================

def build_positioning_table(
    feature,
    value,
    population,
    value_type
):

    series = (
        population[feature]
        .dropna()
    )

    return pd.DataFrame(
        {
            "Indicateur": [
                "VALEUR CLIENT",
                "Valeur minimale globale",
                "1er quartile global",
                "Médiane globale",
                "3e quartile global",
                "Valeur maximale globale"
            ],

            "Valeur": [

                format_by_type(
                    value,
                    value_type
                ),

                format_by_type(
                    series.min(),
                    value_type
                ),

                format_by_type(
                    series.quantile(0.25),
                    value_type
                ),

                format_by_type(
                    series.median(),
                    value_type
                ),

                format_by_type(
                    series.quantile(0.75),
                    value_type
                ),

                format_by_type(
                    series.max(),
                    value_type
                )
            ]
        }
    )


# ==================================================
# GRAPHIQUES
# ==================================================

def plot_feature_distribution(
    data,
    feature,
    label,
    client_value
):

    plot_data = (
        data[[feature]]
        .dropna()
    )

    if plot_data.empty:
        return None

    if len(plot_data) > 50000:

        plot_data = plot_data.sample(
            50000,
            random_state=42
        )

    fig = px.histogram(
        plot_data,
        x=feature,
        nbins=50,
        title=f"Distribution - {label}",
        labels={
            feature: label
        },
        opacity=0.85
    )

    fig.update_traces(
        marker_color=ORANGE
    )

    fig.add_vline(
        x=client_value,
        line_width=3,
        line_dash="dash",
        line_color=BLUE
    )

    fig.add_trace(
        go.Scatter(
            x=[client_value],
            y=[0],
            mode="markers",
            marker=dict(
                size=14,
                color=BLUE,
                symbol="diamond"
            ),
            name="Client sélectionné"
        )
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title="Nombre de clients",
        yaxis_title_font_size=16,
        font=dict(size=14)
    )

    return fig


def plot_bivariate_analysis(
    data,
    x_feature,
    y_feature,
    x_label,
    y_label,
    client_x,
    client_y
):

    plot_data = (
        data[
            [x_feature, y_feature]
        ]
        .dropna()
    )

    if plot_data.empty:
        return None

    if len(plot_data) > 50000:

        plot_data = plot_data.sample(
            50000,
            random_state=42
        )

    x_min = plot_data[x_feature].min()
    x_max = plot_data[x_feature].max()

    y_min = plot_data[y_feature].min()
    y_max = plot_data[y_feature].max()

    x_padding = (
        (x_max - x_min) * 0.05
        if x_max != x_min
        else 1
    )

    y_padding = (
        (y_max - y_min) * 0.05
        if y_max != y_min
        else 1
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=plot_data[x_feature],
            y=plot_data[y_feature],
            mode="markers",
            marker=dict(
                size=6,
                color=ORANGE,
                opacity=0.25
            ),
            name="Population de référence"
        )
    )

    fig.add_shape(
        type="line",
        x0=client_x,
        x1=client_x,
        y0=y_min - y_padding,
        y1=y_max + y_padding,
        line=dict(
            color=BLUE,
            width=2,
            dash="dash"
        )
    )

    fig.add_shape(
        type="line",
        x0=x_min - x_padding,
        x1=x_max + x_padding,
        y0=client_y,
        y1=client_y,
        line=dict(
            color=BLUE,
            width=2,
            dash="dash"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[client_x],
            y=[client_y],
            mode="markers",
            marker=dict(
                size=26,
                color=BLUE,
                symbol="diamond",
                line=dict(
                    color="white",
                    width=3
                )
            ),
            name="Client sélectionné"
        )
    )

    fig.add_annotation(
        x=client_x,
        y=client_y,
        text="Client sélectionné",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=BLUE,
        ax=60,
        ay=-60,
        bgcolor="white",
        bordercolor=BLUE,
        borderwidth=1,
        font=dict(
            size=14,
            color=BLUE
        )
    )

    fig.update_layout(
        title=f"{x_label} vs {y_label}",
        height=700,
        showlegend=True,
        title_font_size=20,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            range=[
                x_min - x_padding,
                x_max + x_padding
            ]
        ),
        yaxis=dict(
            range=[
                y_min - y_padding,
                y_max + y_padding
            ]
        )
    )

    return fig


def plot_global_feature_importance(
    global_importance,
    top_n=15
):

    plot_data = (
        global_importance
        .head(top_n)
        .copy()
    )

    plot_data["feature_label"] = (
        plot_data["feature"]
        .apply(get_feature_label)
    )

    fig = px.bar(
        plot_data.sort_values(
            "importance",
            ascending=True
        ),
        x="importance",
        y="feature_label",
        orientation="h",
        title=(
            f"Top {top_n} des variables "
            "les plus importantes du modèle"
        ),
        labels={
            "importance": "Importance",
            "feature_label": "Variable"
        }
    )

    fig.update_traces(
        marker_color=BLUE
    )

    fig.update_layout(
        height=700,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font=dict(size=14)
    )

    return fig


def plot_local_shap(
    local_explanation
):

    plot_data = (
        local_explanation
        .copy()
    )

    plot_data["Effet"] = (
        plot_data["shap_value"]
        .apply(
            lambda value:
            "Augmente le risque"
            if value > 0
            else "Réduit le risque"
        )
    )

    fig = px.bar(
        plot_data.sort_values(
            "shap_value"
        ),
        x="shap_value",
        y="feature_label",
        orientation="h",
        color="Effet",
        color_discrete_map={
            "Augmente le risque": BLUE,
            "Réduit le risque": ORANGE
        },
        labels={
            "shap_value": "Contribution locale",
            "feature_label": "Variable"
        },
        title="Principales contributions locales"
    )

    fig.update_layout(
        height=700,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font=dict(size=14)
    )

    return fig


def display_shap_table(
    data,
    title
):

    st.markdown(
        f"#### {title}"
    )

    if data.empty:

        st.info(
            "Aucune variable dans cette catégorie "
            "pour ce client."
        )

        return

    columns = [
        "feature_label",
        "value",
        "shap_value",
        "impact",
        "feature_description"
    ]

    available_columns = [
        column
        for column in columns
        if column in data.columns
    ]

    display_data = (
        data[available_columns]
        .copy()
    )

    display_data = display_data.rename(
        columns={
            "feature_label": "Variable",
            "value": "Valeur client",
            "shap_value": "Contribution",
            "impact": "Effet",
            "feature_description":
                "Description métier"
        }
    )

    if "Contribution" in display_data.columns:

        display_data["Contribution"] = (
            display_data["Contribution"]
            .apply(format_shap_value)
        )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# INITIALISATION
# ==================================================

init_session_state()


# ==================================================
# HEADER
# ==================================================

st.title(
    "Prêt à dépenser — Decision insights"
)

st.markdown(
    f"*Version 2 mise à jour le "
    f"{date.today().strftime('%Y/%m/%d')}.*"
)

st.caption(
    "Les données clients sont récupérées "
    "à la demande depuis l'API Données."
)


# ==================================================
# SELECTION CLIENT
# ==================================================

st.subheader(
    "Sélection du client"
)

client_id = st.number_input(
    "Identifiant du client",
    min_value=1,
    step=1,
    value=(
        st.session_state.selected_client_id
        if st.session_state.selected_client_id is not None
        else 100002
    )
)

client_id = int(client_id)

reset_client_results_if_needed(
    client_id
)


# ==================================================
# PROFIL CLIENT
# ==================================================

if st.session_state.client_profile is None:

    try:

        with st.spinner(
            f"Chargement du client {client_id}..."
        ):

            st.session_state.client_profile = (
                get_client_profile(client_id)
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Impossible de joindre l'API Données. "
            "Vérifie que FastAPI est bien lancé "
            "sur le port 8001."
        )

        st.stop()

    except requests.exceptions.HTTPError as error:

        if (
            error.response is not None
            and error.response.status_code == 404
        ):

            st.error(
                f"Le client {client_id} "
                "n'existe pas dans la base."
            )

        else:

            st.error(
                f"Erreur HTTP lors de la récupération "
                f"du client : {error}"
            )

        st.stop()

    except Exception as error:

        st.error(
            f"Erreur lors de la récupération "
            f"du profil client : {error}"
        )

        st.stop()


profile = st.session_state.client_profile


# ==================================================
# NAVIGATION
# ==================================================

(
    tab_decision,
    tab_profile,
    tab_positioning,
    tab_simulation
) = st.tabs(
    [
        "Décision & explication",
        "Profil client",
        "Positionnement",
        "Simulation"
    ]
)


# ==================================================
# ONGLET 1 — DECISION & EXPLICATION
# ==================================================

with tab_decision:

    st.header(
        "Décision & explication"
    )

    # ------------------------------------------
    # CALCUL SCORE
    # ------------------------------------------

    if st.button(
        "Calculer le score",
        key="calculate_score"
    ):

        try:

            with st.spinner(
                "Calcul du score et de "
                "l'explication locale..."
            ):

                prediction = (
                    call_prediction_api(
                        profile
                    )
                )

                st.session_state.prediction = (
                    prediction
                )

                local_explanation = (
                    extract_local_explanation(
                        prediction
                    )
                )

                st.session_state.local_explanation = (
                    local_explanation
                )

                (
                    risk_contributions,
                    protective_contributions
                ) = split_shap_impacts(
                    local_explanation
                )

                st.session_state.risk_contributions = (
                    risk_contributions
                )

                st.session_state.protective_contributions = (
                    protective_contributions
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Impossible de joindre l'API Scoring. "
                "Vérifie que FastAPI est bien lancé "
                "sur le port 8000."
            )

        except requests.exceptions.HTTPError as error:

            st.error(
                f"Erreur HTTP lors de l'appel "
                f"à l'API Scoring : {error}"
            )

        except Exception as error:

            st.error(
                "Erreur lors du calcul du score "
                f"ou de l'explication : {error}"
            )

    # ------------------------------------------
    # AVANT CALCUL
    # ------------------------------------------

    if st.session_state.prediction is None:

        st.info(
            "Clique sur « Calculer le score » "
            "pour obtenir la décision du modèle "
            "et ses explications."
        )

    # ------------------------------------------
    # APRES CALCUL
    # ------------------------------------------

    else:

        prediction = (
            st.session_state.prediction
        )

        default_probability = (
            prediction["default_probability"]
        )

        business_threshold = (
            prediction["business_threshold"]
        )

        decision = (
            prediction["decision"]
        )

        decision_label = (
            translate_decision(
                decision
            ).upper()
        )

        distance_to_threshold = (
            default_probability
            - business_threshold
        )

        feature_coverage_rate = (
            prediction[
                "feature_coverage_rate"
            ]
        )

        data_completeness_label = (
            translate_completeness(
                prediction[
                    "data_completeness_level"
                ]
            )
        )

        warning = prediction.get(
            "warning",
            "Aucun avertissement."
        )

        # ------------------------------------------
        # KPI
        # ------------------------------------------

        st.subheader("KPI")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Probabilité de défaut",
            f"{default_probability:.2%}"
        )

        col2.metric(
            "Seuil métier",
            f"{business_threshold:.2%}"
        )

        col3.metric(
            "Distance au seuil",
            f"{distance_to_threshold:+.2%}"
        )

        # ------------------------------------------
        # DECISION
        # ------------------------------------------

        if decision == "REFUSED":

            st.markdown(
                f"""
                <div class="decision-refused">
                    <strong>
                        DÉCISION : {decision_label}
                    </strong>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="decision-accepted">
                    <strong>
                        DÉCISION : {decision_label}
                    </strong>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        st.markdown(
            "**Interprétation métier**"
        )

        st.write(
            get_decision_message(
                decision,
                default_probability,
                business_threshold
            )
        )

        # ------------------------------------------
        # QUALITE DES DONNEES
        # ------------------------------------------

        st.subheader(
            "Qualité des données envoyées au modèle"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Complétude",
            data_completeness_label
        )

        col2.metric(
            "Couverture des variables",
            f"{feature_coverage_rate:.2f}%"
        )

        st.write(
            warning
        )

        # ------------------------------------------
        # SHAP LOCAL
        # ------------------------------------------

        st.subheader(
            "Explication locale de la décision"
        )

        st.write(
            "Cette analyse identifie les variables "
            "qui influencent le plus la prédiction "
            "du client sélectionné. "

            "Une contribution positive augmente "
            "le risque estimé ; une contribution "
            "négative le réduit."
        )

        if (
            st.session_state.local_explanation
            is not None
        ):

            st.plotly_chart(
                plot_local_shap(
                    st.session_state.local_explanation
                ),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:

                display_shap_table(
                    st.session_state.risk_contributions,
                    "Variables qui augmentent le risque"
                )

            with col2:

                display_shap_table(
                    st.session_state.protective_contributions,
                    "Variables qui réduisent le risque"
                )

        else:

            st.info(
                "L'API Scoring n'a pas retourné "
                "d'explication SHAP locale."
            )

        # ------------------------------------------
        # SHAP GLOBAL
        # ------------------------------------------

        st.subheader(
            "Explication globale du modèle"
        )

        st.write(
            "Cette analyse présente les variables "
            "globalement les plus importantes pour "
            "le modèle LightGBM. "

            "Elle décrit le comportement général "
            "du modèle sur la population et ne "
            "correspond pas à l'effet d'une variable "
            "pour ce client particulier."
        )

        try:

            global_feature_importance = (
                get_global_feature_importance()
            )

            top_n_global = st.slider(
                "Nombre de variables affichées",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="top_n_global"
            )

            st.plotly_chart(
                plot_global_feature_importance(
                    global_feature_importance,
                    top_n=top_n_global
                ),
                use_container_width=True
            )

            top_features = (
                global_feature_importance
                .head(5)["feature"]
                .tolist()
            )

            st.write(
                "Résumé : les principales variables "
                "globales sont "
                + ", ".join(
                    get_feature_label(
                        feature
                    )
                    for feature in top_features
                )
                + "."
            )

        except Exception as error:

            st.warning(
                "Impossible de récupérer "
                f"l'importance globale : {error}"
            )


# ==================================================
# ONGLET 2 — PROFIL CLIENT
# ==================================================

with tab_profile:

    st.header(
        "Profil client"
    )

    st.subheader(
        "Informations principales"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Identifiant",
        str(
            int(
                profile["SK_ID_CURR"]
            )
        )
    )

    col2.metric(
        "Revenu total",
        format_amount(
            profile.get(
                "AMT_INCOME_TOTAL"
            )
        )
    )

    col3.metric(
        "Montant du crédit",
        format_amount(
            profile.get(
                "AMT_CREDIT"
            )
        )
    )

    col4.metric(
        "Annuité",
        format_amount(
            profile.get(
                "AMT_ANNUITY"
            )
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Ratio annuité / crédit",
        format_percent(
            profile.get(
                "PAYMENT_RATE"
            )
        )
    )

    col2.metric(
        "Ratio crédit / revenu",
        format_number(
            profile.get(
                "CREDIT_INCOME_RATIO"
            )
        )
    )

    col3.metric(
        "Score externe moyen",
        format_number(
            profile.get(
                "EXT_SOURCE_MEAN"
            )
        )
    )

    col4.metric(
        "Âge estimé",
        format_years_from_days(
            profile.get(
                "DAYS_BIRTH"
            )
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Ancienneté professionnelle",
        format_years_from_days(
            profile.get(
                "DAYS_EMPLOYED"
            )
        )
    )

    col2.metric(
        "Retard moyen paiement",
        format_number(
            profile.get(
                "INSTAL_DPD_MEAN"
            )
        )
    )

    col3.metric(
        "Retard max paiement",
        format_number(
            profile.get(
                "INSTAL_DPD_MAX"
            )
        )
    )

    col4.metric(
        "Durée moyenne crédits précédents",
        format_number(
            profile.get(
                "PREV_CNT_PAYMENT_MEAN"
            )
        )
    )

    # ------------------------------------------
    # ANALYSE METIER
    # ------------------------------------------

    st.subheader(
        "Analyse métier"
    )

    required_features = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "PAYMENT_RATE",
        "EXT_SOURCE_MEAN",
        "INSTAL_DPD_MEAN"
    ]

    try:

        population = get_population(
            required_features
        )

        st.write(
            build_client_summary(
                profile=profile,
                population=population
            )
        )

    except Exception as error:

        st.warning(
            "Impossible de calculer "
            f"l'analyse métier : {error}"
        )

    st.caption(
        "Cette analyse descriptive situe le client "
        "par rapport à une population de référence. "
        "Elle ne remplace pas l'explication du modèle."
    )


# ==================================================
# ONGLET 3 — POSITIONNEMENT
# ==================================================

with tab_positioning:

    st.header(
        "Positionnement"
    )

    available_features = {
        feature: meta
        for feature, meta
        in FEATURES_TO_COMPARE.items()
    }

    feature_labels = [
        meta["label"]
        for meta in available_features.values()
    ]

    # ------------------------------------------
    # ANALYSE UNIVARIEE
    # ------------------------------------------

    st.subheader(
        "Analyse univariée"
    )

    selected_feature_label = st.selectbox(
        "Choisir une variable à comparer",
        feature_labels,
        key="univariate_feature"
    )

    selected_feature = next(
        feature
        for feature, meta
        in available_features.items()
        if meta["label"]
        == selected_feature_label
    )

    feature_meta = (
        available_features[
            selected_feature
        ]
    )

    client_value = profile.get(
        selected_feature
    )

    st.markdown(
        f"**{feature_meta['label']}**"
    )

    st.write(
        feature_meta["description"]
    )

    if pd.isna(client_value):

        st.warning(
            "La valeur du client n'est pas disponible "
            "pour cette variable."
        )

    else:

        try:

            population = get_population(
                [selected_feature]
            )

            figure = plot_feature_distribution(
                data=population,
                feature=selected_feature,
                label=feature_meta["label"],
                client_value=client_value
            )

            if figure is not None:

                st.plotly_chart(
                    figure,
                    use_container_width=True
                )

            st.write(
                build_distribution_summary(
                    feature=selected_feature,
                    label=feature_meta["label"],
                    value=client_value,
                    population=population,
                    value_type=feature_meta["format"]
                )
            )

            positioning_table = (
                build_positioning_table(
                    feature=selected_feature,
                    value=client_value,
                    population=population,
                    value_type=feature_meta["format"]
                )
            )

            st.dataframe(
                positioning_table,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.error(
                "Impossible de récupérer "
                f"la population : {error}"
            )

    # ------------------------------------------
    # ANALYSE BIVARIEE
    # ------------------------------------------

    st.subheader(
        "Analyse bi-variée"
    )

    col1, col2 = st.columns(2)

    with col1:

        x_feature_label = st.selectbox(
            "Variable X",
            feature_labels,
            index=(
                1
                if len(feature_labels) > 1
                else 0
            ),
            key="bivariate_x"
        )

        x_feature = next(
            feature
            for feature, meta
            in available_features.items()
            if meta["label"]
            == x_feature_label
        )

        x_meta = (
            available_features[
                x_feature
            ]
        )

        st.markdown(
            f"**{x_meta['label']}**"
        )

        st.write(
            x_meta["description"]
        )

    with col2:

        y_feature_label = st.selectbox(
            "Variable Y",
            feature_labels,
            index=0,
            key="bivariate_y"
        )

        y_feature = next(
            feature
            for feature, meta
            in available_features.items()
            if meta["label"]
            == y_feature_label
        )

        y_meta = (
            available_features[
                y_feature
            ]
        )

        st.markdown(
            f"**{y_meta['label']}**"
        )

        st.write(
            y_meta["description"]
        )

    client_x = profile.get(
        x_feature
    )

    client_y = profile.get(
        y_feature
    )

    if x_feature == y_feature:

        st.warning(
            "Choisis deux variables différentes "
            "pour l'analyse bi-variée."
        )

    elif (
        pd.isna(client_x)
        or pd.isna(client_y)
    ):

        st.warning(
            "Les valeurs du client ne sont pas "
            "disponibles pour les deux variables "
            "sélectionnées."
        )

    else:

        try:

            population = get_population(
                [
                    x_feature,
                    y_feature
                ]
            )

            figure = plot_bivariate_analysis(
                data=population,
                x_feature=x_feature,
                y_feature=y_feature,
                x_label=x_meta["label"],
                y_label=y_meta["label"],
                client_x=client_x,
                client_y=client_y
            )

            if figure is not None:

                st.plotly_chart(
                    figure,
                    use_container_width=True
                )

            st.write(
                build_bivariate_analysis_summary(
                    x_label=x_meta["label"],
                    y_label=y_meta["label"],
                    client_x=client_x,
                    client_y=client_y,
                    x_type=x_meta["format"],
                    y_type=y_meta["format"]
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"**{x_meta['label']}**"
                )

                x_table = (
                    build_positioning_table(
                        feature=x_feature,
                        value=client_x,
                        population=population,
                        value_type=x_meta["format"]
                    )
                )

                st.dataframe(
                    x_table,
                    use_container_width=True,
                    hide_index=True
                )

            with col2:

                st.markdown(
                    f"**{y_meta['label']}**"
                )

                y_table = (
                    build_positioning_table(
                        feature=y_feature,
                        value=client_y,
                        population=population,
                        value_type=y_meta["format"]
                    )
                )

                st.dataframe(
                    y_table,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:

            st.error(
                "Impossible de récupérer "
                f"la population : {error}"
            )


# ==================================================
# ONGLET 4 — SIMULATION
# ==================================================

with tab_simulation:

    st.header(
        "Simulation"
    )

    st.info(
        "La simulation avec de nouvelles données "
        "sera intégrée dans cette section."
    )

    st.write(
        "Cette section permettra de modifier les "
        "informations du client puis de demander "
        "un nouveau score à l'API."
    )