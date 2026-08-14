import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from src.business_labels import get_feature_label
from src.explainability import compute_local_shap, split_shap_impacts


# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Dashboard scoring crédit - Prêt à Dépenser",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

DASHBOARD_CLIENTS_PATH = DATA_DIR / "dashboard_clients.parquet"
FEATURE_ENGINEERING_PATH = DATA_DIR / "feature_engineering.parquet"
GLOBAL_FEATURE_IMPORTANCE_PATH = DATA_DIR / "global_feature_importance.parquet"

API_URL = "http://127.0.0.1:8000/predict"


# ==================================================
# VARIABLES METIER
# ==================================================

FEATURES_TO_COMPARE = {
    "AMT_INCOME_TOTAL": {
        "label": "Revenu total",
        "format": "amount",
        "description": "Revenu total déclaré par le client."
    },
    "AMT_CREDIT": {
        "label": "Montant du crédit",
        "format": "amount",
        "description": "Montant du crédit demandé."
    },
    "AMT_ANNUITY": {
        "label": "Annuité",
        "format": "amount",
        "description": "Montant de l'annuité liée au crédit."
    },
    "PAYMENT_RATE": {
        "label": "Ratio annuité / crédit",
        "format": "percent",
        "description": "Part de l'annuité par rapport au montant total du crédit."
    },
    "CREDIT_INCOME_RATIO": {
        "label": "Ratio crédit / revenu",
        "format": "number",
        "description": "Poids du montant du crédit par rapport au revenu total."
    },
    "ANNUITY_INCOME_RATIO": {
        "label": "Ratio annuité / revenu",
        "format": "percent",
        "description": "Poids de l'annuité par rapport au revenu total."
    },
    "EXT_SOURCE_MEAN": {
        "label": "Score externe moyen",
        "format": "number",
        "description": "Score externe synthétique utilisé comme indicateur de solvabilité."
    },
    "INSTAL_DPD_MEAN": {
        "label": "Retard moyen de paiement",
        "format": "number",
        "description": "Retard moyen observé dans les paiements historiques."
    },
    "INSTAL_DPD_MAX": {
        "label": "Retard maximum de paiement",
        "format": "number",
        "description": "Retard maximum observé dans les paiements historiques."
    },
    "PREV_CNT_PAYMENT_MEAN": {
        "label": "Durée moyenne des crédits précédents",
        "format": "number",
        "description": "Nombre moyen d'échéances des précédentes demandes de crédit."
    },
    "BURO_AMT_CREDIT_SUM_MEAN": {
        "label": "Montant moyen des crédits externes",
        "format": "amount",
        "description": "Montant moyen des crédits observés dans l'historique bureau."
    }
}


# ==================================================
# CHARGEMENT
# ==================================================

@st.cache_data
def load_data():
    clients = pd.read_parquet(DASHBOARD_CLIENTS_PATH)
    profiles = pd.read_parquet(FEATURE_ENGINEERING_PATH)
    global_importance = pd.read_parquet(GLOBAL_FEATURE_IMPORTANCE_PATH)
    return clients, profiles, global_importance


# ==================================================
# FORMATAGE
# ==================================================

def format_amount(value):
    return "Non disponible" if pd.isna(value) else f"{value:,.0f} €".replace(",", " ")


def format_percent(value):
    return "Non disponible" if pd.isna(value) else f"{value:.2%}"


def format_number(value, decimals=2):
    return "Non disponible" if pd.isna(value) else f"{value:.{decimals}f}"


def format_years_from_days(value):
    return "Non disponible" if pd.isna(value) or value == 0 else f"{abs(value) / 365:.1f} ans"


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
    }.get(decision, decision)


def translate_completeness(level):
    return {
        "HIGH": "Élevée",
        "MEDIUM": "Moyenne",
        "LOW": "Faible"
    }.get(level, level)


def format_shap_value(value):
    return f"{value:+.4f}"


# ==================================================
# SESSION STATE
# ==================================================

def init_session_state():
    defaults = {
        "selected_client_id": None,
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
        st.session_state.prediction = None
        st.session_state.local_explanation = None
        st.session_state.risk_contributions = None
        st.session_state.protective_contributions = None


# ==================================================
# ANALYSE METIER
# ==================================================

def get_position_label(value, population, higher_is_good=None):
    if pd.isna(value):
        return "information non disponible"

    q1 = population.quantile(0.25)
    q3 = population.quantile(0.75)
    median_value = population.median()

    if value < q1:
        position = "inférieur à la majorité des clients"
    elif value > q3:
        position = "supérieur à la majorité des clients"
    else:
        position = "proche de la zone centrale de la population"

    if higher_is_good is None:
        return position

    if higher_is_good:
        impact = "plutôt favorable" if value > median_value else "moins favorable"
    else:
        impact = "susceptible d'augmenter le risque" if value > median_value else "plutôt favorable"

    return f"{position} ; cet élément est {impact}"


def build_client_summary(profile, population):
    income = profile.get("AMT_INCOME_TOTAL")
    credit = profile.get("AMT_CREDIT")
    annuity = profile.get("AMT_ANNUITY")
    payment_rate = profile.get("PAYMENT_RATE")
    credit_income_ratio = profile.get("CREDIT_INCOME_RATIO")
    ext_source_mean = profile.get("EXT_SOURCE_MEAN")
    instal_dpd_mean = profile.get("INSTAL_DPD_MEAN")

    income_position = get_position_label(income, population["AMT_INCOME_TOTAL"], higher_is_good=True)
    credit_position = get_position_label(credit, population["AMT_CREDIT"])
    payment_rate_position = get_position_label(payment_rate, population["PAYMENT_RATE"], higher_is_good=False)
    ext_source_position = get_position_label(ext_source_mean, population["EXT_SOURCE_MEAN"], higher_is_good=True)
    delay_position = get_position_label(instal_dpd_mean, population["INSTAL_DPD_MEAN"], higher_is_good=False)

    return (
        f"Le client dispose d'un revenu total de {format_amount(income)}. "
        f"Ce revenu est {income_position}. "
        f"Le montant du crédit demandé est de {format_amount(credit)} ; il est {credit_position}. "
        f"L'annuité est de {format_amount(annuity)}. "
        f"Le ratio annuité / crédit est de {format_percent(payment_rate)} ; il est {payment_rate_position}. "
        f"Le ratio crédit / revenu est de {format_number(credit_income_ratio)}. "
        f"Il permet d'apprécier le poids du crédit demandé par rapport aux ressources déclarées. "
        f"Le score externe moyen est de {format_number(ext_source_mean)} ; il est {ext_source_position}. "
        f"Le retard moyen de paiement historique est de {format_number(instal_dpd_mean)} ; il est {delay_position}."
    )


def build_distribution_summary(feature, label, value, population, value_type):
    clean_population = population[feature].dropna()

    if pd.isna(value) or clean_population.empty:
        return f"Aucune comparaison fiable n'est disponible pour la variable « {label} »."

    median_value = clean_population.median()
    percentile = (clean_population < value).mean() * 100

    return (
        f"Pour la variable « {label} », la valeur du client est {format_by_type(value, value_type)}. "
        f"La médiane observée dans la population est {format_by_type(median_value, value_type)}. "
        f"Le client se situe au-dessus d'environ {percentile:.1f} % des clients de la population."
    )


def build_bivariate_summary(x_label, y_label, client_x, client_y, x_type, y_type):
    return (
        f"Le graphique compare « {x_label} » et « {y_label} » pour la population de référence. "
        f"Le client sélectionné présente une valeur de {format_by_type(client_x, x_type)} pour « {x_label} » "
        f"et de {format_by_type(client_y, y_type)} pour « {y_label} ». "
        "Le point rouge permet de visualiser sa position par rapport aux autres clients."
    )


def get_decision_message(decision, probability, threshold):
    distance_abs = abs(probability - threshold)

    if decision == "REFUSED":
        return (
            f"Le client présente une probabilité de défaut estimée à {probability:.2%}. "
            f"Le seuil métier retenu est de {threshold:.2%}. "
            f"Le risque estimé dépasse le seuil métier de {distance_abs:.2%}. "
            "Le dossier est donc classé comme refusé par le modèle."
        )

    return (
        f"Le client présente une probabilité de défaut estimée à {probability:.2%}. "
        f"Le seuil métier retenu est de {threshold:.2%}. "
        f"Le risque estimé est inférieur au seuil métier de {distance_abs:.2%}. "
        "Le dossier est donc classé comme accepté par le modèle."
    )


# ==================================================
# API
# ==================================================

def call_prediction_api(selected_client):
    client_features = selected_client.drop(columns=["SK_ID_CURR"]).iloc[0].to_dict()

    payload = {
        "requested_by": "streamlit_dashboard",
        "clients": [client_features]
    }

    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()

    return response.json()["predictions"][0]


# ==================================================
# GRAPHIQUES
# ==================================================

def plot_feature_distribution(data, feature, label, client_value):
    plot_data = data[[feature]].dropna()

    if len(plot_data) > 50000:
        plot_data = plot_data.sample(50000, random_state=42)

    fig = px.histogram(
        plot_data,
        x=feature,
        nbins=50,
        title=f"Distribution - {label}",
        labels={feature: label},
        opacity=0.85
    )

    fig.add_vline(
        x=client_value,
        line_width=3,
        line_dash="dash",
        line_color="black"
    )

    fig.add_trace(
        go.Scatter(
            x=[client_value],
            y=[0],
            mode="markers",
            marker=dict(size=14, color="black", symbol="diamond"),
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


def plot_bivariate_analysis(data, x_feature, y_feature, x_label, y_label, client_x, client_y):
    plot_data = data[[x_feature, y_feature]].dropna()

    if len(plot_data) > 50000:
        plot_data = plot_data.sample(50000, random_state=42)

    x_min, x_max = plot_data[x_feature].min(), plot_data[x_feature].max()
    y_min, y_max = plot_data[y_feature].min(), plot_data[y_feature].max()

    x_padding = (x_max - x_min) * 0.05 if x_max != x_min else 1
    y_padding = (y_max - y_min) * 0.05 if y_max != y_min else 1

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=plot_data[x_feature],
            y=plot_data[y_feature],
            mode="markers",
            marker=dict(size=6, color="rgba(30, 120, 200, 0.22)", line=dict(width=0)),
            name="Population de référence",
            hovertemplate=f"{x_label} : %{{x}}<br>{y_label} : %{{y}}<extra></extra>"
        )
    )

    fig.add_shape(
        type="line",
        x0=client_x,
        x1=client_x,
        y0=y_min - y_padding,
        y1=y_max + y_padding,
        line=dict(color="black", width=2, dash="dash"),
        layer="below"
    )

    fig.add_shape(
        type="line",
        x0=x_min - x_padding,
        x1=x_max + x_padding,
        y0=client_y,
        y1=client_y,
        line=dict(color="black", width=2, dash="dash"),
        layer="below"
    )

    fig.add_trace(
        go.Scatter(
            x=[client_x],
            y=[client_y],
            mode="markers",
            marker=dict(
                size=26,
                color="red",
                symbol="diamond",
                line=dict(color="black", width=3)
            ),
            name="Client sélectionné",
            hovertemplate=f"Client sélectionné<br>{x_label} : %{{x}}<br>{y_label} : %{{y}}<extra></extra>"
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
        arrowcolor="black",
        ax=60,
        ay=-60,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=14, color="black")
    )

    fig.update_layout(
        title=f"Analyse bi-variée : {x_label} vs {y_label}",
        height=700,
        showlegend=True,
        title_font_size=20,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        xaxis=dict(range=[x_min - x_padding, x_max + x_padding]),
        yaxis=dict(range=[y_min - y_padding, y_max + y_padding])
    )

    return fig


def plot_global_feature_importance(global_importance, top_n=15):
    plot_data = global_importance.head(top_n).copy()
    plot_data["feature_label"] = plot_data["feature"].apply(get_feature_label)

    fig = px.bar(
        plot_data.sort_values("importance", ascending=True),
        x="importance",
        y="feature_label",
        orientation="h",
        title=f"Top {top_n} des variables les plus importantes du modèle",
        labels={
            "importance": "Importance",
            "feature_label": "Variable"
        }
    )

    fig.update_layout(
        height=700,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font=dict(size=14)
    )

    return fig


def plot_local_shap(local_explanation):
    plot_data = local_explanation.copy()
    plot_data["Effet"] = plot_data["shap_value"].apply(
        lambda value: "Augmente le risque" if value > 0 else "Réduit le risque"
    )

    fig = px.bar(
        plot_data.sort_values("shap_value"),
        x="shap_value",
        y="feature_label",
        orientation="h",
        color="Effet",
        color_discrete_map={
            "Augmente le risque": "#B00020",
            "Réduit le risque": "#006B3C"
        },
        labels={
            "shap_value": "Contribution locale",
            "feature_label": "Variable"
        },
        title="Principales contributions locales pour le client sélectionné"
    )

    fig.update_layout(
        height=700,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )

    return fig


def display_shap_table(data, title):
    st.markdown(f"#### {title}")

    if data.empty:
        st.info("Aucune variable dans cette catégorie pour ce client.")
        return

    display_data = data[
        ["feature_label", "value", "shap_value", "impact", "feature_description"]
    ].copy()

    display_data = display_data.rename(
        columns={
            "feature_label": "Variable",
            "value": "Valeur client",
            "shap_value": "Contribution",
            "impact": "Effet",
            "feature_description": "Description métier"
        }
    )

    display_data["Contribution"] = display_data["Contribution"].apply(format_shap_value)

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# DONNEES
# ==================================================

dashboard_clients, feature_engineering, global_feature_importance = load_data()
init_session_state()


# ==================================================
# HEADER
# ==================================================

st.title("Prêt à Dépenser")
st.caption("Dashboard d'aide à la décision crédit")
st.write("Version 1 du dashboard de scoring crédit.")
st.info(f"Nombre de clients disponibles : {len(dashboard_clients):,}".replace(",", " "))


# ==================================================
# SELECTION CLIENT
# ==================================================

client_id = st.selectbox(
    "Sélectionner un client",
    dashboard_clients["SK_ID_CURR"].tolist()
)

reset_client_results_if_needed(client_id)

selected_client = dashboard_clients[dashboard_clients["SK_ID_CURR"] == client_id]
selected_profile = feature_engineering[feature_engineering["SK_ID_CURR"] == client_id]


# ==================================================
# PROFIL CLIENT
# ==================================================

st.divider()
st.subheader("Profil client")

if selected_profile.empty:
    st.warning("Aucune information descriptive disponible pour ce client.")
    st.stop()

profile = selected_profile.iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Client", str(int(profile["SK_ID_CURR"])))
col2.metric("Revenu total", format_amount(profile.get("AMT_INCOME_TOTAL")))
col3.metric("Montant du crédit", format_amount(profile.get("AMT_CREDIT")))
col4.metric("Annuité", format_amount(profile.get("AMT_ANNUITY")))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ratio annuité / crédit", format_percent(profile.get("PAYMENT_RATE")))
col2.metric("Ratio crédit / revenu", format_number(profile.get("CREDIT_INCOME_RATIO")))
col3.metric("Score externe moyen", format_number(profile.get("EXT_SOURCE_MEAN")))
col4.metric("Âge estimé", format_years_from_days(profile.get("DAYS_BIRTH")))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ancienneté professionnelle", format_years_from_days(profile.get("DAYS_EMPLOYED")))
col2.metric("Retard moyen paiement", format_number(profile.get("INSTAL_DPD_MEAN")))
col3.metric("Retard max paiement", format_number(profile.get("INSTAL_DPD_MAX")))
col4.metric("Durée moyenne crédits précédents", format_number(profile.get("PREV_CNT_PAYMENT_MEAN")))


# ==================================================
# ANALYSE METIER
# ==================================================

st.divider()
st.subheader("Analyse métier du client")
st.write(build_client_summary(profile=profile, population=feature_engineering))
st.caption(
    "Cette lecture descriptive situe le client par rapport à la population de référence. "
    "Elle ne remplace pas l'explication complète du modèle."
)


# ==================================================
# SCORING API
# ==================================================

st.divider()
st.subheader("Score et décision")

if st.button("Calculer le score"):

    try:
        st.session_state.prediction = call_prediction_api(selected_client)

    except requests.exceptions.ConnectionError:
        st.error(
            "Impossible de joindre l'API. "
            "Vérifie que FastAPI est bien lancé sur http://127.0.0.1:8000."
        )

    except requests.exceptions.HTTPError as error:
        st.error(f"Erreur HTTP lors de l'appel API : {error}")

    except Exception as error:
        st.error(f"Erreur inattendue : {error}")

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    default_probability = prediction["default_probability"]
    business_threshold = prediction["business_threshold"]
    decision = prediction["decision"]
    decision_label = translate_decision(decision)
    feature_coverage_rate = prediction["feature_coverage_rate"]
    data_completeness_label = translate_completeness(prediction["data_completeness_level"])
    warning = prediction["warning"]
    distance_to_threshold = default_probability - business_threshold

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probabilité de défaut", f"{default_probability:.2%}")
    col2.metric("Seuil métier", f"{business_threshold:.2%}")
    col3.metric("Distance au seuil", f"{distance_to_threshold:.2%}")
    col4.metric("Décision", decision_label)

    st.divider()
    st.subheader("Qualité des données")

    col1, col2 = st.columns(2)
    col1.metric("Complétude", data_completeness_label)
    col2.metric("Couverture des variables", f"{feature_coverage_rate:.2f}%")

    st.divider()
    st.subheader("Interprétation métier")

    message = get_decision_message(
        decision=decision,
        probability=default_probability,
        threshold=business_threshold
    )

    if decision == "REFUSED":
        st.error(message)
    else:
        st.success(message)

    st.subheader("Message de complétude")
    st.warning(warning)

else:
    st.info("Clique sur « Calculer le score » pour obtenir la décision du modèle.")


# ==================================================
# EXPLICATION LOCALE SHAP
# ==================================================

st.divider()
st.subheader("Explication locale de la décision")

if st.session_state.prediction is None:
    st.info("Calcule d'abord le score du client avant d'afficher l'explication locale.")

else:
    st.write(
        "Cette section identifie les variables qui influencent le plus la prédiction pour le client sélectionné. "
        "Une contribution positive augmente le risque estimé ; une contribution négative le réduit."
    )

    top_n_shap = st.slider(
        "Nombre de variables locales à expliquer",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )

    if st.button("Calculer l'explication locale SHAP"):

        with st.spinner("Calcul de l'explication locale en cours..."):
            client_features_for_shap = selected_client.drop(columns=["SK_ID_CURR"])

            local_explanation = compute_local_shap(
                client_features=client_features_for_shap,
                top_n=top_n_shap
            )

            risk_contributions, protective_contributions = split_shap_impacts(
                local_explanation
            )

            st.session_state.local_explanation = local_explanation
            st.session_state.risk_contributions = risk_contributions
            st.session_state.protective_contributions = protective_contributions

    if st.session_state.local_explanation is not None:

        st.success("Explication locale disponible.")

        fig_local_shap = plot_local_shap(st.session_state.local_explanation)
        st.plotly_chart(fig_local_shap, use_container_width=True)

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

        st.caption(
            "Cette analyse explique la prédiction du client sélectionné uniquement. "
            "Elle ne doit pas être confondue avec l'importance globale du modèle."
        )


# ==================================================
# IMPORTANCE GLOBALE DU MODELE
# ==================================================

st.divider()
st.subheader("Importance globale du modèle")

st.write(
    "Ce graphique présente les variables globalement les plus importantes dans le modèle LightGBM. "
    "Il décrit le comportement général du modèle sur la population."
)

top_n = st.slider(
    "Nombre de variables à afficher",
    min_value=5,
    max_value=30,
    value=15,
    step=5
)

fig_global_importance = plot_global_feature_importance(
    global_importance=global_feature_importance,
    top_n=top_n
)

st.plotly_chart(fig_global_importance, use_container_width=True)

top_features = global_feature_importance.head(5)["feature"].tolist()

st.caption(
    "Résumé : les principales variables globales sont "
    + ", ".join([get_feature_label(feature) for feature in top_features])
    + ". Cette analyse globale ne décrit pas encore l'effet précis de ces variables pour le client sélectionné."
)


# ==================================================
# COMPARAISON A LA POPULATION
# ==================================================

st.divider()
st.subheader("Comparaison à la population")

available_features = {
    feature: meta
    for feature, meta in FEATURES_TO_COMPARE.items()
    if feature in feature_engineering.columns
}

feature_labels = [meta["label"] for meta in available_features.values()]

selected_feature_label = st.selectbox("Choisir une variable à comparer", feature_labels)

selected_feature = [
    feature
    for feature, meta in available_features.items()
    if meta["label"] == selected_feature_label
][0]

feature_meta = available_features[selected_feature]
client_value = profile.get(selected_feature)

st.write(feature_meta["description"])

if pd.isna(client_value):
    st.warning("La valeur du client n'est pas disponible pour cette variable.")
else:
    fig = plot_feature_distribution(
        data=feature_engineering,
        feature=selected_feature,
        label=feature_meta["label"],
        client_value=client_value
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        build_distribution_summary(
            feature=selected_feature,
            label=feature_meta["label"],
            value=client_value,
            population=feature_engineering,
            value_type=feature_meta["format"]
        )
    )


# ==================================================
# ANALYSE BI-VARIEE
# ==================================================

st.divider()
st.subheader("Analyse bi-variée")

col1, col2 = st.columns(2)

with col1:
    x_feature_label = st.selectbox(
        "Variable X",
        feature_labels,
        index=1 if len(feature_labels) > 1 else 0
    )

with col2:
    y_feature_label = st.selectbox(
        "Variable Y",
        feature_labels,
        index=0
    )

x_feature = [
    feature
    for feature, meta in available_features.items()
    if meta["label"] == x_feature_label
][0]

y_feature = [
    feature
    for feature, meta in available_features.items()
    if meta["label"] == y_feature_label
][0]

x_meta = available_features[x_feature]
y_meta = available_features[y_feature]

client_x = profile.get(x_feature)
client_y = profile.get(y_feature)

if x_feature == y_feature:
    st.warning("Choisis deux variables différentes pour l'analyse bi-variée.")
elif pd.isna(client_x) or pd.isna(client_y):
    st.warning("Les valeurs du client ne sont pas disponibles pour les deux variables sélectionnées.")
else:
    fig_bivariate = plot_bivariate_analysis(
        data=feature_engineering,
        x_feature=x_feature,
        y_feature=y_feature,
        x_label=x_meta["label"],
        y_label=y_meta["label"],
        client_x=client_x,
        client_y=client_y
    )

    st.plotly_chart(fig_bivariate, use_container_width=True)

    st.caption(
        build_bivariate_summary(
            x_label=x_meta["label"],
            y_label=y_meta["label"],
            client_x=client_x,
            client_y=client_y,
            x_type=x_meta["format"],
            y_type=y_meta["format"]
        )
    )