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
# Streamlit constitue l'interface du conseiller bancaire.
# Les URLs sont configurables par variables d'environnement afin
# de pouvoir utiliser la même application en local ou en déploiement.
st.set_page_config(page_title="Prêt à dépenser — Decision insights", layout="wide", initial_sidebar_state="collapsed")
SCORING_API_URL = os.getenv("SCORING_API_URL", "http://127.0.0.1:8000")
DATA_API_URL = os.getenv("DATA_API_URL", "http://127.0.0.1:8001")
ORANGE, BLUE = "#FFB24A", "#06006C"
LIGHT_GREEN, LIGHT_RED = "#E8F5E9", "#FDECEC"

# ==================================================
# STYLE
# ==================================================
# Le CSS simplifie l'interface et met visuellement en avant
# les éléments importants pour la décision : onglets, boutons et résultat.
st.markdown(f"""<style>
[data-testid="stSidebar"],[data-testid="collapsedControl"]{{display:none}}
[data-testid="stTabs"] [role="tablist"]{{gap:8px}}
[data-testid="stTabs"] button{{color:{BLUE};font-weight:600;border-radius:8px 8px 0 0;padding:10px 18px}}
[data-testid="stTabs"] button[aria-selected="true"]{{background-color:{BLUE};color:white}}
.stButton > button{{background-color:{ORANGE};color:{BLUE};border:1px solid {ORANGE};font-weight:700;border-radius:8px}}
.stButton > button:hover{{background-color:{BLUE};color:white;border-color:{BLUE}}}
h1,h2,h3{{color:{BLUE}}}
.decision-accepted{{background-color:{LIGHT_GREEN};border-left:6px solid #2E7D32;padding:1rem;border-radius:8px}}
.decision-refused{{background-color:{LIGHT_RED};border-left:6px solid #C62828;padding:1rem;border-radius:8px}}
</style>""", unsafe_allow_html=True)

# ==================================================
# VARIABLES METIER
# ==================================================
# Ces métadonnées permettent de transformer les noms techniques
# des variables en informations compréhensibles par un conseiller.
FEATURES_TO_COMPARE = {
    "AMT_INCOME_TOTAL":{"label":"Revenu total","format":"amount","description":"Revenu total déclaré par le client."},
    "AMT_CREDIT":{"label":"Montant du crédit","format":"amount","description":"Montant du crédit demandé."},
    "AMT_ANNUITY":{"label":"Annuité","format":"amount","description":"Montant de l'annuité liée au crédit."},
    "PAYMENT_RATE":{"label":"Ratio annuité / crédit","format":"percent","description":"Part de l'annuité par rapport au montant total du crédit."},
    "CREDIT_INCOME_RATIO":{"label":"Ratio crédit / revenu","format":"number","description":"Poids du montant du crédit par rapport au revenu total."},
    "ANNUITY_INCOME_RATIO":{"label":"Ratio annuité / revenu","format":"percent","description":"Poids de l'annuité par rapport au revenu total."},
    "EXT_SOURCE_MEAN":{"label":"Score externe moyen","format":"number","description":"Score externe synthétique utilisé comme indicateur de solvabilité."},
    "INSTAL_DPD_MEAN":{"label":"Retard moyen de paiement","format":"number","description":"Retard moyen observé dans les paiements historiques."},
    "INSTAL_DPD_MAX":{"label":"Retard maximum de paiement","format":"number","description":"Retard maximum observé dans les paiements historiques."},
    "PREV_CNT_PAYMENT_MEAN":{"label":"Durée moyenne des crédits précédents","format":"number","description":"Nombre moyen d'échéances des précédentes demandes de crédit."},
    "BURO_AMT_CREDIT_SUM_MEAN":{"label":"Montant moyen des crédits externes","format":"amount","description":"Montant moyen des crédits observés dans l'historique bureau."}
}

# ==================================================
# FORMATAGE
# ==================================================
# Ces fonctions convertissent les valeurs techniques en valeurs
# lisibles pour un utilisateur métier.
def format_amount(value):
    return "Non disponible" if pd.isna(value) else f"{value:,.0f} €".replace(",", " ")

def format_percent(value):
    return "Non disponible" if pd.isna(value) else f"{value:.2%}"

def format_number(value, decimals=2):
    return "Non disponible" if pd.isna(value) else f"{value:.{decimals}f}"

def format_years_from_days(value):
    return "Non disponible" if pd.isna(value) or value == 0 else f"{abs(value)/365:.1f} ans"

def format_by_type(value, value_type):
    return format_amount(value) if value_type == "amount" else format_percent(value) if value_type == "percent" else format_number(value)

def translate_decision(decision):
    return {"REFUSED":"Crédit refusé","ACCEPTED":"Crédit accepté"}.get(decision, decision)

def translate_completeness(level):
    return {"HIGH":"Élevée","MEDIUM":"Moyenne","LOW":"Faible"}.get(level, level)

def format_shap_value(value):
    return f"{value:+.4f}"

# ==================================================
# SESSION STATE
# ==================================================
# Streamlit réexécute le script à chaque interaction.
# session_state conserve donc le client, son score et les explications
# afin d'éviter de recalculer inutilement les informations.
def init_session_state():
    defaults = {"selected_client_id":None,"client_profile":None,"prediction":None,"local_explanation":None,"risk_contributions":None,"protective_contributions":None}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def reset_client_results_if_needed(client_id):
    if st.session_state.selected_client_id != client_id:
        st.session_state.selected_client_id = client_id
        for key in ["client_profile","prediction","local_explanation","risk_contributions","protective_contributions"]:
            st.session_state[key] = None

# ==================================================
# API — DONNEES
# ==================================================
# L'interface ne lit jamais directement la base SQLite.
# Elle passe par l'API Données : cela sépare l'interface, les données
# et la logique d'accès à la base.
@st.cache_data(ttl=300)
def get_client_profile(client_id):
    response = requests.get(f"{DATA_API_URL}/clients/{client_id}", timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["client"] if "client" in data else data

@st.cache_data(ttl=300)
def get_population(features, sample_size=5000):
    response = requests.get(f"{DATA_API_URL}/population", params={"features":",".join(features),"limit":sample_size}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "clients" not in data: raise ValueError("La réponse de l'API Données ne contient pas la clé 'clients'.")
    population = pd.DataFrame(data["clients"])
    missing = [f for f in features if f not in population.columns]
    if missing: raise ValueError(f"Variables absentes de la population : {missing}")
    return population[features]

@st.cache_data(ttl=300)
def get_global_feature_importance():
    response = requests.get(f"{DATA_API_URL}/feature-importance", timeout=30)
    response.raise_for_status()
    data = response.json()
    if "features" not in data: raise ValueError("La réponse de l'API Données ne contient pas la clé 'features'.")
    return pd.DataFrame(data["features"])

# ==================================================
# API — SCORING
# ==================================================
# Le dashboard transmet les variables du client à l'API Scoring.
# Le modèle reste donc isolé de l'interface utilisateur.
def call_prediction_api(profile):
    client_features={key:value for key,value in profile.items() if key!="SK_ID_CURR"}
    payload={"requested_by":"streamlit_dashboard","clients":[client_features]}

    response=requests.post(f"{SCORING_API_URL}/predict",json=payload,timeout=60)
    response.raise_for_status()
    prediction=response.json()["predictions"][0]

    shap_response=requests.post(f"{SCORING_API_URL}/shap",json=payload,timeout=60)
    shap_response.raise_for_status()
    explanation=shap_response.json().get("explanations",[{}])[0]

    prediction["local_explanation"]=explanation.get("shap_contributions")
    prediction["base_value"]=explanation.get("base_value")

    return prediction

# ==================================================
# EXTRACTION SHAP
# ==================================================
# SHAP explique localement pourquoi le modèle a produit
# une prédiction donnée pour CE client.
def extract_local_explanation(prediction):
    shap_data=prediction.get("local_explanation") or prediction.get("shap") or prediction.get("shap_values")
    if shap_data is None: return None
    if isinstance(shap_data,dict): shap_data=shap_data.get("features") or shap_data.get("data") or shap_data
    dataframe=pd.DataFrame(shap_data)
    if dataframe.empty: return None
    if "feature_value" in dataframe.columns and "value" not in dataframe.columns: dataframe["value"]=dataframe["feature_value"]
    if "effect" in dataframe.columns and "impact" not in dataframe.columns:
        dataframe["impact"]=dataframe["effect"].map({"INCREASES_DEFAULT_RISK":"Augmente le risque","DECREASES_DEFAULT_RISK":"Réduit le risque"}).fillna(dataframe["effect"])
    if "feature_label" not in dataframe.columns and "feature" in dataframe.columns: dataframe["feature_label"]=dataframe["feature"].apply(get_feature_label)
    if "feature_description" not in dataframe.columns: dataframe["feature_description"]=""
    return dataframe

def split_shap_impacts(data):
    if data is None or data.empty or "shap_value" not in data.columns: return pd.DataFrame(),pd.DataFrame()
    data=data.copy()
    if "feature_label" not in data.columns:
        data["feature_label"]=data["feature"].apply(get_feature_label) if "feature" in data.columns else "Variable"
    if "feature_description" not in data.columns: data["feature_description"]=""
    if "impact" not in data.columns: data["impact"]=data["shap_value"].apply(lambda value:"Augmente le risque" if value>0 else "Réduit le risque")
    risk=data[data["shap_value"]>0].sort_values("shap_value",ascending=False).copy()
    protective=data[data["shap_value"]<0].sort_values("shap_value").copy()
    return risk,protective

# ==================================================
# ANALYSE METIER
# ==================================================
# On situe le client dans la population : quartiles, médiane et
# position relative. Cela apporte une lecture métier complémentaire
# à la décision du modèle.
def get_position_label(value, population, higher_is_good=None):
    population = population.dropna()
    if pd.isna(value) or population.empty: return "information non disponible"
    q1, q3, median_value = population.quantile(.25), population.quantile(.75), population.median()
    position = "inférieur à la majorité des clients" if value < q1 else "supérieur à la majorité des clients" if value > q3 else "proche de la zone centrale de la population"
    if higher_is_good is None: return position
    impact = ("plutôt favorable" if value > median_value else "moins favorable") if higher_is_good else ("susceptible d'augmenter le risque" if value > median_value else "plutôt favorable")
    return f"{position} ; cet élément est {impact}"

def build_client_summary(profile, population):
    income, credit, annuity = profile.get("AMT_INCOME_TOTAL"), profile.get("AMT_CREDIT"), profile.get("AMT_ANNUITY")
    payment_rate, credit_income_ratio = profile.get("PAYMENT_RATE"), profile.get("CREDIT_INCOME_RATIO")
    ext_source_mean, instal_dpd_mean = profile.get("EXT_SOURCE_MEAN"), profile.get("INSTAL_DPD_MEAN")
    return (
        f"Le client dispose d'un revenu total de {format_amount(income)}. Ce revenu est {get_position_label(income,population['AMT_INCOME_TOTAL'],True)}. "
        f"Le montant du crédit demandé est de {format_amount(credit)} ; il est {get_position_label(credit,population['AMT_CREDIT'])}. "
        f"L'annuité est de {format_amount(annuity)}. Le ratio annuité / crédit est de {format_percent(payment_rate)} ; il est {get_position_label(payment_rate,population['PAYMENT_RATE'],False)}. "
        f"Le ratio crédit / revenu est de {format_number(credit_income_ratio)}. Il permet d'apprécier le poids du crédit demandé par rapport aux ressources déclarées. "
        f"Le score externe moyen est de {format_number(ext_source_mean)} ; il est {get_position_label(ext_source_mean,population['EXT_SOURCE_MEAN'],True)}. "
        f"Le retard moyen de paiement historique est de {format_number(instal_dpd_mean)} ; il est {get_position_label(instal_dpd_mean,population['INSTAL_DPD_MEAN'],False)}."
    )

def build_distribution_summary(feature,label,value,population,value_type):
    clean = population[feature].dropna()
    if pd.isna(value) or clean.empty: return f"Aucune comparaison fiable n'est disponible pour la variable « {label} »."
    median, percentile = clean.median(), (clean < value).mean()*100
    return f"Pour « {label} », la valeur du client est {format_by_type(value,value_type)}. La médiane de la population est {format_by_type(median,value_type)}. Le client se situe au-dessus d'environ {percentile:.1f} % des clients."

def build_bivariate_analysis_summary(x_label,y_label,client_x,client_y,x_type,y_type):
    return f"Le graphique compare « {x_label} » et « {y_label} » dans la population de référence. Le client présente {format_by_type(client_x,x_type)} pour « {x_label} » et {format_by_type(client_y,y_type)} pour « {y_label} ». Le point du client permet de situer son profil par rapport à la population."

def get_decision_message(decision,probability,threshold):
    distance = abs(probability-threshold)
    if decision == "REFUSED":
        return f"Le client présente une probabilité de défaut estimée à {probability:.2%}. Le seuil métier est de {threshold:.2%}. Le risque estimé dépasse le seuil de {distance:.2%}. Le dossier est donc classé comme refusé par le modèle."
    return f"Le client présente une probabilité de défaut estimée à {probability:.2%}. Le seuil métier est de {threshold:.2%}. Le risque estimé est inférieur au seuil de {distance:.2%}. Le dossier est donc classé comme accepté par le modèle."

# ==================================================
# TABLEAUX
# ==================================================
# Le tableau permet au conseiller de comparer rapidement
# la valeur du client aux statistiques de la population.
def build_positioning_table(feature,value,population,value_type):
    series = population[feature].dropna()
    return pd.DataFrame({
        "Indicateur":["VALEUR CLIENT","Valeur minimale globale","1er quartile global","Médiane globale","3e quartile global","Valeur maximale globale"],
        "Valeur":[
            format_by_type(value,value_type),format_by_type(series.min(),value_type),
            format_by_type(series.quantile(.25),value_type),format_by_type(series.median(),value_type),
            format_by_type(series.quantile(.75),value_type),format_by_type(series.max(),value_type)
        ]
    })

# ==================================================
# GRAPHIQUES
# ==================================================
# Les graphiques donnent une représentation visuelle de la position
# du client dans la population de référence.
def plot_feature_distribution(data,feature,label,client_value):
    plot_data = data[[feature]].dropna()
    if plot_data.empty: return None
    if len(plot_data) > 50000: plot_data = plot_data.sample(50000,random_state=42)
    fig = px.histogram(plot_data,x=feature,nbins=50,title=f"Distribution - {label}",labels={feature:label},opacity=.85)
    fig.update_traces(marker_color=ORANGE)
    fig.add_vline(x=client_value,line_width=3,line_dash="dash",line_color=BLUE)
    fig.add_trace(go.Scatter(x=[client_value],y=[0],mode="markers",marker=dict(size=14,color=BLUE,symbol="diamond"),name="Client sélectionné"))
    fig.update_layout(height=600,showlegend=True,title_font_size=20,xaxis_title_font_size=16,yaxis_title="Nombre de clients",yaxis_title_font_size=16,font=dict(size=14))
    return fig

def plot_bivariate_analysis(data,x_feature,y_feature,x_label,y_label,client_x,client_y):
    plot_data = data[[x_feature,y_feature]].dropna()
    if plot_data.empty: return None
    if len(plot_data)>50000: plot_data=plot_data.sample(50000,random_state=42)
    x_min,x_max,y_min,y_max = plot_data[x_feature].min(),plot_data[x_feature].max(),plot_data[y_feature].min(),plot_data[y_feature].max()
    xp,yp = ((x_max-x_min)*.05 if x_max!=x_min else 1),((y_max-y_min)*.05 if y_max!=y_min else 1)
    fig=go.Figure()
    fig.add_trace(go.Scattergl(x=plot_data[x_feature],y=plot_data[y_feature],mode="markers",marker=dict(size=6,color=ORANGE,opacity=.25),name="Population de référence"))
    fig.add_shape(type="line",x0=client_x,x1=client_x,y0=y_min-yp,y1=y_max+yp,line=dict(color=BLUE,width=2,dash="dash"))
    fig.add_shape(type="line",x0=x_min-xp,x1=x_max+xp,y0=client_y,y1=client_y,line=dict(color=BLUE,width=2,dash="dash"))
    fig.add_trace(go.Scatter(x=[client_x],y=[client_y],mode="markers",marker=dict(size=26,color=BLUE,symbol="diamond",line=dict(color="white",width=3)),name="Client sélectionné"))
    fig.add_annotation(x=client_x,y=client_y,text="Client sélectionné",showarrow=True,arrowhead=2,arrowwidth=2,arrowcolor=BLUE,ax=60,ay=-60,bgcolor="white",bordercolor=BLUE,borderwidth=1,font=dict(size=14,color=BLUE))
    fig.update_layout(title=f"{x_label} vs {y_label}",height=700,showlegend=True,title_font_size=20,xaxis_title=x_label,yaxis_title=y_label,xaxis_title_font_size=16,yaxis_title_font_size=16,font=dict(size=14),xaxis=dict(range=[x_min-xp,x_max+xp]),yaxis=dict(range=[y_min-yp,y_max+yp]))
    return fig

def plot_global_feature_importance(global_importance,top_n=15):
    data=global_importance.head(top_n).copy()
    data["feature_label"]=data["feature"].apply(get_feature_label)
    fig=px.bar(data.sort_values("importance"),x="importance",y="feature_label",orientation="h",title=f"Top {top_n} des variables les plus importantes du modèle",labels={"importance":"Importance","feature_label":"Variable"})
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(height=700,title_font_size=20,xaxis_title_font_size=16,yaxis_title_font_size=16,font=dict(size=14))
    return fig

def plot_local_shap(local_explanation):
    data=local_explanation.copy()
    data["Effet"]=data["shap_value"].apply(lambda value:"Augmente le risque" if value>0 else "Réduit le risque")
    fig=px.bar(data.sort_values("shap_value"),x="shap_value",y="feature_label",orientation="h",color="Effet",color_discrete_map={"Augmente le risque":BLUE,"Réduit le risque":ORANGE},labels={"shap_value":"Contribution locale","feature_label":"Variable"},title="Principales contributions locales")
    fig.update_layout(height=700,title_font_size=20,xaxis_title_font_size=16,yaxis_title_font_size=16,font=dict(size=14))
    return fig

def display_shap_table(data,title):
    st.markdown(f"#### {title}")
    if data.empty:
        st.info("Aucune variable dans cette catégorie pour ce client.")
        return
    columns=["feature_label","value","shap_value","impact","feature_description"]
    data=data[[c for c in columns if c in data.columns]].copy().rename(columns={"feature_label":"Variable","value":"Valeur client","shap_value":"Contribution","impact":"Effet","feature_description":"Description métier"})
    if "Contribution" in data.columns: data["Contribution"]=data["Contribution"].apply(format_shap_value)
    st.dataframe(data,use_container_width=True,hide_index=True)

# ==================================================
# INITIALISATION
# ==================================================
init_session_state()

# ==================================================
# HEADER
# ==================================================
st.title("Prêt à dépenser — Decision insights")
st.markdown(f"*Version 2 mise à jour le {date.today().strftime('%Y/%m/%d')}.*")
st.caption("Les données clients sont récupérées à la demande depuis l'API Données.")

# ==================================================
# SELECTION CLIENT
# ==================================================
st.subheader("Sélection du client")
client_id=int(st.number_input("Identifiant du client",min_value=1,step=1,value=st.session_state.selected_client_id if st.session_state.selected_client_id is not None else 100002))
reset_client_results_if_needed(client_id)

# ==================================================
# CHARGEMENT DU PROFIL
# ==================================================
# Le profil est chargé uniquement lorsque nécessaire.
# Le conseiller travaille donc sur le client sélectionné sans charger
# toute la base en mémoire.
if st.session_state.client_profile is None:
    try:
        with st.spinner(f"Chargement du client {client_id}..."):
            st.session_state.client_profile=get_client_profile(client_id)
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API Données. Vérifie que FastAPI est bien lancé sur le port 8001.")
        st.stop()
    except requests.exceptions.HTTPError as error:
        if error.response is not None and error.response.status_code==404:
            st.error(f"Le client {client_id} n'existe pas dans la base.")
        else:
            st.error(f"Erreur HTTP lors de la récupération du client : {error}")
        st.stop()
    except Exception as error:
        st.error(f"Erreur lors de la récupération du profil client : {error}")
        st.stop()

profile=st.session_state.client_profile

# ==================================================
# NAVIGATION
# ==================================================
tab_decision,tab_profile,tab_positioning,tab_simulation=st.tabs(["Décision & explication","Profil client","Positionnement","Simulation"])

# ==================================================
# ONGLET 1 — DECISION
# ==================================================
with tab_decision:
    st.header("Décision & explication")
    # L'appel au modèle n'est effectué que lorsque le conseiller le demande.
    if st.button("Calculer le score",key="calculate_score"):
        try:
            with st.spinner("Calcul du score et de l'explication locale..."):
                prediction=call_prediction_api(profile)
                st.session_state.prediction=prediction
                local_explanation=extract_local_explanation(prediction)
                st.session_state.local_explanation=local_explanation
                st.session_state.risk_contributions,st.session_state.protective_contributions=split_shap_impacts(local_explanation)
        except requests.exceptions.ConnectionError:
            st.error("Impossible de joindre l'API Scoring. Vérifie que FastAPI est bien lancé sur le port 8000.")
        except requests.exceptions.HTTPError as error:
            st.error(f"Erreur HTTP lors de l'appel à l'API Scoring : {error}")
        except Exception as error:
            st.error(f"Erreur lors du calcul du score ou de l'explication : {error}")

    if st.session_state.prediction is None:
        st.info("Clique sur « Calculer le score » pour obtenir la décision du modèle et ses explications.")
    else:
        prediction=st.session_state.prediction
        probability=prediction["default_probability"]
        threshold=prediction["business_threshold"]
        decision=prediction["decision"]
        decision_label=translate_decision(decision).upper()
        distance=probability-threshold
        coverage=prediction["feature_coverage_rate"]
        completeness=translate_completeness(prediction["data_completeness_level"])
        warning=prediction.get("warning","Aucun avertissement.")

        # Ces KPI permettent au conseiller de comprendre immédiatement
        # le niveau de risque et la proximité de la frontière de décision.
        st.subheader("KPI")
        col1,col2,col3=st.columns(3)
        col1.metric("Probabilité de défaut",f"{probability:.2%}")
        col2.metric("Seuil métier",f"{threshold:.2%}")
        col3.metric("Distance au seuil",f"{distance:+.2%}")

        css_class="decision-refused" if decision=="REFUSED" else "decision-accepted"
        st.markdown(f'<div class="{css_class}"><strong>DÉCISION : {decision_label}</strong></div>',unsafe_allow_html=True)
        st.markdown("**Interprétation métier**")
        st.write(get_decision_message(decision,probability,threshold))

        # La qualité des données est affichée pour éviter de présenter
        # une prédiction sans indiquer si les données étaient complètes.
        st.subheader("Qualité des données envoyées au modèle")
        col1,col2=st.columns(2)
        col1.metric("Complétude",completeness)
        col2.metric("Couverture des variables",f"{coverage:.2f}%")
        st.write(warning)

        # SHAP local : quelles variables ont influencé CE client ?
        st.subheader("Explication locale de la décision")
        st.write("Une contribution positive augmente le risque estimé ; une contribution négative le réduit.")
        if st.session_state.local_explanation is not None:
            st.plotly_chart(plot_local_shap(st.session_state.local_explanation),use_container_width=True)
            col1,col2=st.columns(2)
            with col1: display_shap_table(st.session_state.risk_contributions,"Variables qui augmentent le risque")
            with col2: display_shap_table(st.session_state.protective_contributions,"Variables qui réduisent le risque")
        else:
            st.info("L'API Scoring n'a pas retourné d'explication SHAP locale.")

        # SHAP global : quelles variables sont importantes pour le modèle
        # dans son ensemble ? Ce résultat n'est pas spécifique au client.
        st.subheader("Explication globale du modèle")
        st.write("Cette analyse présente les variables globalement les plus importantes pour le modèle LightGBM. Elle décrit le comportement général du modèle sur la population.")
        try:
            global_importance=get_global_feature_importance()
            top_n=st.slider("Nombre de variables affichées",5,30,15,5,key="top_n_global")
            st.plotly_chart(plot_global_feature_importance(global_importance,top_n),use_container_width=True)
            top_features=global_importance.head(5)["feature"].tolist()
            st.write("Résumé : les principales variables globales sont "+", ".join(get_feature_label(f) for f in top_features)+".")
        except Exception as error:
            st.warning(f"Impossible de récupérer l'importance globale : {error}")

# ==================================================
# ONGLET 2 — PROFIL CLIENT
# ==================================================
with tab_profile:
    st.header("Profil client")
    st.subheader("Informations principales")

    # Les KPI donnent une vision synthétique de la situation financière du client.
    values=[
        ("Identifiant",str(int(profile["SK_ID_CURR"]))),
        ("Revenu total",format_amount(profile.get("AMT_INCOME_TOTAL"))),
        ("Montant du crédit",format_amount(profile.get("AMT_CREDIT"))),
        ("Annuité",format_amount(profile.get("AMT_ANNUITY"))),
        ("Ratio annuité / crédit",format_percent(profile.get("PAYMENT_RATE"))),
        ("Ratio crédit / revenu",format_number(profile.get("CREDIT_INCOME_RATIO"))),
        ("Score externe moyen",format_number(profile.get("EXT_SOURCE_MEAN"))),
        ("Âge estimé",format_years_from_days(profile.get("DAYS_BIRTH"))),
        ("Ancienneté professionnelle",format_years_from_days(profile.get("DAYS_EMPLOYED"))),
        ("Retard moyen paiement",format_number(profile.get("INSTAL_DPD_MEAN"))),
        ("Retard max paiement",format_number(profile.get("INSTAL_DPD_MAX"))),
        ("Durée moyenne crédits précédents",format_number(profile.get("PREV_CNT_PAYMENT_MEAN")))
    ]
    for i in range(0,len(values),4):
        cols=st.columns(4)
        for col,(label,value) in zip(cols,values[i:i+4]): col.metric(label,value)

    # Cette analyse compare le client à une population réelle.
    st.subheader("Analyse métier")
    required_features=["AMT_INCOME_TOTAL","AMT_CREDIT","PAYMENT_RATE","EXT_SOURCE_MEAN","INSTAL_DPD_MEAN"]
    try:
        population=get_population(required_features)
        st.write(build_client_summary(profile,population))
    except Exception as error:
        st.warning(f"Impossible de calculer l'analyse métier : {error}")
    st.caption("Cette analyse descriptive situe le client par rapport à une population de référence. Elle ne remplace pas l'explication du modèle.")

# ==================================================
# ONGLET 3 — POSITIONNEMENT
# ==================================================
with tab_positioning:
    st.header("Positionnement")
    available_features=FEATURES_TO_COMPARE
    feature_labels=[meta["label"] for meta in available_features.values()]

    # ------------------------------
    # ANALYSE UNIVARIEE
    # ------------------------------
    st.subheader("Analyse univariée")
    selected_feature_label=st.selectbox("Choisir une variable à comparer",feature_labels,key="univariate_feature")
    selected_feature=next(f for f,m in available_features.items() if m["label"]==selected_feature_label)
    meta=available_features[selected_feature]
    client_value=profile.get(selected_feature)
    st.markdown(f"**{meta['label']}**")
    st.write(meta["description"])

    if pd.isna(client_value):
        st.warning("La valeur du client n'est pas disponible pour cette variable.")
    else:
        try:
            population=get_population([selected_feature])
            figure=plot_feature_distribution(population,selected_feature,meta["label"],client_value)
            if figure: st.plotly_chart(figure,use_container_width=True)
            st.write(build_distribution_summary(selected_feature,meta["label"],client_value,population,meta["format"]))
            st.dataframe(build_positioning_table(selected_feature,client_value,population,meta["format"]),use_container_width=True,hide_index=True)
        except Exception as error:
            st.error(f"Impossible de récupérer la population : {error}")

    # ------------------------------
    # ANALYSE BIVARIEE
    # ------------------------------
    st.subheader("Analyse bi-variée")
    col1,col2=st.columns(2)

    with col1:
        x_label=st.selectbox("Variable X",feature_labels,index=1 if len(feature_labels)>1 else 0,key="bivariate_x")
        x_feature=next(f for f,m in available_features.items() if m["label"]==x_label)
        x_meta=available_features[x_feature]
        st.markdown(f"**{x_meta['label']}**")
        st.write(x_meta["description"])

    with col2:
        y_label=st.selectbox("Variable Y",feature_labels,index=0,key="bivariate_y")
        y_feature=next(f for f,m in available_features.items() if m["label"]==y_label)
        y_meta=available_features[y_feature]
        st.markdown(f"**{y_meta['label']}**")
        st.write(y_meta["description"])

    client_x,client_y=profile.get(x_feature),profile.get(y_feature)

    if x_feature==y_feature:
        st.warning("Choisis deux variables différentes pour l'analyse bi-variée.")
    elif pd.isna(client_x) or pd.isna(client_y):
        st.warning("Les valeurs du client ne sont pas disponibles pour les deux variables sélectionnées.")
    else:
        try:
            population=get_population([x_feature,y_feature])
            figure=plot_bivariate_analysis(population,x_feature,y_feature,x_meta["label"],y_meta["label"],client_x,client_y)
            if figure: st.plotly_chart(figure,use_container_width=True)
            st.write(build_bivariate_analysis_summary(x_meta["label"],y_meta["label"],client_x,client_y,x_meta["format"],y_meta["format"]))
            col1,col2=st.columns(2)
            with col1:
                st.markdown(f"**{x_meta['label']}**")
                st.dataframe(build_positioning_table(x_feature,client_x,population,x_meta["format"]),use_container_width=True,hide_index=True)
            with col2:
                st.markdown(f"**{y_meta['label']}**")
                st.dataframe(build_positioning_table(y_feature,client_y,population,y_meta["format"]),use_container_width=True,hide_index=True)
        except Exception as error:
            st.error(f"Impossible de récupérer la population : {error}")

# ==================================================
# ONGLET 4 — SIMULATION
# ==================================================
with tab_simulation:
    st.header("Simulation")
    # Cette partie prépare l'évolution fonctionnelle prévue :
    # modifier les caractéristiques d'un client puis recalculer son score.
    st.info("La simulation avec de nouvelles données sera intégrée dans cette section.")
    st.write("Cette section permettra de modifier les informations du client puis de demander un nouveau score à l'API.")